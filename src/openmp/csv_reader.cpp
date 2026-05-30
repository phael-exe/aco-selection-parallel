#include "csv_reader.h"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <cstdint>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Helpers internos
// ---------------------------------------------------------------------------

static std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, last - first + 1);
}

static std::vector<std::string> split_line(const std::string& line, char delim) {
    std::vector<std::string> fields;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, delim)) {
        fields.push_back(field);
    }
    return fields;
}

// Retorna true se o campo representa um valor ausente ou inválido conhecido
static bool is_invalid_field(const std::string& f) {
    return f.empty()          ||
           f == "NA"          || f == "na"          || f == "N/A"  || f == "n/a"  ||
           f == "null"        || f == "NULL"         ||
           f == "nan"         || f == "NaN"          || f == "NAN"         ||
           f == "inf"         || f == "Inf"          || f == "INF"         ||
           f == "-inf"        || f == "-Inf"         || f == "-INF"        ||
           f == "missing"     || f == "MISSING"      || f == "?"           ||
           f == "#N/A"        || f == "#NA"          || f == "None"        ||
           f == "none";
}

// ---------------------------------------------------------------------------
// detect_separator
// ---------------------------------------------------------------------------

char detect_separator(const char* filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) return ';';

    std::string first_line;
    std::getline(file, first_line);

    size_t n_semi  = std::count(first_line.begin(), first_line.end(), ';');
    size_t n_comma = std::count(first_line.begin(), first_line.end(), ',');

    return (n_comma > n_semi) ? ',' : ';';
}

// ---------------------------------------------------------------------------
// read_csv
// ---------------------------------------------------------------------------

Dataset read_csv(const char* filepath,
                 const char* target_column,
                 char*       error_msg) {
    Dataset ds{nullptr, nullptr, 0, 0, nullptr};

    // ----- Abrir arquivo -----
    std::ifstream file(filepath);
    if (!file.is_open()) {
        snprintf(error_msg, 256, "arquivo nao encontrado: %s", filepath);
        return ds;
    }

    // ----- Detectar separador -----
    char sep = detect_separator(filepath);

    // ----- Parse do header -----
    std::string header_line;
    if (!std::getline(file, header_line)) {
        snprintf(error_msg, 256, "arquivo vazio: %s", filepath);
        return ds;
    }

    std::vector<std::string> columns = split_line(header_line, sep);
    for (auto& c : columns) c = trim(c);

    size_t num_cols = columns.size();
    if (num_cols < 2) {
        snprintf(error_msg, 256,
            "CSV deve ter pelo menos 2 colunas, encontrado: %zu", num_cols);
        return ds;
    }

    // Encontrar índice da coluna-alvo
    size_t target_col_idx = SIZE_MAX;
    for (size_t i = 0; i < num_cols; ++i) {
        if (columns[i] == std::string(target_column)) {
            target_col_idx = i;
            break;
        }
    }
    if (target_col_idx == SIZE_MAX) {
        snprintf(error_msg, 256,
            "coluna '%s' nao encontrada no header", target_column);
        return ds;
    }

    size_t F = num_cols - 1;  // número de features

    // ----- Passagem 1: contar linhas de dados não-vazias -----
    size_t total_data_lines = 0;
    {
        std::string line;
        while (std::getline(file, line)) {
            if (!trim(line).empty()) total_data_lines++;
        }
    }

    if (total_data_lines == 0) {
        snprintf(error_msg, 256, "CSV sem linhas de dados: %s", filepath);
        return ds;
    }

    // ----- Alocar memória -----
    double* X = new double[total_data_lines * F];
    double* Y = new double[total_data_lines];

    // ----- Passagem 2: preencher X e Y com validação numérica -----
    file.clear();
    file.seekg(0);
    std::getline(file, header_line);  // pula header

    size_t valid_idx     = 0;
    size_t invalid_count = 0;

    // Buffer reutilizável para features de uma linha (evita realocar por linha)
    std::vector<double> x_buf(F);

    std::string line;
    while (std::getline(file, line)) {
        std::string line_t = trim(line);
        if (line_t.empty()) continue;

        std::vector<std::string> fields = split_line(line_t, sep);

        // Número de colunas inconsistente
        if (fields.size() != num_cols) {
            invalid_count++;
            continue;
        }

        bool   line_valid = true;
        double y_val      = 0.0;
        size_t feat_idx   = 0;

        for (size_t j = 0; j < num_cols && line_valid; ++j) {
            std::string fld = trim(fields[j]);

            // Nível 2 — Validação Numérica
            if (is_invalid_field(fld)) {
                invalid_count++;
                line_valid = false;
                break;
            }

            try {
                double val = std::stod(fld);

                // stod pode retornar NaN/Inf em alguns casos de borda
                if (std::isnan(val) || std::isinf(val)) {
                    invalid_count++;
                    line_valid = false;
                    break;
                }

                if (j == target_col_idx) {
                    y_val = val;
                } else {
                    x_buf[feat_idx++] = val;
                }
            } catch (const std::exception&) {
                invalid_count++;
                line_valid = false;
            }
        }

        if (line_valid) {
            Y[valid_idx] = y_val;
            for (size_t j = 0; j < F; ++j) {
                X[valid_idx * F + j] = x_buf[j];
            }
            valid_idx++;
        }
    }

    // ----- Verificar tolerância de erro -----
    size_t total_processed = valid_idx + invalid_count;
    double error_rate = (total_processed > 0)
        ? (static_cast<double>(invalid_count) / static_cast<double>(total_processed)) * 100.0
        : 0.0;

    if (error_rate > 5.0) {
        snprintf(error_msg, 256,
            "Dataset corrompido: %.1f%% linhas invalidas (limite: 5%%) — %zu rejeitadas",
            error_rate, invalid_count);
        delete[] X;
        delete[] Y;
        return ds;  // ds.X ainda é nullptr
    }

    if (error_rate > 1.0) {
        fprintf(stderr,
            "Aviso: %.1f%% linhas rejeitadas (%zu de %zu) — mantendo %zu validas\n",
            error_rate, invalid_count, total_processed, valid_idx);
    }

    // ----- Montar array de headers de features (sem coluna-alvo) -----
    char** headers = new char*[F];
    size_t h = 0;
    for (size_t i = 0; i < num_cols; ++i) {
        if (i == target_col_idx) continue;
        headers[h] = new char[columns[i].size() + 1];
        std::strcpy(headers[h], columns[i].c_str());
        h++;
    }

    ds.X       = X;
    ds.Y       = Y;
    ds.N       = valid_idx;
    ds.F       = F;
    ds.headers = headers;
    return ds;
}

// ---------------------------------------------------------------------------
// free_dataset
// ---------------------------------------------------------------------------

void free_dataset(Dataset* ds) {
    if (!ds) return;
    delete[] ds->X;
    delete[] ds->Y;
    if (ds->headers) {
        for (size_t i = 0; i < ds->F; ++i) {
            delete[] ds->headers[i];
        }
        delete[] ds->headers;
    }
    ds->X       = nullptr;
    ds->Y       = nullptr;
    ds->headers = nullptr;
    ds->N       = 0;
    ds->F       = 0;
}
