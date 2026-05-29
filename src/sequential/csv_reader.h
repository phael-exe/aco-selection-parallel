#pragma once

#include <cstddef>

// Dataset carregado na memória — X e Y como arrays contíguos row-major
// compatíveis com CUDA (coalesced memory access: X[i * F + j])
struct Dataset {
    double* X;       // features [N * F], row-major
    double* Y;       // labels   [N]
    size_t  N;       // instâncias válidas carregadas
    size_t  F;       // número de features (colunas - 1)
    char**  headers; // nomes das features (sem coluna-alvo), tamanho F
};

// Lê CSV com separador auto-detectado (';' ou ','), separa X e Y.
// target_column: nome da coluna-alvo (ex: "DEATH_EVENT", "Diabetes_binary")
// error_msg: buffer de saída para mensagem de erro (mínimo 256 bytes)
// Retorna Dataset com X == nullptr em caso de erro crítico.
Dataset read_csv(const char* filepath,
                 const char* target_column,
                 char*       error_msg);

// Detecta o separador dominante na primeira linha do arquivo (';' ou ',')
char detect_separator(const char* filepath);

// Libera toda a memória alocada por read_csv
void free_dataset(Dataset* ds);
