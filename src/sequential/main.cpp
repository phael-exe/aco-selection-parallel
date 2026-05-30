#include <iostream>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include "csv_reader.h"
#include "aco.h"

static void print_usage(const char* prog) {
    std::cerr << "Uso: " << prog
              << " <dataset_path> <coluna_alvo>"
              << " [--ants K] [--iter N] [--evap R] [--Q V] [--eval-top-k K]\n\n"
              << "Exemplos:\n"
              << "  " << prog << " data/baseline/heart_failure.csv DEATH_EVENT\n"
              << "  " << prog << " data/baseline/heart_failure.csv DEATH_EVENT --ants 64 --iter 100\n"
              << "  " << prog << " data/cdc/cdc_diabetes.csv Diabetes_binary --evap 0.1 --Q 1 --eval-top-k 1\n";
}

int main(int argc, char* argv[]) {
    // Argumentos obrigatórios: path e coluna-alvo
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }

    const char* path       = argv[1];
    const char* target_col = argv[2];

    // Defaults dos parâmetros ACO
    int    num_ants         = 64;
    int    max_iterations   = 100;
    double evaporation_rate = 0.1;
    int    Q                = 1;
    int    eval_top_k_explicit  = 0;  // 0 = não especificado (auto-detectar)
    int    eval_sample_explicit = -1;  // -1 = não especificado (auto-detectar)

    // Parse de argumentos opcionais (--flag valor, aos pares)
    for (int i = 3; i + 1 < argc; i += 2) {
        if (std::strcmp(argv[i], "--ants") == 0) {
            num_ants = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--iter") == 0) {
            max_iterations = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--evap") == 0) {
            evaporation_rate = std::atof(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--Q") == 0) {
            Q = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--eval-top-k") == 0) {
            eval_top_k_explicit = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--eval-sample") == 0) {
            eval_sample_explicit = std::atoi(argv[i + 1]);
        } else {
            std::cerr << "Aviso: argumento desconhecido '" << argv[i] << "' ignorado\n";
        }
    }

    // Leitura do dataset com medição de tempo
    char error_msg[256] = {};
    auto t0 = std::chrono::high_resolution_clock::now();
    Dataset ds = read_csv(path, target_col, error_msg);
    auto t1 = std::chrono::high_resolution_clock::now();

    if (!ds.X) {
        std::cerr << "Erro ao ler CSV: " << error_msg << "\n";
        return 1;
    }

    double elapsed_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "Dataset carregado: " << ds.N << " instancias, "
              << ds.F << " features\n";
    std::cout << "Tempo de leitura:  " << elapsed_ms << " ms\n";
    std::cout << "Parametros ACO:    ants=" << num_ants
              << ", iter=" << max_iterations
              << ", evap=" << evaporation_rate
              << ", Q=" << Q << "\n\n";

    // ===== Executar ACO =====
    ACOConfig config{};
    config.K        = static_cast<size_t>(num_ants);
    config.max_iter = static_cast<size_t>(max_iterations);
    config.rho      = evaporation_rate;
    config.Q        = Q;
    config.patience = 10;  // default
    
    // Auto-detectar eval_top_k baseado em N (se não especificado)
    if (eval_top_k_explicit == 0) {
        if (ds.N <= 1000) {
            config.eval_top_k = 5;
        } else if (ds.N <= 10000) {
            config.eval_top_k = 3;
        } else {
            config.eval_top_k = 1;
        }
    } else {
        config.eval_top_k = static_cast<size_t>(eval_top_k_explicit);
    }

    // Auto-detectar eval_sample baseado em N (se não especificado)
    if (eval_sample_explicit == -1) {
        config.eval_sample = (ds.N > 10000) ? 5000 : 0;
    } else {
        config.eval_sample = static_cast<size_t>(eval_sample_explicit);
    }

    auto t_aco_0 = std::chrono::high_resolution_clock::now();
    ACOResult result = run_aco(ds.X, ds.Y, ds.N, ds.F, config);
    auto t_aco_1 = std::chrono::high_resolution_clock::now();

    double aco_time_ms = std::chrono::duration<double, std::milli>(t_aco_1 - t_aco_0).count();

    // Exibir resultado
    std::cout << "\n=== Resultado ACO ===\n";
    std::cout << "Melhor solucao: " << result.selected << "/" << ds.N
              << " instancias (reducao ~" << (100.0 * (ds.N - result.selected) / ds.N) << "%)\n";
    std::cout << "Acuracia: " << result.accuracy << "\n";
    std::cout << "Precisao: " << result.precision << "\n";
    std::cout << "Recall: " << result.recall << "\n";
    std::cout << "F1-Score: " << result.best_fitness << "\n";
    std::cout << "Iteracoes executadas: " << result.iterations << "/" << config.max_iter;
    if (result.iterations < config.max_iter) {
        std::cout << " (early stop)\n";
    } else {
        std::cout << "\n";
    }
    std::cout << "Tempo ACO: " << aco_time_ms << " ms\n";
    std::cout << "Eval strategy: top-" << config.eval_top_k << " formigas/iteracao (";
    if (eval_top_k_explicit == 0) {
        std::cout << "auto-detectado para N=" << ds.N << ")\n";
    } else {
        std::cout << "explicitamente configurado)\n";
    }
    if (config.eval_sample > 0) {
        std::cout << "Eval sample: " << config.eval_sample << "/" << ds.N
                  << " instancias por avaliacao (";
        std::cout << (eval_sample_explicit == -1 ? "auto-detectado" : "explicitamente configurado") << ")\n";
    }

    free_dataset(&ds);
    free_aco_result(&result);
    return 0;
}
