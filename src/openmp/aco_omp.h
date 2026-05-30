#pragma once

#include <cstddef>

// Configuração do algoritmo ACO (versão OpenMP)
struct ACOConfig {
    size_t K;           // formigas (default 128)
    size_t max_iter;    // iterações (default 100)
    double rho;         // taxa evaporação (default 0.1)
    double Q;           // fator depósito (default 1.0)
    size_t patience;    // early stopping (default 10)
    size_t eval_top_k;  // quantas formigas avaliar por iteração com 1-NN (default: auto-detectado em main.cpp)
};

// Resultado do algoritmo ACO
struct ACOResult {
    int*    best_solution;  // binário [N]: 1=selecionado, 0=não
    size_t  selected;       // instâncias selecionadas
    double  best_fitness;   // F1-score real (não mais proxy)
    double  accuracy;       // acurácia da melhor solução
    double  precision;      // precisão da melhor solução
    double  recall;         // recall da melhor solução
    size_t  iterations;     // iterações executadas
    double  time_seconds;   // tempo total
};

// Função principal: executa o algoritmo ACO paralelizado com OpenMP
// X: features [N*F] row-major
// Y: labels [N]
// N: número de instâncias
// F: número de features
// config: parâmetros do ACO
// Retorna ACOResult com melhor solução encontrada
ACOResult run_aco(const double* X, const double* Y, size_t N, size_t F, ACOConfig config);

// Libera memória do ACOResult
void free_aco_result(ACOResult* result);
