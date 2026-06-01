#pragma once

#include <cstddef>

// Métricas de qualidade da seleção de instâncias
// Avalia uma solução (vetor binário) usando 1-NN contra dataset completo

struct QualityMetrics {
    double accuracy;        // Acurácia = (TP + TN) / Total
    double precision;       // Binário: TP/(TP+FP), Multiclasse: macro-average
    double recall;          // Binário: TP/(TP+FN), Multiclasse: macro-average
    double f1_score;        // F1 = 2 * (P * R) / (P + R)
    double reduction_rate;  // Taxa de redução = 1 - (selecionadas / N)
    size_t selected_count;  // Número de instâncias selecionadas
};

// Avalia a qualidade de uma solução (vetor binário de seleção)
// solution: [N] — 1=selecionada, 0=não selecionada
// X: features [N*F], row-major
// Y: labels [N]
// N: total de instâncias
// F: número de features
// Retorna: struct QualityMetrics com todas as métricas calculadas
//
// Algoritmo:
// 1. Seleciona instâncias de treino: X_train = X[i] where solution[i]==1
// 2. Treina 1-NN com X_train, Y_train
// 3. Avalia 1-NN no dataset completo (X, Y)
// 4. Calcula acurácia, P, R, F1 (binário ou multiclasse auto-detectado)
// eval_sample: se > 0 e N > eval_sample, avalia apenas uma amostra aleatória do test set
QualityMetrics evaluate_solution(
    const int* solution,
    const double* X, const double* Y, size_t N, size_t F,
    size_t eval_sample = 0
);
