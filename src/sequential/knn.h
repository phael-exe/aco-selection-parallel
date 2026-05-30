#pragma once

#include <cstddef>

// Classificador 1-NN (1-Nearest Neighbor) brute-force
// Prediz a classe de cada instância de teste encontrando o vizinho mais próximo no treino

// Predições 1-NN para dataset de teste
// X_train: features de treino [N_train * F], row-major
// Y_train: labels de treino [N_train]
// N_train: número de instâncias de treino
// X_test: features de teste [N_test * F], row-major
// N_test: número de instâncias de teste
// F: número de features
// Retorna: array [N_test] com predições (Y_train[vizinho_mais_próximo])
// Nota: Alocação dinâmica — caller é responsável por free() o resultado
double* knn_1nn_predict(
    const double* X_train, const double* Y_train, size_t N_train,
    const double* X_test, size_t N_test,
    size_t F
);
