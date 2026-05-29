#include "knn.h"

#include <cmath>
#include <algorithm>
#include <vector>

// ---------------------------------------------------------------------------
// Helper: Distância Euclidiana entre 2 pontos
// ---------------------------------------------------------------------------

static double compute_distance_single(const double* x1, const double* x2, size_t F) {
    double sum = 0.0;
    for (size_t k = 0; k < F; ++k) {
        double diff = x1[k] - x2[k];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

// ---------------------------------------------------------------------------
// Classificador 1-NN Brute-Force
// ---------------------------------------------------------------------------

double* knn_1nn_predict(
    const double* X_train, const double* Y_train, size_t N_train,
    const double* X_test, size_t N_test,
    size_t F) {
    
    // Alocar vetor de predições
    double* Y_pred = new double[N_test];
    
    // Para cada instância de teste
    for (size_t j = 0; j < N_test; ++j) {
        const double* x_test_j = X_test + j * F;
        
        // Encontrar vizinho mais próximo em X_train
        double min_distance = 1e308;  // inicializa com infinito
        size_t nearest_idx = 0;
        
        for (size_t i = 0; i < N_train; ++i) {
            const double* x_train_i = X_train + i * F;
            double distance = compute_distance_single(x_test_j, x_train_i, F);
            
            if (distance < min_distance) {
                min_distance = distance;
                nearest_idx = i;
            }
        }
        
        // Predição = label do vizinho mais próximo
        Y_pred[j] = Y_train[nearest_idx];
    }
    
    return Y_pred;
}
