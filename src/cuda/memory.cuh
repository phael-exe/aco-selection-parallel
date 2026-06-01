#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

struct GpuBuffers {
    double* d_X;             // [N * F] features (sem coluna-alvo)
    double* d_Y;             // [N] labels (para knn_1nn_kernel na GPU)
    double* d_pheromone;     // [N] vetor de feromônio
    double* d_vis;           // [N] visibilidade 1D (η[i] = 1/(1+avg_dist[i]))
    double* d_select_prob;   // [N] prob. de seleção (τ^α·η^β/max)
    double* d_deposit;       // [N] acumulador de depósito de feromônio
    int*    d_colony;        // [K * N] colônia (K formigas × N instâncias)
    int*    d_selected_count;// [K] contagem de instâncias selecionadas por formiga
    int*    d_predictions;   // [N] predições do 1-NN na GPU
    int     N, K, F;
};

inline GpuBuffers alloc_gpu(int N, int K, int F) {
    GpuBuffers buf;
    buf.N = N; buf.K = K; buf.F = F;
    CUDA_CHECK(cudaMalloc(&buf.d_X,              (size_t)N * F * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf.d_Y,              (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf.d_pheromone,      (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf.d_vis,            (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf.d_select_prob,    (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf.d_deposit,        (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf.d_colony,         (size_t)K * N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_selected_count, (size_t)K     * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&buf.d_predictions,    (size_t)N     * sizeof(int)));
    return buf;
}

inline void free_gpu(GpuBuffers& buf) {
    CUDA_CHECK(cudaFree(buf.d_X));
    CUDA_CHECK(cudaFree(buf.d_Y));
    CUDA_CHECK(cudaFree(buf.d_pheromone));
    CUDA_CHECK(cudaFree(buf.d_vis));
    CUDA_CHECK(cudaFree(buf.d_select_prob));
    CUDA_CHECK(cudaFree(buf.d_deposit));
    CUDA_CHECK(cudaFree(buf.d_colony));
    CUDA_CHECK(cudaFree(buf.d_selected_count));
    CUDA_CHECK(cudaFree(buf.d_predictions));
}
