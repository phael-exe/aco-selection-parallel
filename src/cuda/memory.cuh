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

struct TransferTiming {
    float h2d_ms;
    float d2h_ms;
};

struct GpuBuffers {
    double* d_X;          // [N * F] features
    double* d_dist;       // [N * N] distâncias (NULL se on-the-fly)
    double* d_vis;        // [N * N] visibilidades (NULL se on-the-fly)
    double* d_pheromone;  // [N] vetor de feromônio 1D
    int*    d_colony;     // [N * N] colônia (solução)
    int     N;
    int     F;
    bool    onthefly;     // true se N > 10000
};

inline GpuBuffers alloc_gpu(int N, int F, bool onthefly) {
    GpuBuffers buf;
    buf.N = N; buf.F = F; buf.onthefly = onthefly;
    CUDA_CHECK(cudaMalloc(&buf.d_X,         (size_t)N * F * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf.d_pheromone, (size_t)N     * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&buf.d_colony,    (size_t)N * N * sizeof(int)));
    if (!onthefly) {
        CUDA_CHECK(cudaMalloc(&buf.d_dist, (size_t)N * N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&buf.d_vis,  (size_t)N * N * sizeof(double)));
    } else {
        buf.d_dist = nullptr;
        buf.d_vis  = nullptr;
    }
    return buf;
}

inline void free_gpu(GpuBuffers& buf) {
    CUDA_CHECK(cudaFree(buf.d_X));
    CUDA_CHECK(cudaFree(buf.d_pheromone));
    CUDA_CHECK(cudaFree(buf.d_colony));
    if (buf.d_dist) CUDA_CHECK(cudaFree(buf.d_dist));
    if (buf.d_vis)  CUDA_CHECK(cudaFree(buf.d_vis));
}

inline TransferTiming upload_X(GpuBuffers& buf, const double* h_X) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(buf.d_X, h_X,
                          (size_t)buf.N * buf.F * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    TransferTiming t; t.d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return t;
}

inline float download_colony(const GpuBuffers& buf, int* h_colony) {
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    CUDA_CHECK(cudaMemcpy(h_colony, buf.d_colony,
                          (size_t)buf.N * buf.N * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}
