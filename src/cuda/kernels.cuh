#pragma once
#include <cuda_runtime.h>
#include <curand_kernel.h>

// Atomic add para double (compatível com GPUs < sm_60 via CAS)
__device__ inline double atomicAddDouble(double* addr, double val) {
#if __CUDA_ARCH__ >= 600
    return atomicAdd(addr, val);
#else
    unsigned long long int* addr_ull = (unsigned long long int*)addr;
    unsigned long long int old = *addr_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(addr_ull, assumed,
            __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// T12: Distância euclidiana par-a-par (apenas para N <= 10k)
__global__ void pairwise_distance_kernel(
    const double* X, double* dist, int N, int F);

// T13: Visibilidade eta[i][j] = 1/dist[i][j]
__global__ void visibility_kernel(
    const double* dist, double* vis, int N);

// T14+T15: Construção de soluções (1 thread = 1 par formiga×instância)
// K*N threads: thread tid → ant=tid/N, inst=tid%N
// Decisão independente: P(select inst) = pheromone[inst] ∈ [0,1]
__global__ void ant_construction_kernel(
    const double* pheromone,
    int*          colony,
    double*       deposit,
    curandState*  rng_states,
    int N, int K);

// T16: Evaporação de feromônio tau[i] *= (1 - rho)
__global__ void pheromone_evaporation_kernel(
    double* pheromone, int N, double evap_rate);

// Aplica depósitos acumulados ao vetor de feromônio
__global__ void apply_deposit_kernel(
    double* pheromone, const double* deposit, int N, double Q);

// T19: 1-NN paralelo para avaliação de qualidade
__global__ void knn_1nn_kernel(
    const double* X_train, const int* selected_mask,
    const double* X_test,  int* predictions,
    int N, int F, int N_test, int n_classes);

// Inicialização do cuRAND state por thread
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int N);
