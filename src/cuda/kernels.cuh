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

// Distância média de cada instância às demais: avg_dist[i] = mean_{j≠i} euclidean(i,j)
// Usado uma vez para computar visibilidade 1D
__global__ void avg_distance_kernel(const double* X, double* avg_dist, int N, int F);

// Construção paralela: 1 thread por par (formiga, instância)
// select_prob[N] é τ^α·η^β normalizado pelo máximo (precomputado na CPU a cada iteração)
// selected_count[K]: contador atômico de quantas instâncias cada formiga selecionou
__global__ void ant_construction_kernel(
    const double* select_prob,
    int*          colony,
    int*          selected_count,
    curandState*  rng_states,
    int N, int K);

// Depósito: para cada (k,i) selecionado atomicamente acumula Q/selected_count[k] em deposit[i]
__global__ void deposit_kernel(
    const int*    colony,
    const int*    selected_count,
    double*       deposit,
    int N, int K, double Q);

// Atualização de feromônio: tau = (tau + deposit) * (1-rho), clamp a tau_min
// Ordem equivale a: depositar primeiro, depois evaporar (igual ao sequencial)
__global__ void apply_pheromone_kernel(
    double* pheromone, const double* deposit, int N, double rho);

// Inicialização dos estados cuRAND
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int N);

// 1-NN paralelo para avaliação de qualidade
// N_test threads: cada thread avalia 1 instância de teste contra o conjunto de treino
// selected_mask[j] == 1 → instância j faz parte do treino
// X[N*F] contém só features; Y[N] contém labels
__global__ void knn_1nn_kernel(
    const double* X, const double* Y,
    const int*    selected_mask,
    int*          predictions,
    int N, int F, int N_test);
