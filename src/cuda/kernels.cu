#include "kernels.cuh"
#include <math.h>

// Distância média de cada instância às demais
// Cada thread i calcula mean_{j≠i} ||X_i - X_j||
__global__ void avg_distance_kernel(const double* X, double* avg_dist, int N, int F) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
        if (i == j) continue;
        double d = 0.0;
        for (int k = 0; k < F; ++k) {
            double diff = X[(long long)i * F + k] - X[(long long)j * F + k];
            d += diff * diff;
        }
        sum += sqrt(d);
    }
    avg_dist[i] = (N > 1) ? sum / (N - 1) : 0.0;
}

// Construção paralela: 1 thread por par (formiga k, instância i)
// Decide inclusão com probabilidade select_prob[i] (precomputado τ^α·η^β/max)
__global__ void ant_construction_kernel(
    const double* select_prob,
    int*          colony,
    int*          selected_count,
    curandState*  rng_states,
    int N, int K)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= K * N) return;

    int ant  = tid / N;
    int inst = tid % N;

    curandState local = rng_states[tid];
    double p = select_prob[inst];
    int selected = (curand_uniform_double(&local) < p) ? 1 : 0;
    rng_states[tid] = local;

    colony[(long long)ant * N + inst] = selected;

    if (selected)
        atomicAdd(&selected_count[ant], 1);
}

// Depósito: cada thread (k,i) selecionado adiciona Q/tour_length[k] ao deposit[i]
__global__ void deposit_kernel(
    const int*    colony,
    const int*    selected_count,
    double*       deposit,
    int N, int K, double Q)
{
    int tid  = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= K * N) return;

    int ant  = tid / N;
    int inst = tid % N;

    if (colony[(long long)ant * N + inst] == 1) {
        int cnt = selected_count[ant];
        if (cnt > 0)
            atomicAddDouble(&deposit[inst], Q / (double)cnt);
    }
}

// Atualiza feromônio: deposita primeiro, depois evapora (igual ao sequencial)
// tau_new = (tau + deposit) * (1 - rho)
__global__ void apply_pheromone_kernel(
    double* pheromone, const double* deposit, int N, double rho)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    double tau_new = (pheromone[i] + deposit[i]) * (1.0 - rho);
    if (tau_new < 1e-10) tau_new = 1e-10;
    pheromone[i] = tau_new;
}

// Inicializa cuRAND: 1 estado por thread, seed fixo + sequência diferente por tid
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    curand_init(seed, tid, 0, &states[tid]);
}

// 1-NN paralelo: N_test threads, cada um encontra o vizinho mais próximo no conjunto de treino
// X[N*F] — features; Y[N] — labels; selected_mask[N] — 1 = treino
__global__ void knn_1nn_kernel(
    const double* X, const double* Y,
    const int*    selected_mask,
    int*          predictions,
    int N, int F, int N_test)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N_test) return;

    double best_d = 1e300;
    double pred   = 0.0;

    for (int j = 0; j < N; ++j) {
        if (selected_mask[j] != 1) continue;
        double d = 0.0;
        for (int k = 0; k < F; ++k) {
            double diff = X[(long long)t * F + k] - X[(long long)j * F + k];
            d += diff * diff;
        }
        if (d < best_d) { best_d = d; pred = Y[j]; }
    }
    predictions[t] = (int)pred;
}
