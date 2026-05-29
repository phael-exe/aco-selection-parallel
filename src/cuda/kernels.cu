#include "kernels.cuh"
#include <math.h>

// T12 — Distância euclidiana par-a-par
// Cada thread calcula uma entrada (i,j) do triângulo inferior e espelha em (j,i)
// idx mapeia triângulo: row > col, row em [1..N-1], col em [0..row-1]
__global__ void pairwise_distance_kernel(
    const double* X, double* dist, int N, int F)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long total = (long long)N * (N - 1) / 2;
    if (idx >= total) return;

    // Mapear idx para (row, col) do triângulo inferior (row > col)
    // row é o maior inteiro tal que row*(row-1)/2 <= idx
    int row = (int)((1.0 + sqrt(1.0 + 8.0 * (double)idx)) / 2.0);
    // ajuste fino caso a aproximação de sqrt dê row errado
    while ((long long)row * (row - 1) / 2 > idx) row--;
    while ((long long)(row + 1) * row / 2 <= idx) row++;
    int col = (int)(idx - (long long)row * (row - 1) / 2);

    if (row >= N || col >= row) return;

    double sum = 0.0;
    for (int k = 0; k < F; ++k) {
        double diff = X[(long long)row * F + k] - X[(long long)col * F + k];
        sum += diff * diff;
    }
    double d = sqrt(sum);
    dist[(long long)row * N + col] = d;
    dist[(long long)col * N + row] = d;
}

// T13 — Visibilidade eta[i][j] = 1/dist[i][j]
__global__ void visibility_kernel(
    const double* dist, double* vis, int N)
{
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long long)N * N) return;
    int i = (int)(idx / N);
    int j = (int)(idx % N);
    if (i == j) {
        vis[idx] = 0.0;
    } else {
        double d = dist[idx];
        vis[idx] = (d == 0.0) ? 0.0 : 1.0 / d;
    }
}

// T16 — Evaporação: tau[i] *= (1 - rho)
__global__ void pheromone_evaporation_kernel(
    double* pheromone, int N, double evap_rate)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    pheromone[i] *= (1.0 - evap_rate);
}

// Aplica depósitos acumulados ao vetor de feromônio
__global__ void apply_deposit_kernel(
    double* pheromone, const double* deposit, int N, double Q)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    pheromone[i] += deposit[i];  // Q já é incorporado no cálculo do depósito
}

// T19 — 1-NN paralelo
// Cada thread avalia uma instância de teste (índice t)
// X_train e X_test têm F colunas (F-1 features + 1 classe na última coluna)
// selected_mask[j] = 1 se instância j foi selecionada
__global__ void knn_1nn_kernel(
    const double* X_train, const int* selected_mask,
    const double* X_test,  int* predictions,
    int N, int F, int N_test, int n_classes)
{
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N_test) return;

    double best_dist = 1e300;
    int    best_label = 0;

    for (int j = 0; j < N; ++j) {
        if (selected_mask[j] != 1) continue;
        double sum = 0.0;
        for (int k = 0; k < F - 1; ++k) {
            double d = X_test[(long long)t * F + k] - X_train[(long long)j * F + k];
            sum += d * d;
        }
        if (sum < best_dist) {
            best_dist = sum;
            best_label = (int)X_train[(long long)j * F + (F - 1)];
        }
    }
    predictions[t] = best_label;
}

// Inicializa estado cuRAND por thread
__global__ void init_curand_kernel(curandState* states, unsigned long long seed, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    curand_init(seed, tid, 0, &states[tid]);
}
