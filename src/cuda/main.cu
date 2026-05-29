#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "memory.cuh"
#include "kernels.cuh"

using namespace std;

// Lê CSV separado por ';', descarta header, retorna linhas como double[]
// F_out = número de colunas (features + classe)
static vector<vector<double>> read_csv(const string& path, int& F_out) {
    ifstream file(path);
    if (!file) { cerr << "Erro ao abrir: " << path << "\n"; exit(1); }
    vector<vector<double>> rows;
    string line;
    bool header = true;
    F_out = 0;
    while (getline(file, line)) {
        if (line.empty()) continue;
        if (header) {
            stringstream ss(line);
            string cell;
            while (getline(ss, cell, ';')) F_out++;
            header = false;
            continue;
        }
        vector<double> row;
        row.reserve(F_out);
        stringstream ss(line);
        string cell;
        while (getline(ss, cell, ';')) {
            try { row.push_back(stod(cell)); }
            catch (...) { row.push_back(0.0); }
        }
        if ((int)row.size() == F_out)
            rows.push_back(move(row));
    }
    return rows;
}

static void write_solutions_csv(const vector<int>& colony, int N, const string& path) {
    ofstream out(path);
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            out << colony[(long long)i * N + j];
            if (j + 1 < N) out << ";";
        }
        out << "\n";
    }
    cout << "Resultado salvo em " << path << "\n";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Uso: " << argv[0] << " <dataset_name> [data_dir]\n";
        cerr << "Ex:  " << argv[0] << " heart_failure data/baseline\n";
        return 1;
    }

    string dataset  = argv[argc - 1];
    string data_dir = (argc >= 3) ? argv[argc - 2] : "data/baseline";
    string path     = data_dir + "/" + dataset + ".csv";

    // Parâmetros ACO
    const double initial_pheromone = 1.0;
    const double evaporation_rate  = 0.1;
    const double Q                 = 1.0;
    const int    BLOCK             = 256;

    // Leitura do dataset
    int F = 0;
    auto rows = read_csv(path, F);
    int N = (int)rows.size();
    printf("Dataset: %s | N=%d F=%d\n", path.c_str(), N, F);

    if (N == 0) { cerr << "Dataset vazio!\n"; return 1; }

    // Montar array flat [N * F]
    vector<double> h_X((long long)N * F);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < F; ++j)
            h_X[(long long)i * F + j] = rows[i][j];
    rows.clear();  // libera memória host

    // Estratégia de memória
    bool onthefly = (N > 10000);
    printf("Estrategia: %s\n", onthefly ? "on-the-fly (N>10k)" : "pre-computado");

    // Alocar GPU
    GpuBuffers buf = alloc_gpu(N, F, onthefly);

    // Upload X
    cudaEvent_t ev0, ev1; float ms;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(buf.d_X, h_X.data(),
                          (size_t)N * F * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    printf("Transferencia H->D X: %.2f ms\n", ms);

    // Inicializar feromônio
    vector<double> h_phe(N, initial_pheromone);
    CUDA_CHECK(cudaMemcpy(buf.d_pheromone, h_phe.data(),
                          N * sizeof(double), cudaMemcpyHostToDevice));

    // Inicializar colônia: -1 exceto diagonal = 1
    vector<int> h_colony((long long)N * N, -1);
    for (int i = 0; i < N; ++i) h_colony[(long long)i * N + i] = 1;
    CUDA_CHECK(cudaMemcpy(buf.d_colony, h_colony.data(),
                          (long long)N * N * sizeof(int), cudaMemcpyHostToDevice));

    // Pré-computar distâncias e visibilidades se não onthefly
    if (!onthefly) {
        // Inicializar diagonal de dist a 0
        CUDA_CHECK(cudaMemset(buf.d_dist, 0, (long long)N * N * sizeof(double)));

        long long total_pairs = (long long)N * (N - 1) / 2;
        int grid_dist = (int)((total_pairs + BLOCK - 1) / BLOCK);
        pairwise_distance_kernel<<<grid_dist, BLOCK>>>(buf.d_X, buf.d_dist, N, F);
        CUDA_CHECK(cudaDeviceSynchronize());

        int grid_vis = (int)(((long long)N * N + BLOCK - 1) / BLOCK);
        visibility_kernel<<<grid_vis, BLOCK>>>(buf.d_dist, buf.d_vis, N);
        CUDA_CHECK(cudaDeviceSynchronize());
        printf("Distancias e visibilidades pre-computadas na GPU.\n");
    }

    // Inicializar cuRAND
    curandState* d_rng;
    CUDA_CHECK(cudaMalloc(&d_rng, (long long)N * sizeof(curandState)));
    int grid_rng = (N + BLOCK - 1) / BLOCK;
    init_curand_kernel<<<grid_rng, BLOCK>>>(d_rng, 42ULL, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Buffer de depósito de feromônio
    double* d_deposit;
    CUDA_CHECK(cudaMalloc(&d_deposit, N * sizeof(double)));

    // Loop principal do ACO
    int grid_ants = (N + BLOCK - 1) / BLOCK;
    int grid_phe  = (N + BLOCK - 1) / BLOCK;

    CUDA_CHECK(cudaEventRecord(ev0));

    bool colony_complete = false;
    int  max_iters = N * 2 + 100;  // safety bound
    int  iter = 0;

    while (!colony_complete && iter < max_iters) {
        // Zerar buffer de depósito
        CUDA_CHECK(cudaMemset(d_deposit, 0, N * sizeof(double)));

        // Construção de soluções + depósito
        ant_construction_kernel<<<grid_ants, BLOCK>>>(
            buf.d_X, buf.d_dist, buf.d_vis,
            buf.d_pheromone, buf.d_colony,
            d_deposit, d_rng,
            N, F, (int)onthefly);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Aplicar depósito + evaporação
        apply_deposit_kernel<<<grid_phe, BLOCK>>>(
            buf.d_pheromone, d_deposit, N, Q);
        pheromone_evaporation_kernel<<<grid_phe, BLOCK>>>(
            buf.d_pheromone, N, evaporation_rate);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Verificar colônia completa via sample: copiar e checar
        // (para N grande, checar apenas uma amostra das formigas para performance)
        int check_n = (N < 1000) ? N : 1000;
        vector<int> h_sample((long long)check_n * N);
        CUDA_CHECK(cudaMemcpy(h_sample.data(), buf.d_colony,
                              (long long)check_n * N * sizeof(int),
                              cudaMemcpyDeviceToHost));
        colony_complete = true;
        for (int x : h_sample) {
            if (x < 0) { colony_complete = false; break; }
        }
        ++iter;
    }

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    printf("Tempo compute GPU: %.2f ms (%d iters)\n", ms, iter);

    // Download resultado final
    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(h_colony.data(), buf.d_colony,
                          (long long)N * N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    printf("Transferencia D->H colony: %.2f ms\n", ms);

    // Salvar resultado
    write_solutions_csv(h_colony, N, "results/solutions_cuda.csv");

    // Cleanup
    free_gpu(buf);
    CUDA_CHECK(cudaFree(d_rng));
    CUDA_CHECK(cudaFree(d_deposit));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    return 0;
}
