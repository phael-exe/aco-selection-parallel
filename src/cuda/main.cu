#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "memory.cuh"
#include "kernels.cuh"

using namespace std;

struct EvalMetrics { double f1, acc, precision, recall, reduction; int selected; };

// Avalia solução com 1-NN em amostra (eval_sample=0 → tudo)
static EvalMetrics evaluate_solution_cpu(
    const vector<int>& solution,
    const vector<double>& h_X, int N, int F,
    int eval_sample)
{
    // Extrair treino (instâncias selecionadas) — label na última coluna
    int Nfeat = F - 1;
    vector<int> train_idx;
    train_idx.reserve(N / 2);
    for (int i = 0; i < N; ++i)
        if (solution[i] == 1) train_idx.push_back(i);

    int Ntr = (int)train_idx.size();
    EvalMetrics m{};
    m.selected  = Ntr;
    m.reduction = 1.0 - (double)Ntr / N;
    if (Ntr == 0) return m;

    // Montar conjunto de teste (sample ou tudo)
    vector<int> test_idx(N);
    for (int i = 0; i < N; ++i) test_idx[i] = i;
    if (eval_sample > 0 && N > eval_sample) {
        // reservoir sampling
        for (int i = eval_sample; i < N; ++i) {
            int j = rand() % (i + 1);
            if (j < eval_sample) test_idx[j] = i;
        }
        test_idx.resize(eval_sample);
    }
    int Nte = (int)test_idx.size();

    // 1-NN brute-force (distância ao quadrado — suficiente para argmin)
    int tp = 0, fp = 0, tn = 0, fn = 0;
    double pos_class = 1.0;  // assume classe positiva = 1
    for (int ti = 0; ti < Nte; ++ti) {
        int t = test_idx[ti];
        const double* xt = h_X.data() + (long long)t * F;
        double true_label = xt[Nfeat];

        double best_d = 1e300;
        double pred_label = 0.0;
        for (int si = 0; si < Ntr; ++si) {
            int s = train_idx[si];
            const double* xs = h_X.data() + (long long)s * F;
            double d = 0.0;
            for (int k = 0; k < Nfeat; ++k) { double diff = xt[k]-xs[k]; d += diff*diff; }
            if (d < best_d) { best_d = d; pred_label = xs[Nfeat]; }
        }

        bool pred_pos = (pred_label == pos_class);
        bool true_pos = (true_label == pos_class);
        if (pred_pos && true_pos)  tp++;
        else if (pred_pos)         fp++;
        else if (true_pos)         fn++;
        else                       tn++;
    }

    m.acc       = (double)(tp + tn) / Nte;
    m.precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
    m.recall    = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
    m.f1        = (m.precision + m.recall > 0)
                  ? 2.0 * m.precision * m.recall / (m.precision + m.recall) : 0.0;
    return m;
}

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

static void write_solutions_csv(const vector<int>& colony, int K, int N, const string& path) {
    ofstream out(path);
    for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
            out << colony[(long long)k * N + j];
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
    const int    K                 = 64;
    const double initial_pheromone = 0.5;   // P(select)=0.5 inicialmente
    const double evaporation_rate  = 0.1;
    const double Q                 = evaporation_rate;  // mantém tau ≈ 0.5 no steady-state
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
    GpuBuffers buf = alloc_gpu(N, K, F, onthefly);

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

    // Inicializar colônia: -1 exceto posição inicial de cada formiga k = k
    vector<int> h_colony((long long)K * N, -1);
    for (int k = 0; k < K; ++k) h_colony[(long long)k * N + k] = 1;
    CUDA_CHECK(cudaMemcpy(buf.d_colony, h_colony.data(),
                          (long long)K * N * sizeof(int), cudaMemcpyHostToDevice));

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
    // K*N estados cuRAND: 1 por par (formiga, instância)
    CUDA_CHECK(cudaMalloc(&d_rng, (long long)K * N * sizeof(curandState)));
    int grid_rng = (K * N + BLOCK - 1) / BLOCK;
    init_curand_kernel<<<grid_rng, BLOCK>>>(d_rng, 42ULL, K * N);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Buffer de depósito de feromônio
    double* d_deposit;
    CUDA_CHECK(cudaMalloc(&d_deposit, N * sizeof(double)));

    // Loop principal do ACO
    int grid_ants = (K * N + BLOCK - 1) / BLOCK;
    int grid_phe  = (N + BLOCK - 1) / BLOCK;
    const int max_iters = 50;
    const int eval_sample = (N > 10000) ? 5000 : 0;

    // Buffer host para log por iteração (1 formiga = N ints)
    vector<int> h_ant0(N);

    CUDA_CHECK(cudaEventRecord(ev0));

    cudaEvent_t iter_ev0, iter_ev1;
    CUDA_CHECK(cudaEventCreate(&iter_ev0));
    CUDA_CHECK(cudaEventCreate(&iter_ev1));

    for (int iter = 0; iter < max_iters; ++iter) {
        CUDA_CHECK(cudaEventRecord(iter_ev0));

        // Zerar buffer de depósito (colônia é sobrescrita pelo kernel)
        CUDA_CHECK(cudaMemset(d_deposit, 0, N * sizeof(double)));

        // Construção paralela: K*N threads, 1 por (formiga, instância)
        ant_construction_kernel<<<grid_ants, BLOCK>>>(
            buf.d_pheromone, buf.d_colony,
            d_deposit, d_rng,
            N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Update feromônio: tau = (1-rho)*tau + rho*deposit_fraction
        // Mantém tau ∈ [0,1] sem deriva (evaporação já incorporada no apply)
        apply_deposit_kernel<<<grid_phe, BLOCK>>>(
            buf.d_pheromone, d_deposit, N, Q);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(iter_ev1));
        CUDA_CHECK(cudaEventSynchronize(iter_ev1));
        float iter_ms; CUDA_CHECK(cudaEventElapsedTime(&iter_ms, iter_ev0, iter_ev1));

        // Log: baixar formiga 0, avaliar métricas com 1-NN
        CUDA_CHECK(cudaMemcpy(h_ant0.data(), buf.d_colony,
                              N * sizeof(int), cudaMemcpyDeviceToHost));
        auto t_eval0 = chrono::high_resolution_clock::now();
        EvalMetrics em = evaluate_solution_cpu(h_ant0, h_X, N, F, eval_sample);
        auto t_eval1 = chrono::high_resolution_clock::now();
        double eval_ms = chrono::duration<double, milli>(t_eval1 - t_eval0).count();
        printf("Iter %d: F1=%.4f, Acc=%.4f, Reducao=%.1f%%, GPU=%.0f ms, Eval=%.0f ms%s\n",
               iter + 1, em.f1, em.acc, em.reduction * 100.0, iter_ms, eval_ms,
               eval_sample > 0 ? " (sample)" : "");
    }

    CUDA_CHECK(cudaEventDestroy(iter_ev0));
    CUDA_CHECK(cudaEventDestroy(iter_ev1));

    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    printf("Tempo compute GPU total: %.2f ms (%d iters)\n", ms, max_iters);

    // Download resultado final
    CUDA_CHECK(cudaEventRecord(ev0));
    CUDA_CHECK(cudaMemcpy(h_colony.data(), buf.d_colony,
                          (long long)K * N * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
    printf("Transferencia D->H colony: %.2f ms\n", ms);

    // Salvar resultado
    write_solutions_csv(h_colony, K, N, "results/solutions_cuda.csv");

    // Cleanup
    free_gpu(buf);
    CUDA_CHECK(cudaFree(d_rng));
    CUDA_CHECK(cudaFree(d_deposit));
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    return 0;
}
