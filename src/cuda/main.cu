#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "memory.cuh"
#include "kernels.cuh"

using namespace std;

// ---------------------------------------------------------------------------
// CSV reader (replicado do sequencial — suporta coluna-alvo por nome)
// ---------------------------------------------------------------------------

static string trim_str(const string& s) {
    size_t a = s.find_first_not_of(" \t\r\n\xef\xbb\xbf");  // strip BOM tbm
    if (a == string::npos) return "";
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

static vector<string> split_csv(const string& line, char sep) {
    vector<string> fields;
    stringstream ss(line);
    string f;
    while (getline(ss, f, sep)) fields.push_back(f);
    return fields;
}

static char detect_sep(const string& path) {
    ifstream file(path);
    if (!file) return ';';
    string line; getline(file, line);
    size_t nc = count(line.begin(), line.end(), ',');
    size_t ns = count(line.begin(), line.end(), ';');
    return (nc > ns) ? ',' : ';';
}

struct Dataset {
    vector<double> X;  // [N * F] features
    vector<double> Y;  // [N] labels
    int N, F;
};

static Dataset read_csv(const string& path, const string& target_col, string& err) {
    Dataset ds{};
    ifstream file(path);
    if (!file) { err = "arquivo nao encontrado: " + path; return ds; }

    char sep = detect_sep(path);

    string header_line;
    if (!getline(file, header_line)) { err = "arquivo vazio"; return ds; }

    auto cols = split_csv(header_line, sep);
    for (auto& c : cols) c = trim_str(c);
    int ncols = (int)cols.size();

    int target_idx = -1;
    for (int i = 0; i < ncols; ++i)
        if (cols[i] == target_col) { target_idx = i; break; }
    if (target_idx < 0) { err = "coluna '" + target_col + "' nao encontrada"; return ds; }

    int F = ncols - 1;

    // Coletar todas as linhas validas
    vector<double> Xbuf, Ybuf;
    string line;
    int skipped = 0;
    while (getline(file, line)) {
        string lt = trim_str(line);
        if (lt.empty()) continue;
        auto fields = split_csv(lt, sep);
        if ((int)fields.size() != ncols) { skipped++; continue; }

        bool ok = true;
        double y = 0.0;
        vector<double> x(F);
        int xi = 0;
        for (int j = 0; j < ncols && ok; ++j) {
            string fld = trim_str(fields[j]);
            try {
                double v = stod(fld);
                if (isnan(v) || isinf(v)) { ok = false; break; }
                if (j == target_idx) y = v;
                else x[xi++] = v;
            } catch (...) { ok = false; }
        }
        if (!ok) { skipped++; continue; }
        Ybuf.push_back(y);
        Xbuf.insert(Xbuf.end(), x.begin(), x.end());
    }

    if (Ybuf.empty()) { err = "nenhuma linha valida"; return ds; }
    ds.X = move(Xbuf);
    ds.Y = move(Ybuf);
    ds.N = (int)ds.Y.size();
    ds.F = F;
    return ds;
}

// ---------------------------------------------------------------------------
// Métricas a partir de predições GPU (int[N]) + labels CPU (double[N])
// ---------------------------------------------------------------------------

struct Metrics { double f1, acc, precision, recall, reduction; int selected; };

// Detecta classes uma única vez por dataset
static void detect_classes(const vector<double>& Y,
                            vector<double>& classes, bool& binary) {
    set<double> s(Y.begin(), Y.end());
    classes.assign(s.begin(), s.end());  // sorted ascending
    binary = (classes.size() == 2);
}

static Metrics compute_metrics(const vector<int>& preds,
                                const vector<double>& Y, int N,
                                int selected_count,
                                const vector<double>& classes, bool binary)
{
    Metrics m{};
    m.selected  = selected_count;
    m.reduction = 1.0 - (double)selected_count / N;
    if (selected_count == 0) return m;

    int correct = 0;
    for (int i = 0; i < N; ++i)
        if ((double)preds[i] == Y[i]) correct++;
    m.acc = (double)correct / N;

    if (binary) {
        double pos = classes[1];
        int tp=0,fp=0,fn=0,tn=0;
        for (int i=0;i<N;++i){
            bool pp=((double)preds[i]==pos), tp_=(Y[i]==pos);
            if(pp&&tp_)tp++; else if(pp)fp++; else if(tp_)fn++; else tn++;
        }
        m.precision=(tp+fp>0)?(double)tp/(tp+fp):0.0;
        m.recall   =(tp+fn>0)?(double)tp/(tp+fn):0.0;
        m.f1=(m.precision+m.recall>0)?2*m.precision*m.recall/(m.precision+m.recall):0.0;
    } else {
        double sp=0,sr=0,sf=0;
        for(double cls:classes){
            int tp=0,fp=0,fn=0;
            for(int i=0;i<N;++i){
                bool pp=((double)preds[i]==cls),tp_=(Y[i]==cls);
                if(pp&&tp_)tp++; else if(pp)fp++; else if(tp_)fn++;
            }
            double p=(tp+fp>0)?(double)tp/(tp+fp):0.0;
            double r=(tp+fn>0)?(double)tp/(tp+fn):0.0;
            double f=(p+r>0)?2*p*r/(p+r):0.0;
            sp+=p; sr+=r; sf+=f;
        }
        int nc=(int)classes.size();
        m.precision=sp/nc; m.recall=sr/nc; m.f1=sf/nc;
    }
    return m;
}

// (mantida apenas como fallback para debugging — não usada no loop principal)
static Metrics evaluate_1nn_cpu(
    const vector<int>& sol,
    const vector<double>& X, const vector<double>& Y,
    int N, int F, int eval_sample)
{
    vector<int> tr;
    tr.reserve(N / 2);
    for (int i = 0; i < N; ++i)
        if (sol[i] == 1) tr.push_back(i);

    Metrics m{};
    m.selected  = (int)tr.size();
    m.reduction = 1.0 - (double)m.selected / N;
    if (tr.empty()) return m;

    // Conjunto de teste (sample ou tudo)
    vector<int> te(N);
    iota(te.begin(), te.end(), 0);
    if (eval_sample > 0 && N > eval_sample) {
        for (int i = eval_sample; i < N; ++i) {
            int j = rand() % (i + 1);
            if (j < eval_sample) te[j] = i;
        }
        te.resize(eval_sample);
    }
    int Nte = (int)te.size();
    int Ntr = (int)tr.size();

    // 1-NN brute-force: predizer label para cada instância de teste
    vector<double> Y_pred(Nte);
    for (int ti = 0; ti < Nte; ++ti) {
        int t = te[ti];
        double best_d   = 1e300;
        double pred_lbl = Y[tr[0]];
        for (int si = 0; si < Ntr; ++si) {
            int s = tr[si];
            double d = 0.0;
            for (int k = 0; k < F; ++k) {
                double diff = X[(long long)t * F + k] - X[(long long)s * F + k];
                d += diff * diff;
            }
            if (d < best_d) { best_d = d; pred_lbl = Y[s]; }
        }
        Y_pred[ti] = pred_lbl;
    }
    // Y_true para o conjunto de teste
    vector<double> Y_true(Nte);
    for (int ti = 0; ti < Nte; ++ti) Y_true[ti] = Y[te[ti]];

    // Detectar classes únicas (no dataset completo)
    set<double> cls_set(Y.begin(), Y.end());
    vector<double> classes(cls_set.begin(), cls_set.end());  // sorted ascending
    bool binary = (classes.size() == 2);

    // Acurácia
    int correct = 0;
    for (int ti = 0; ti < Nte; ++ti)
        if (Y_pred[ti] == Y_true[ti]) correct++;
    m.acc = (double)correct / Nte;

    if (binary) {
        // Classe positiva = classes[1] (valor maior, igual ao sequencial)
        double pos = classes[1];
        int tp = 0, fp = 0, fn = 0, tn = 0;
        for (int ti = 0; ti < Nte; ++ti) {
            bool pp = (Y_pred[ti] == pos), tp_ = (Y_true[ti] == pos);
            if (pp && tp_)  tp++;
            else if (pp)    fp++;
            else if (tp_)   fn++;
            else            tn++;
        }
        m.precision = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
        m.recall    = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
        m.f1        = (m.precision + m.recall > 0)
                      ? 2.0 * m.precision * m.recall / (m.precision + m.recall) : 0.0;
    } else {
        // Macro-average F1 (igual ao sequencial para multiclasse)
        double sum_p = 0.0, sum_r = 0.0, sum_f1 = 0.0;
        for (double cls : classes) {
            int tp = 0, fp = 0, fn = 0;
            for (int ti = 0; ti < Nte; ++ti) {
                bool pp = (Y_pred[ti] == cls), tp_ = (Y_true[ti] == cls);
                if (pp && tp_) tp++;
                else if (pp)   fp++;
                else if (tp_)  fn++;
            }
            double p = (tp + fp > 0) ? (double)tp / (tp + fp) : 0.0;
            double r = (tp + fn > 0) ? (double)tp / (tp + fn) : 0.0;
            double f = (p + r > 0) ? 2.0 * p * r / (p + r) : 0.0;
            sum_p += p; sum_r += r; sum_f1 += f;
        }
        int nc = (int)classes.size();
        m.precision = sum_p / nc;
        m.recall    = sum_r / nc;
        m.f1        = sum_f1 / nc;
    }
    return m;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

static void print_usage(const char* p) {
    cerr << "Uso: " << p << " <dataset_path> <coluna_alvo>"
         << " [--ants K] [--iter N] [--evap R] [--Q V]"
         << " [--alpha A] [--beta B] [--eval-top-k K]\n";
}

int main(int argc, char** argv) {
    if (argc < 3) { print_usage(argv[0]); return 1; }

    string path       = argv[1];
    string target_col = argv[2];

    int    num_ants         = 64;
    int    max_iter         = 100;
    double evap_rate        = 0.1;
    double Q                = 1.0;
    double alpha            = 1.0;
    double beta             = 1.0;
    int    eval_top_k_exp   = 0;   // 0 = auto-detectar

    for (int i = 3; i + 1 < argc; i += 2) {
        if      (!strcmp(argv[i], "--ants"))       num_ants       = atoi(argv[i+1]);
        else if (!strcmp(argv[i], "--iter"))       max_iter       = atoi(argv[i+1]);
        else if (!strcmp(argv[i], "--evap"))       evap_rate      = atof(argv[i+1]);
        else if (!strcmp(argv[i], "--Q"))          Q              = atof(argv[i+1]);
        else if (!strcmp(argv[i], "--alpha"))      alpha          = atof(argv[i+1]);
        else if (!strcmp(argv[i], "--beta"))       beta           = atof(argv[i+1]);
        else if (!strcmp(argv[i], "--eval-top-k")) eval_top_k_exp = atoi(argv[i+1]);
        else cerr << "Aviso: argumento desconhecido '" << argv[i] << "' ignorado\n";
    }

    // ---- Leitura CSV ----
    auto t_csv0 = chrono::high_resolution_clock::now();
    string err;
    Dataset ds = read_csv(path, target_col, err);
    auto t_csv1 = chrono::high_resolution_clock::now();
    if (ds.N == 0) { cerr << "Erro CSV: " << err << "\n"; return 1; }
    double csv_ms = chrono::duration<double, milli>(t_csv1 - t_csv0).count();

    int N = ds.N, F = ds.F;
    int K = num_ants;

    cout << "Dataset carregado: " << N << " instancias, " << F << " features\n";
    cout << "Tempo de leitura:  " << csv_ms << " ms\n";
    cout << "Parametros ACO:    ants=" << K
         << ", iter=" << max_iter
         << ", evap=" << evap_rate
         << ", Q=" << Q
         << ", alpha=" << alpha
         << ", beta=" << beta << "\n\n";

    // Auto-detectar eval_top_k
    int eval_top_k = (eval_top_k_exp > 0) ? eval_top_k_exp
                   : (N <= 1000) ? 5 : (N <= 10000) ? 3 : 1;

    // Detectar classes uma única vez (para métricas)
    vector<double> classes; bool binary;
    detect_classes(ds.Y, classes, binary);

    srand(42);

    // ---- Alocar GPU ----
    GpuBuffers buf = alloc_gpu(N, K, F);

    // Upload X e Y
    CUDA_CHECK(cudaMemcpy(buf.d_X, ds.X.data(),
                          (size_t)N * F * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(buf.d_Y, ds.Y.data(),
                          (size_t)N * sizeof(double), cudaMemcpyHostToDevice));

    // ---- Computar visibilidade 1D na GPU ----
    // avg_dist[i] = média das distâncias euclidianas de i a todos os outros
    // vis[i] = 1 / (1 + avg_dist[i])
    double* d_avg_dist;
    CUDA_CHECK(cudaMalloc(&d_avg_dist, N * sizeof(double)));
    avg_distance_kernel<<<(N + 255) / 256, 256>>>(buf.d_X, d_avg_dist, N, F);
    CUDA_CHECK(cudaDeviceSynchronize());

    vector<double> h_avg(N), h_vis(N);
    CUDA_CHECK(cudaMemcpy(h_avg.data(), d_avg_dist, N * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_avg_dist));
    for (int i = 0; i < N; ++i)
        h_vis[i] = 1.0 / (1.0 + h_avg[i]);
    CUDA_CHECK(cudaMemcpy(buf.d_vis, h_vis.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // ---- Inicializar feromônio ----
    vector<double> h_tau(N, 1.0);
    CUDA_CHECK(cudaMemcpy(buf.d_pheromone, h_tau.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    // ---- Inicializar cuRAND ----
    curandState* d_rng;
    CUDA_CHECK(cudaMalloc(&d_rng, (size_t)K * N * sizeof(curandState)));
    {
        int grid_rng = (K * N + 255) / 256;
        init_curand_kernel<<<grid_rng, 256>>>(d_rng, 12345ULL, K * N);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // ---- Buffers host para loop ----
    vector<double> h_select_prob(N);
    vector<int>    h_selected_count(K);   // só K ints por iter (256 bytes para K=64)
    vector<int>    h_predictions(N);      // predições 1-NN GPU → CPU (N ints ≈ 20 KB)
    vector<int>    h_best(N, 0);

    double best_f1       = -1.0;
    double best_acc      = 0.0, best_prec = 0.0, best_rec = 0.0;
    int    best_selected = 0;
    int    iters_done    = 0;
    int    no_improve    = 0;
    int    patience      = 10;

    int grid_ants = (K * N + 255) / 256;
    int grid_phe  = (N + 255) / 256;
    int grid_knn  = (N + 255) / 256;

    auto t_aco0 = chrono::high_resolution_clock::now();
    double gpu_compute_ms = 0.0;
    double gpu_eval_ms    = 0.0;

    for (int iter = 0; iter < max_iter; ++iter) {
        // -- 1. select_prob na CPU: τ^α·η^β normalizado --
        CUDA_CHECK(cudaMemcpy(h_tau.data(), buf.d_pheromone,
                              N * sizeof(double), cudaMemcpyDeviceToHost));
        double wmax = 0.0;
        for (int i = 0; i < N; ++i) {
            h_select_prob[i] = pow(h_tau[i], alpha) * pow(h_vis[i], beta);
            if (h_select_prob[i] > wmax) wmax = h_select_prob[i];
        }
        if (wmax <= 0.0) wmax = 1.0;
        for (int i = 0; i < N; ++i) h_select_prob[i] /= wmax;
        CUDA_CHECK(cudaMemcpy(buf.d_select_prob, h_select_prob.data(),
                              N * sizeof(double), cudaMemcpyHostToDevice));

        // -- 2. Reset --
        CUDA_CHECK(cudaMemset(buf.d_deposit,        0, N * sizeof(double)));
        CUDA_CHECK(cudaMemset(buf.d_selected_count, 0, K * sizeof(int)));

        // -- 3. GPU: construção + depósito + evaporação --
        cudaEvent_t ev0, ev1; float ms_iter;
        CUDA_CHECK(cudaEventCreate(&ev0));
        CUDA_CHECK(cudaEventCreate(&ev1));
        CUDA_CHECK(cudaEventRecord(ev0));

        ant_construction_kernel<<<grid_ants, 256>>>(
            buf.d_select_prob, buf.d_colony, buf.d_selected_count, d_rng, N, K);
        CUDA_CHECK(cudaDeviceSynchronize());

        deposit_kernel<<<grid_ants, 256>>>(
            buf.d_colony, buf.d_selected_count, buf.d_deposit, N, K, Q);
        CUDA_CHECK(cudaDeviceSynchronize());

        apply_pheromone_kernel<<<grid_phe, 256>>>(
            buf.d_pheromone, buf.d_deposit, N, evap_rate);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaEventRecord(ev1));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        CUDA_CHECK(cudaEventElapsedTime(&ms_iter, ev0, ev1));
        gpu_compute_ms += ms_iter;
        CUDA_CHECK(cudaEventDestroy(ev0));
        CUDA_CHECK(cudaEventDestroy(ev1));

        // -- 4. Download de selected_count (K ints ≈ 256 bytes) para proxy ranking --
        CUDA_CHECK(cudaMemcpy(h_selected_count.data(), buf.d_selected_count,
                              K * sizeof(int), cudaMemcpyDeviceToHost));

        // Proxy: top-K formigas por número de instâncias selecionadas
        vector<pair<int,int>> scores(K);
        for (int k = 0; k < K; ++k) scores[k] = {k, h_selected_count[k]};
        stable_sort(scores.begin(), scores.end(),
            [](const auto& a, const auto& b){ return a.second > b.second; });

        // -- 5. GPU 1-NN: avalia top-K formigas diretamente na GPU --
        double best_f1_iter = -1.0;
        int    eval_count   = min(eval_top_k, K);

        for (int ei = 0; ei < eval_count; ++ei) {
            int k = scores[ei].first;

            cudaEvent_t ek0, ek1; float ms_knn;
            CUDA_CHECK(cudaEventCreate(&ek0));
            CUDA_CHECK(cudaEventCreate(&ek1));
            CUDA_CHECK(cudaEventRecord(ek0));

            // knn_1nn_kernel: N threads, cada um encontra NN no subconjunto selecionado
            knn_1nn_kernel<<<grid_knn, 256>>>(
                buf.d_X, buf.d_Y,
                buf.d_colony + (long long)k * N,
                buf.d_predictions,
                N, F, N);
            CUDA_CHECK(cudaDeviceSynchronize());

            CUDA_CHECK(cudaEventRecord(ek1));
            CUDA_CHECK(cudaEventSynchronize(ek1));
            CUDA_CHECK(cudaEventElapsedTime(&ms_knn, ek0, ek1));
            gpu_eval_ms += ms_knn;
            CUDA_CHECK(cudaEventDestroy(ek0));
            CUDA_CHECK(cudaEventDestroy(ek1));

            // Download predições (N ints ≈ 20 KB)
            CUDA_CHECK(cudaMemcpy(h_predictions.data(), buf.d_predictions,
                                  N * sizeof(int), cudaMemcpyDeviceToHost));

            Metrics m = compute_metrics(h_predictions, ds.Y, N,
                                        h_selected_count[k], classes, binary);
            fprintf(stderr, "  Ant %d: F1=%.4f Acc=%.4f Red=%.1f%%\n",
                    k, m.f1, m.acc, m.reduction * 100.0);

            if (m.f1 > best_f1_iter)
                best_f1_iter = m.f1;
            if (m.f1 > best_f1) {
                best_f1       = m.f1;
                best_acc      = m.acc;
                best_prec     = m.precision;
                best_rec      = m.recall;
                best_selected = m.selected;
                // Download só da melhor solução (N ints ≈ 20 KB) — acontece poucas vezes
                CUDA_CHECK(cudaMemcpy(h_best.data(),
                                      buf.d_colony + (long long)k * N,
                                      N * sizeof(int), cudaMemcpyDeviceToHost));
                no_improve = -1;  // será incrementado logo abaixo
            }
        }

        no_improve++;
        fprintf(stderr,
            "Iter %d: F1=%.4f Acc=%.4f Red=%.1f%% GPU=%.2fms+%.2fms_knn (avaliadas=%d)\n",
            iter + 1, best_f1, best_acc,
            100.0 * (N - best_selected) / N,
            ms_iter, gpu_eval_ms / (iter + 1),
            eval_count);

        iters_done = iter + 1;
        if (no_improve >= patience) {
            fprintf(stderr, "Early stopping: sem melhoria por %d iteracoes\n", patience);
            break;
        }
    }

    auto t_aco1 = chrono::high_resolution_clock::now();
    double aco_ms = chrono::duration<double, milli>(t_aco1 - t_aco0).count();

    // ---- Output (formato igual ao sequencial, parseável pelo script) ----
    cout << "\n=== Resultado ACO (CUDA) ===\n";
    cout << "Melhor solucao: " << best_selected << "/" << N
         << " instancias (reducao ~"
         << (100.0 * (N - best_selected) / N) << "%)\n";
    cout << "Acuracia: "   << best_acc  << "\n";
    cout << "Precisao: "   << best_prec << "\n";
    cout << "Recall: "     << best_rec  << "\n";
    cout << "F1-Score: "   << best_f1   << "\n";
    cout << "Iteracoes executadas: " << iters_done << "/" << max_iter;
    if (iters_done < max_iter) cout << " (early stop)\n"; else cout << "\n";
    cout << "Tempo ACO: "       << aco_ms        << " ms\n";
    cout << "GPU compute: "     << gpu_compute_ms << " ms\n";
    cout << "GPU eval (1-NN): " << gpu_eval_ms    << " ms\n";
    cout << "Eval strategy: top-" << eval_top_k << " formigas/iteracao (";
    if (eval_top_k_exp > 0) cout << "explicitamente configurado)\n";
    else cout << "auto-detectado para N=" << N << ")\n";

    // ---- Cleanup ----
    free_gpu(buf);
    CUDA_CHECK(cudaFree(d_rng));
    return 0;
}
