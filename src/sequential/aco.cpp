#include "aco.h"
#include "metrics.h"

#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <vector>
#include <ctime>

// ---------------------------------------------------------------------------
// Constantes e configurações
// ---------------------------------------------------------------------------

static constexpr size_t DISTANCE_THRESHOLD = 10000;  // N > 10k usa on-the-fly
static constexpr double PHEROMONE_MIN = 1e-10;
static constexpr double EPSILON = 1e-9;

// ---------------------------------------------------------------------------
// Helpers: Distância
// ---------------------------------------------------------------------------

static double compute_distance_single(const double* x1, const double* x2, size_t F) {
    double sum = 0.0;
    for (size_t k = 0; k < F; ++k) {
        double diff = x1[k] - x2[k];
        sum += diff * diff;
    }
    return std::sqrt(sum);
}

static void compute_distances_precomputed(const double* X, size_t N, size_t F, double* distances) {
    // Calcula só j > i (lower triangular), depois copia para [j][i]
    for (size_t i = 0; i < N; ++i) {
        distances[i * N + i] = 0.0;  // diagonal = 0
        
        for (size_t j = i + 1; j < N; ++j) {
            const double* xi = X + i * F;
            const double* xj = X + j * F;
            double dist = compute_distance_single(xi, xj, F);
            distances[i * N + j] = dist;
            distances[j * N + i] = dist;  // simetria
        }
    }
}

// Função auxiliar (não usada por enquanto, deixada para futuro on-the-fly)
// static double compute_distance_on_the_fly(const double* X, size_t i, size_t j, size_t F) {
//     const double* xi = X + i * F;
//     const double* xj = X + j * F;
//     return compute_distance_single(xi, xj, F);
// }

// ---------------------------------------------------------------------------
// Helpers: Visibilidade
// ---------------------------------------------------------------------------

static void compute_visibility_precomputed(const double* distances, size_t N, double* visibility) {
    // Calcular média de distâncias para cada instância
    std::vector<double> avg_dist(N, 0.0);
    
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t j = 0; j < N; ++j) {
            if (i != j) {
                sum += distances[i * N + j];
            }
        }
        avg_dist[i] = sum / (N - 1);
    }
    
    // visibility[i] = 1.0 / (1.0 + avg_dist[i])
    for (size_t i = 0; i < N; ++i) {
        visibility[i] = 1.0 / (1.0 + avg_dist[i]);
    }
}

static void compute_visibility_on_the_fly(const double* X, size_t N, size_t F, double* visibility) {
    // Calcular centróide
    std::vector<double> centroid(F, 0.0);
    for (size_t i = 0; i < N; ++i) {
        for (size_t f = 0; f < F; ++f) {
            centroid[f] += X[i * F + f];
        }
    }
    for (size_t f = 0; f < F; ++f) {
        centroid[f] /= N;
    }
    
    // Calcular distância de cada instância ao centróide
    for (size_t i = 0; i < N; ++i) {
        double sum = 0.0;
        for (size_t f = 0; f < F; ++f) {
            double diff = X[i * F + f] - centroid[f];
            sum += diff * diff;
        }
        double dist_to_centroid = std::sqrt(sum);
        visibility[i] = 1.0 / (1.0 + dist_to_centroid);
    }
}

// ---------------------------------------------------------------------------
// Helpers: Colônia
// ---------------------------------------------------------------------------

static int* init_colony(size_t K, size_t N) {
    int* colony = new int[K * N];
    
    // Inicializar com -1 (não visitado)
    std::fill(colony, colony + K * N, -1);
    
    // Cada formiga começa em uma posição aleatória única
    std::vector<size_t> random_starts(N);
    for (size_t i = 0; i < N; ++i) {
        random_starts[i] = i;
    }
    
    // Shuffle (Fisher-Yates)
    for (size_t i = N - 1; i > 0; --i) {
        size_t j = rand() % (i + 1);
        std::swap(random_starts[i], random_starts[j]);
    }
    
    // Cada formiga k começa na posição random_starts[k % N]
    for (size_t k = 0; k < K; ++k) {
        size_t start_pos = random_starts[k % N];
        colony[k * N + start_pos] = 1;
    }
    
    return colony;
}

static void free_colony(int* colony) {
    delete[] colony;
}

// ---------------------------------------------------------------------------
// Helpers: Feromônio
// ---------------------------------------------------------------------------

static void init_pheromone(double* pheromone, size_t N, double initial_value) {
    for (size_t i = 0; i < N; ++i) {
        pheromone[i] = initial_value;
    }
}

static void deposit_pheromone(double* pheromone, size_t N, const int* ant_solution, double Q) {
    // Contar instâncias selecionadas (tour_length)
    size_t tour_length = 0;
    for (size_t i = 0; i < N; ++i) {
        if (ant_solution[i] == 1) {
            tour_length++;
        }
    }
    
    // Se nenhuma selecionada, não deposita
    if (tour_length == 0) {
        return;
    }
    
    // Depositar feromônio nas selecionadas
    double deposit = Q / tour_length;
    for (size_t i = 0; i < N; ++i) {
        if (ant_solution[i] == 1) {
            pheromone[i] += deposit;
        }
    }
}

static void evaporate_pheromone(double* pheromone, size_t N, double rho) {
    for (size_t i = 0; i < N; ++i) {
        pheromone[i] *= (1.0 - rho);
        if (pheromone[i] < PHEROMONE_MIN) {
            pheromone[i] = PHEROMONE_MIN;
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers: Construção de Soluções
// ---------------------------------------------------------------------------

static void construct_solution_for_ant(int* ant_solution, size_t N,
                                        const double* /* pheromone */, const double* /* visibility */) {
    // Para cada instância não visitada, decidir seleção com 50% de chance
    for (size_t i = 0; i < N; ++i) {
        if (ant_solution[i] == -1) {  // não visitado
            // Aplicar 50% de chance (baseline)
            double r = static_cast<double>(rand()) / RAND_MAX;
            if (r < 0.5) {
                ant_solution[i] = 1;  // selecionada
            } else {
                ant_solution[i] = 0;  // rejeitada
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Função Principal: ACO
// ---------------------------------------------------------------------------

ACOResult run_aco(const double* X, const double* Y, size_t N, size_t F, ACOConfig config) {
    auto time_start = std::chrono::high_resolution_clock::now();
    
    ACOResult result{};
    result.best_solution = new int[N];
    result.selected = 0;
    result.best_fitness = 0.0;
    result.iterations = 0;
    result.time_seconds = 0.0;
    
    // Inicializar aleatoriedade
    srand(static_cast<unsigned>(time(nullptr)));
    
    // Detectar modo
    bool precomputed = (N <= DISTANCE_THRESHOLD);
    
    // Alocar arrays
    double* distances = nullptr;
    double* pheromone = new double[N];
    double* visibility = new double[N];
    int* colony = init_colony(config.K, N);
    
    // Calcular distâncias uma vez
    if (precomputed) {
        distances = new double[N * N];
        compute_distances_precomputed(X, N, F, distances);
        compute_visibility_precomputed(distances, N, visibility);
    } else {
        // On-the-fly: calcular visibilidade baseada em centróide
        compute_visibility_on_the_fly(X, N, F, visibility);
    }
    
    // Inicializar feromônio
    init_pheromone(pheromone, N, 1.0);
    
    // Inicializar melhor solução
    std::fill(result.best_solution, result.best_solution + N, -1);
    
    size_t no_improve_count = 0;
    
    // ===== Loop Principal =====
    for (size_t iter = 0; iter < config.max_iter; ++iter) {
        // Resetar colônia para esta iteração
        std::fill(colony, colony + config.K * N, -1);
        
        // Cada formiga começa em uma posição aleatória
        std::vector<size_t> random_starts(N);
        for (size_t i = 0; i < N; ++i) {
            random_starts[i] = i;
        }
        for (size_t i = N - 1; i > 0; --i) {
            size_t j = rand() % (i + 1);
            std::swap(random_starts[i], random_starts[j]);
        }
        
        for (size_t k = 0; k < config.K; ++k) {
            size_t start_pos = random_starts[k % N];
            colony[k * N + start_pos] = 1;
        }
        
        // Construir solução para cada formiga
        for (size_t k = 0; k < config.K; ++k) {
            int* ant_solution = colony + k * N;
            construct_solution_for_ant(ant_solution, N, pheromone, visibility);
        }
        
        // Depositar feromônio de todas as formigas
        for (size_t k = 0; k < config.K; ++k) {
            deposit_pheromone(pheromone, N, colony + k * N, config.Q);
        }
        
        // Evaporar feromônio
        evaporate_pheromone(pheromone, N, config.rho);
        
        // ===== NOVO: Avaliar top-K formigas com 1-NN =====
        // Ordenar formigas por proxy fitness (contagem selecionadas)
        // Em caso de empate: conserva ordem original (formiga com menor índice vem primeiro)
        std::vector<std::pair<size_t, double>> ant_scores;
        for (size_t k = 0; k < config.K; ++k) {
            size_t count = 0;
            for (size_t i = 0; i < N; ++i) {
                if (colony[k * N + i] == 1) count++;
            }
            ant_scores.push_back({k, static_cast<double>(count) / N});
        }
        
        // Sort descending por fitness, mantém ordem original em empates (stable_sort)
        std::stable_sort(ant_scores.begin(), ant_scores.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Avaliar top-K formigas com 1-NN
        double best_f1_this_iter = 0.0;
        size_t best_ant_idx = 0;
        QualityMetrics best_metrics_this_iter{};
        
        size_t eval_count = std::min(config.eval_top_k, config.K);
        for (size_t i = 0; i < eval_count; ++i) {
            size_t k = ant_scores[i].first;
            int* ant_solution = colony + k * N;
            QualityMetrics metrics = evaluate_solution(ant_solution, X, Y, N, F);
            
            if (metrics.f1_score > best_f1_this_iter) {
                best_f1_this_iter = metrics.f1_score;
                best_ant_idx = k;
                best_metrics_this_iter = metrics;
            }
            
            fprintf(stderr, "  Ant %zu: F1=%.4f, Acc=%.4f, Redução=%.1f%%\n",
                k, metrics.f1_score, metrics.accuracy, metrics.reduction_rate * 100.0);
        }
        
        // Usar F1 como fitness (não mais contagem proxy)
        double current_fitness = best_f1_this_iter;
        size_t current_selected = best_metrics_this_iter.selected_count;
        
        // Atualizar melhor solução global (baseado em F1 real)
        if (current_fitness > result.best_fitness) {
            result.best_fitness = current_fitness;
            result.accuracy = best_metrics_this_iter.accuracy;
            result.precision = best_metrics_this_iter.precision;
            result.recall = best_metrics_this_iter.recall;
            result.selected = current_selected;
            std::memcpy(result.best_solution, colony + best_ant_idx * N, N * sizeof(int));
            no_improve_count = 0;
        } else {
            no_improve_count++;
        }
        
        // Logging
        fprintf(stderr, "Iter %zu: F1=%.4f, Acc=%.4f, Redução=%.1f%%, Ants avaliadas=%zu\n",
            iter + 1, current_fitness, result.accuracy, best_metrics_this_iter.reduction_rate * 100.0, eval_count);
        
        // Early stopping
        if (no_improve_count >= config.patience) {
            fprintf(stderr, "Early stopping: sem melhoria por %zu iterações\n", config.patience);
            result.iterations = iter + 1;
            break;
        }
        
        result.iterations = iter + 1;
    }
    
    // Libertar memória
    free_colony(colony);
    delete[] pheromone;
    delete[] visibility;
    if (distances != nullptr) {
        delete[] distances;
    }
    
    auto time_end = std::chrono::high_resolution_clock::now();
    result.time_seconds = std::chrono::duration<double>(time_end - time_start).count();
    
    return result;
}

void free_aco_result(ACOResult* result) {
    if (result) {
        delete[] result->best_solution;
        result->best_solution = nullptr;
    }
}
