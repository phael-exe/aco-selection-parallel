#include "metrics.h"
#include "knn.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <set>
#include <map>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Helper: Detectar se é binário ou multiclasse
// ---------------------------------------------------------------------------

static bool is_binary_classification(const double* Y, size_t N) {
    std::set<double> unique_classes;
    for (size_t i = 0; i < N; ++i) {
        unique_classes.insert(Y[i]);
    }
    return unique_classes.size() == 2;
}

// ---------------------------------------------------------------------------
// Helper: Calcular TP, FP, TN, FN para classificação binária
// ---------------------------------------------------------------------------

struct BinaryMetrics {
    size_t tp, fp, tn, fn;
};

static BinaryMetrics compute_binary_metrics(
    const double* Y_pred, const double* Y_true, size_t N,
    double positive_class = 1.0) {
    
    BinaryMetrics m{0, 0, 0, 0};
    
    for (size_t i = 0; i < N; ++i) {
        bool pred_positive = (Y_pred[i] == positive_class);
        bool true_positive = (Y_true[i] == positive_class);
        
        if (pred_positive && true_positive) m.tp++;
        else if (pred_positive && !true_positive) m.fp++;
        else if (!pred_positive && !true_positive) m.tn++;
        else m.fn++;
    }
    
    return m;
}

// ---------------------------------------------------------------------------
// Helper: Calcular métricas para multiclasse (macro-average)
// ---------------------------------------------------------------------------

struct MulticlassMetrics {
    double accuracy;
    double precision;  // macro-average
    double recall;     // macro-average
    double f1_score;   // macro-average
};

static MulticlassMetrics compute_multiclass_metrics(
    const double* Y_pred, const double* Y_true, size_t N) {
    
    // Encontrar classes únicas
    std::set<double> classes_set;
    for (size_t i = 0; i < N; ++i) {
        classes_set.insert(Y_true[i]);
    }
    std::vector<double> classes(classes_set.begin(), classes_set.end());
    
    // Acurácia global
    size_t correct = 0;
    for (size_t i = 0; i < N; ++i) {
        if (Y_pred[i] == Y_true[i]) correct++;
    }
    double accuracy = static_cast<double>(correct) / N;
    
    // Calcular P, R, F1 para cada classe (one-vs-rest)
    double sum_precision = 0.0, sum_recall = 0.0, sum_f1 = 0.0;
    
    for (double cls : classes) {
        size_t tp = 0, fp = 0, fn = 0;
        
        for (size_t i = 0; i < N; ++i) {
            bool pred_cls = (Y_pred[i] == cls);
            bool true_cls = (Y_true[i] == cls);
            
            if (pred_cls && true_cls) tp++;
            else if (pred_cls && !true_cls) fp++;
            else if (!pred_cls && true_cls) fn++;
        }
        
        double precision = (tp + fp > 0) ? static_cast<double>(tp) / (tp + fp) : 0.0;
        double recall = (tp + fn > 0) ? static_cast<double>(tp) / (tp + fn) : 0.0;
        double f1 = (precision + recall > 0) ? 2.0 * (precision * recall) / (precision + recall) : 0.0;
        
        sum_precision += precision;
        sum_recall += recall;
        sum_f1 += f1;
    }
    
    size_t num_classes = classes.size();
    return {
        accuracy,
        sum_precision / num_classes,
        sum_recall / num_classes,
        sum_f1 / num_classes
    };
}

// ---------------------------------------------------------------------------
// Função Principal: Avaliar Solução
// ---------------------------------------------------------------------------

QualityMetrics evaluate_solution(
    const int* solution,
    const double* X, const double* Y, size_t N, size_t F) {
    
    QualityMetrics result{};
    
    // Contar instâncias selecionadas
    size_t selected_count = 0;
    for (size_t i = 0; i < N; ++i) {
        if (solution[i] == 1) selected_count++;
    }
    
    result.selected_count = selected_count;
    result.reduction_rate = 1.0 - (static_cast<double>(selected_count) / N);
    
    // Se nenhuma instância selecionada, retorna métrica zero
    if (selected_count == 0) {
        result.accuracy = 0.0;
        result.precision = 0.0;
        result.recall = 0.0;
        result.f1_score = 0.0;
        return result;
    }
    
    // Alocar X_train e Y_train
    double* X_train = new double[selected_count * F];
    double* Y_train = new double[selected_count];
    
    size_t train_idx = 0;
    for (size_t i = 0; i < N; ++i) {
        if (solution[i] == 1) {
            for (size_t f = 0; f < F; ++f) {
                X_train[train_idx * F + f] = X[i * F + f];
            }
            Y_train[train_idx] = Y[i];
            train_idx++;
        }
    }
    
    // Executar 1-NN
    double* Y_pred = knn_1nn_predict(X_train, Y_train, selected_count, X, N, F);
    
    // Calcular métricas
    bool binary = is_binary_classification(Y, N);
    
    if (binary) {
        // Classificação binária
        // Detectar qual é a classe positiva (assumir 1.0 ou maior valor único)
        std::set<double> classes_set;
        for (size_t i = 0; i < N; ++i) {
            classes_set.insert(Y[i]);
        }
        std::vector<double> classes(classes_set.begin(), classes_set.end());
        double positive_class = (classes.size() == 2) ? classes[1] : 1.0;
        
        BinaryMetrics bm = compute_binary_metrics(Y_pred, Y, N, positive_class);
        
        result.accuracy = static_cast<double>(bm.tp + bm.tn) / N;
        result.precision = (bm.tp + bm.fp > 0) ? static_cast<double>(bm.tp) / (bm.tp + bm.fp) : 0.0;
        result.recall = (bm.tp + bm.fn > 0) ? static_cast<double>(bm.tp) / (bm.tp + bm.fn) : 0.0;
        result.f1_score = (result.precision + result.recall > 0)
            ? 2.0 * (result.precision * result.recall) / (result.precision + result.recall)
            : 0.0;
    } else {
        // Classificação multiclasse (macro-average)
        MulticlassMetrics mm = compute_multiclass_metrics(Y_pred, Y, N);
        result.accuracy = mm.accuracy;
        result.precision = mm.precision;
        result.recall = mm.recall;
        result.f1_score = mm.f1_score;
    }
    
    // Liberar memória
    delete[] X_train;
    delete[] Y_train;
    delete[] Y_pred;
    
    return result;
}
