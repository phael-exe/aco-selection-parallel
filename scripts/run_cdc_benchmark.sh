#!/usr/bin/env bash
# ===========================================================================
# run_cdc_benchmark.sh — Execução do ACO no dataset CDC Diabetes (N=253.680)
#
# Executa a redução de instâncias nas três implementações:
#   1. Sequencial (sem amostragem)
#   2. OpenMP (com número de threads detectado ou configurado)
#   3. CUDA (GPU)
#
# Uso:
#   bash scripts/run_cdc_benchmark.sh [iteracões] [formigas]
# ===========================================================================
set -euo pipefail

SEQ_BIN="./build/aco_seq"
OMP_BIN="./build/aco_omp"
CUD_BIN="./build/aco_cuda"
DATASET="data/cdc/cdc_diabetes.csv"
TARGET="Diabetes_binary"
OUT_FILE="results/cdc_diabetes_benchmark.log"

ITER="${1:-100}"
ANTS="${2:-64}"

# Detectar threads para OpenMP (usa nproc ou default 12)
THREADS=$(nproc 2>/dev/null || echo "12")

echo "======================================================="
echo " ACO CDC Diabetes Benchmark (N = 253.680)"
echo " Ants = $ANTS | Iterations = $ITER | Threads = $THREADS"
echo "======================================================="
echo "Os resultados detalhados serão salvos em: $OUT_FILE"
echo ""

mkdir -p results

# Garantir que o dataset existe
if [ ! -f "$DATASET" ]; then
    echo "[ERRO] Dataset '$DATASET' não encontrado."
    echo "Certifique-se de que o download foi feito na pasta data/cdc/."
    exit 1
fi

# Compilar binários se necessário
echo "=== 1. Compilando binários ==="
make all
echo "Compilação concluída com sucesso."
echo ""

# Função auxiliar para rodar e registrar logs
run_and_log() {
    local NAME="$1"
    local CMD="$2"
    
    echo "-------------------------------------------------------" | tee -a "$OUT_FILE"
    echo " Iniciando: $NAME" | tee -a "$OUT_FILE"
    echo " Comando: $CMD" | tee -a "$OUT_FILE"
    echo " Hora de início: $(date)" | tee -a "$OUT_FILE"
    echo "-------------------------------------------------------" | tee -a "$OUT_FILE"
    
    # Executa e joga tanto stdout quanto stderr no log, mas mantém o stderr na tela para progresso
    eval "$CMD" 2>&1 | tee -a "$OUT_FILE"
    
    echo "" | tee -a "$OUT_FILE"
    echo " Finalizado: $NAME às $(date)" | tee -a "$OUT_FILE"
    echo "-------------------------------------------------------" | tee -a "$OUT_FILE"
    echo ""
}

# Limpar arquivo de log anterior
echo "=== ACO CDC Diabetes Benchmark Run - $(date) ===" > "$OUT_FILE"

# 1. CUDA
echo "=== 2. Executando versão CUDA (GPU) ==="
run_and_log "CUDA (GPU)" "$CUD_BIN $DATASET $TARGET --ants $ANTS --iter $ITER"

# 2. OpenMP
echo "=== 3. Executando versão OpenMP ($THREADS threads) ==="
run_and_log "OpenMP ($THREADS threads)" "OMP_NUM_THREADS=$THREADS $OMP_BIN $DATASET $TARGET --ants $ANTS --iter $ITER"

# 3. Sequencial
echo "=== 4. Executando versão Sequencial (CPU) ==="
echo "ATENÇÃO: A versão sequencial sem amostragem no dataset completo"
echo "pode demorar várias horas para concluir."
run_and_log "Sequencial (CPU)" "$SEQ_BIN $DATASET $TARGET --ants $ANTS --iter $ITER --eval-sample 0"

echo "======================================================="
echo " Execução finalizada! Resultados salvos em: $OUT_FILE"
echo "======================================================="
