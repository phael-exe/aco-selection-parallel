#!/usr/bin/env bash
# ===========================================================================
# run_hpc_experiment.sh — Suite de Experimentos HPC (ACO Selection Parallel)
#
# Realiza um benchmark abrangente:
#   1. OpenMP: Diferentes threads (1, 2, 4, 8, 16, 32) + Schedulers (static, dynamic, guided)
#   2. CUDA: Diferentes Block Sizes (32, 64, 128, 256, 512, 1024)
#   3. Coleta automática de GFLOPS
#   4. Coleta opcional de IPC e Cache Misses usando 'perf' (se permitido pelo kernel)
#
# Uso:
#   bash scripts/run_hpc_experiment.sh [iteracões] [formigas] [dataset_path]
# ===========================================================================
set -euo pipefail

ITER="${1:-5}"
ANTS="${2:-16}"
DATASET="${3:-data/cdc/cdc_diabetes.csv}"
TARGET="${4:-Diabetes_binary}"

SEQ_BIN="./build/aco_seq"
OMP_BIN="./build/aco_omp"
CUD_BIN="./build/aco_cuda"
REPORT_MD="results/hpc_experiment_report.md"
RAW_LOGS="results/hpc_raw_runs.log"

echo "======================================================="
echo " ACO HIGH-PERFORMANCE COMPUTING EXPERIMENTAL SUITE"
echo " Dataset: $DATASET | Ants: $ANTS | Iterations: $ITER"
echo "======================================================="
echo ""

# Garantir que o dataset existe
if [ ! -f "$DATASET" ]; then
    echo "[ERRO] Dataset '$DATASET' não encontrado."
    exit 1
fi

# Compilar
echo "=== 1. Compilando Binários Otimizados ==="
make clean && make all
echo ""

# Verificar suporte ao 'perf'
echo "=== 2. Verificando privilégios do 'perf' ==="
USE_PERF=false
if perf stat true >/dev/null 2>&1; then
    echo "[INFO] 'perf' disponível e funcional! Métricas de IPC e Cache serão coletadas."
    USE_PERF=true
else
    echo "[AVISO] 'perf' restrito pelo kernel (perf_event_paranoid). Pulando IPC/Cache."
    echo "Para habilitar, você pode rodar: sudo sysctl kernel.perf_event_paranoid=-1"
fi
echo ""

mkdir -p results
echo "=== ACO HPC Experiment Run - $(date) ===" > "$RAW_LOGS"

# Criar cabeçalho do relatório Markdown
cat << EOF > "$REPORT_MD"
# Relatório de Performance Científico - ACO Instance Reduction

*   **Data:** $(date)
*   **Dataset:** $DATASET
*   **Parâmetros:** $ANTS formigas, $ITER iterações (sem early-stopping)

## Tabela Geral de Resultados

| Modo | Configuração | Scheduler | Tempo ACO (ms) | Tempo Eval (ms) | Throughput (GFLOPS) | Speedup (vs Seq) | IPC | Cache Misses (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
EOF

# Função para executar e fazer o parse das métricas
run_benchmark() {
    local MODE="$1"
    local CONFIG="$2"
    local SCHEDULER="$3"
    local CMD="$4"

    echo "Running: $MODE | Config: $CONFIG | Scheduler: $SCHEDULER..."

    # Executar e salvar saída
    local RUN_OUT="results/temp_run.txt"
    local PERF_OUT="results/temp_perf.txt"

    # Se perf estiver ativo, rodar com perf stat
    if [ "$USE_PERF" = true ]; then
        # Captura apenas eventos cruciais para não sobrecarregar
        perf stat -o "$PERF_OUT" -e cycles,instructions,cache-misses,cache-references -- $CMD > "$RUN_OUT" 2>&1 || true
    else
        eval "$CMD" > "$RUN_OUT" 2>&1 || true
    fi

    # Append to raw logs
    cat "$RUN_OUT" >> "$RAW_LOGS"

    # Fazer o parse das métricas do programa
    local TIME_ACO=$(grep -i "Tempo ACO:" "$RUN_OUT" | awk '{print $3}')
    local TIME_EVAL=$(grep -i "Tempo eval (1-NN):" "$RUN_OUT" | awk '{print $4}')
    # Se não achar Tempo eval (1-NN) (no caso de CUDA que imprime diferente)
    if [ -z "$TIME_EVAL" ]; then
        TIME_EVAL=$(grep -i "GPU eval (1-NN):" "$RUN_OUT" | awk '{print $4}')
    fi
    
    local GFLOPS=$(grep -i "throughput:" "$RUN_OUT" | awk '{print $4}')

    # Tratar valores vazios
    [ -z "$TIME_ACO" ] && TIME_ACO="0.0"
    [ -z "$TIME_EVAL" ] && TIME_EVAL="0.0"
    [ -z "$GFLOPS" ] && GFLOPS="0.0"

    # Parse das métricas do perf
    local IPC="N/A"
    local CACHE_MISS="N/A"
    if [ "$USE_PERF" = true ] && [ -f "$PERF_OUT" ]; then
        # Tenta extrair IPC
        IPC=$(grep -i "insn per cycle" "$PERF_OUT" | awk '{print $4}' || echo "N/A")
        # Se não achar por texto exato, tenta extrair por regex
        if [ "$IPC" = "N/A" ] || [ -z "$IPC" ]; then
             IPC=$(awk '/instructions/ {print $4}' "$PERF_OUT" || echo "N/A")
        fi

        # Tenta extrair cache misses percentual
        local MISSES=$(awk '/cache-misses/ {print $1}' "$PERF_OUT" | tr -d '.' | tr -d ',' || echo "")
        local REFS=$(awk '/cache-references/ {print $1}' "$PERF_OUT" | tr -d '.' | tr -d ',' || echo "")
        if [ -n "$MISSES" ] && [ -n "$REFS" ] && [ "$REFS" -gt 0 ]; then
            CACHE_MISS=$(awk -v m="$MISSES" -v r="$REFS" 'BEGIN {printf "%.2f%%", (m/r)*100}')
        fi
    fi

    [ -z "$IPC" ] && IPC="N/A"
    [ -z "$CACHE_MISS" ] && CACHE_MISS="N/A"

    # Armazena em variáveis globais para a tabela
    LAST_TIME_ACO="$TIME_ACO"
    LAST_TIME_EVAL="$TIME_EVAL"
    LAST_GFLOPS="$GFLOPS"
    LAST_IPC="$IPC"
    LAST_CACHE_MISS="$CACHE_MISS"
}

# --- 1. SEQUENCIAL (BASEILINE) ---
echo "=== 3. Executando baseline Sequencial ==="
run_benchmark "Sequencial" "1 thread" "N/A" "$SEQ_BIN $DATASET $TARGET --ants $ANTS --iter $ITER --eval-sample 0"
T_SEQ="$LAST_TIME_ACO"

# Adicionar ao markdown
echo "| Sequencial | 1 thread | N/A | $LAST_TIME_ACO | $LAST_TIME_EVAL | $LAST_GFLOPS | 1.00x (ref) | $LAST_IPC | $LAST_CACHE_MISS |" >> "$REPORT_MD"
echo ""

# --- 2. OPENMP (THREADS & SCHEDULERS) ---
echo "=== 4. Iniciando Experimentos OpenMP ==="
THREADS_LIST=(1 2 4 8 16 32)
SCHEDULERS=(static dynamic guided)

for thr in "${THREADS_LIST[@]}"; do
    for sched in "${SCHEDULERS[@]}"; do
        # Rodar benchmark OMP
        OMP_NUM_THREADS="$thr" OMP_SCHEDULE="$sched" \
            run_benchmark "OpenMP" "$thr threads" "$sched" \
            "$OMP_BIN $DATASET $TARGET --ants $ANTS --iter $ITER"
        
        # Calcular speedup
        SPEEDUP=$(awk -v seq="$T_SEQ" -v par="$LAST_TIME_ACO" 'BEGIN { if (par > 0) printf "%.2fx", seq/par; else printf "0.0x" }')
        
        # Adicionar ao markdown
        echo "| OpenMP | $thr threads | $sched | $LAST_TIME_ACO | $LAST_TIME_EVAL | $LAST_GFLOPS | $SPEEDUP | $LAST_IPC | $LAST_CACHE_MISS |" >> "$REPORT_MD"
    done
done
echo ""

# --- 3. CUDA (BLOCK SIZES) ---
echo "=== 5. Iniciando Experimentos CUDA ==="
BLOCK_SIZES=(32 64 128 256 512 1024)

for bs in "${BLOCK_SIZES[@]}"; do
    run_benchmark "CUDA" "Block size: $bs" "N/A" \
        "$CUD_BIN $DATASET $TARGET --ants $ANTS --iter $ITER --block-size $bs"
    
    # Calcular speedup
    SPEEDUP=$(awk -v seq="$T_SEQ" -v par="$LAST_TIME_ACO" 'BEGIN { if (par > 0) printf "%.2fx", seq/par; else printf "0.0x" }')
    
    # Adicionar ao markdown
    echo "| CUDA | Block size: $bs | N/A | $LAST_TIME_ACO | $LAST_TIME_EVAL | $LAST_GFLOPS | $SPEEDUP | $LAST_IPC | $LAST_CACHE_MISS |" >> "$REPORT_MD"
done

# Limpar arquivos temporários
rm -f results/temp_run.txt results/temp_perf.txt

echo ""
echo "======================================================="
echo " EXPERIMENTOS CONCLUÍDOS!"
echo " Relatório Markdown gerado em: $REPORT_MD"
echo " Logs brutos salvos em: $RAW_LOGS"
echo "======================================================="
