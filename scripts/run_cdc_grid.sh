#!/usr/bin/env bash
# ===========================================================================
# run_cdc_grid.sh — Grade de experimentos no CDC grande.
#   1) CUDA por block size (roda primeiro; valida GPU no N grande)
#   2) OpenMP por nº de threads × escalonador
# Saídas incrementais (1 linha/run): results/cdc_grid_results.csv + raw log.
# Tolerante a falha de run (status=ERRO, segue). Resume: ver Task 4.
# Sobrescreva qualquer parâmetro via env var.
# ===========================================================================
set -uo pipefail

DATASET="${DATASET:-data/cdc/cdc_diabetes.csv}"
TARGET="${TARGET:-Diabetes_binary}"
ANTS="${ANTS:-64}"
ITER="${ITER:-100}"
PATIENCE="${PATIENCE:-20}"
THREAD_LIST="${THREAD_LIST:-2 4 8 16 32}"
BLOCK_LIST="${BLOCK_LIST:-32 64 128 256 512 1024}"
SCHED_LIST="${SCHED_LIST:-static dynamic}"
OUT_PREFIX="${OUT_PREFIX:-results/cdc_grid}"

CUD_BIN=./build/aco_cuda
OMP_BIN=./build/aco_omp
CSV="${OUT_PREFIX}_results.csv"
RAW="${OUT_PREFIX}_raw.log"
HEADER="modo,threads,escalonador,block_size,tempo_aco_ms,tempo_eval_ms,gflops,f1,acuracia,reducao_pct,iteracoes,status,timestamp"

mkdir -p results

if [ ! -f "$DATASET" ]; then
    echo "[ERRO] Dataset '$DATASET' nao encontrado." >&2
    exit 1
fi

echo "=== Compilando binarios (make all) ==="
make all || { echo "[ERRO] compilacao falhou" >&2; exit 1; }

# Cria CSV com header se ainda nao existir
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"
[ -f "$RAW" ] || : > "$RAW"

# Extrai um campo numerico de um arquivo de saida. $1=arquivo $2=regex grep $3=indice awk
extract() {
    local val
    val=$(grep -iE "$2" "$1" | head -1 | awk "{print \$$3}")
    [ -z "$val" ] && val="0"
    echo "$val"
}

# Retorna 0 se a config (modo,threads,sched,bs) já está no CSV com status=OK.
already_done() {
    local modo="$1" threads="$2" sched="$3" bs="$4"
    [ -f "$CSV" ] || return 1
    awk -F, -v m="$modo" -v t="$threads" -v s="$sched" -v b="$bs" \
        '$1==m && $2==t && $3==s && $4==b && $12=="OK" {found=1} END{exit found?0:1}' "$CSV"
}

# Roda um config e grava 1 linha no CSV + bloco no RAW.
# $1=modo $2=threads $3=sched $4=block_size $5=comando
run_and_record() {
    local modo="$1" threads="$2" sched="$3" bs="$4" cmd="$5"
    if already_done "$modo" "$threads" "$sched" "$bs"; then
        echo "  -> pulando (ja concluido): modo=$modo threads=$threads sched=$sched bs=$bs"
        return 0
    fi
    local tmp; tmp=$(mktemp)
    echo "Running: modo=$modo threads=$threads sched=$sched bs=$bs"

    {
        echo "----- modo=$modo threads=$threads sched=$sched bs=$bs -----"
        echo "CMD: $cmd"
        echo "INICIO: $(date)"
    } >> "$RAW"

    local status="OK"
    eval "$cmd" > "$tmp" 2>&1 || status="ERRO"
    cat "$tmp" >> "$RAW"
    echo "FIM: $(date)" >> "$RAW"

    local tempo_aco tempo_eval gflops f1 acc reducao iters
    tempo_aco=$(extract "$tmp" "Tempo ACO:" 3)
    tempo_eval=$(grep -iE "Tempo eval \(1-NN\):|GPU eval \(1-NN\):" "$tmp" | head -1 | awk '{print $4}'); [ -z "$tempo_eval" ] && tempo_eval=0
    gflops=$(grep -iE "throughput:" "$tmp" | head -1 | awk '{print $4}'); [ -z "$gflops" ] && gflops=0
    f1=$(extract "$tmp" "F1-Score:" 2)
    acc=$(extract "$tmp" "Acuracia:" 2)
    reducao=$(grep -iE "reducao ~" "$tmp" | head -1 | sed -E 's/.*reducao ~?([0-9.]+).*/\1/'); [ -z "$reducao" ] && reducao=0
    iters=$(grep -iE "Iteracoes executadas:" "$tmp" | head -1 | awk '{print $3}' | cut -d/ -f1); [ -z "$iters" ] && iters=0

    echo "$modo,$threads,$sched,$bs,$tempo_aco,$tempo_eval,$gflops,$f1,$acc,$reducao,$iters,$status,$(date -Iseconds)" >> "$CSV"
    rm -f "$tmp"
    echo "  -> status=$status iters=$iters tempo_aco=$tempo_aco ms"
}

echo "=== 1) CUDA (block sizes) ==="
for bs in $BLOCK_LIST; do
    run_and_record "CUDA" "" "" "$bs" \
        "$CUD_BIN $DATASET $TARGET --ants $ANTS --iter $ITER --patience $PATIENCE --block-size $bs"
done

echo "=== 2) OpenMP (threads × escalonador) ==="
for sched in $SCHED_LIST; do
    for t in $THREAD_LIST; do
        run_and_record "OpenMP" "$t" "$sched" "" \
            "OMP_NUM_THREADS=$t OMP_SCHEDULE=$sched $OMP_BIN $DATASET $TARGET --ants $ANTS --iter $ITER --patience $PATIENCE"
    done
done

echo "=== Grade concluida. CSV: $CSV | RAW: $RAW ==="
