#!/usr/bin/env bash
# ===========================================================================
# run_baseline_sched_benchmark.sh — Benchmark padronizado dos 9 datasets
# baseline na MESMA máquina do CDC (SSH). Para cada dataset:
#   - seq (baseline), cuda (GPU), omp 1-thread (T1)
#   - omp em threads {2,4,8,16,32} × escalonadores {static, static,2, static,3,
#     dynamic, dynamic,2, dynamic,3}  (sem guided)
# Saída: CSV único com coluna `schedule` (datasets pequenos => rápido).
# Early stop consistente: omp/cuda recebem --patience 10 (seq usa 10 fixo).
# ===========================================================================
set -uo pipefail

SEQ=./build/aco_seq
OMP=./build/aco_omp
CUD=./build/aco_cuda
ANTS="${ANTS:-64}"
ITER="${ITER:-100}"
# Early stop DESLIGADO (0): o binário sequencial não honra early stopping, então
# rodamos 100 iterações fixas em TODOS os modos -> comparação de tempo/qualidade justa.
PATIENCE="${PATIENCE:-0}"
THREADS="${THREADS:-2 4 8 16 32}"
SCHEDULES="${SCHEDULES:-static static,2 static,3 dynamic dynamic,2 dynamic,3}"
OUT="${OUT:-results/baseline_sched_benchmark.csv}"

DATASETS=(
    "heart_failure:DEATH_EVENT" "haberman:class" "cirrhosis:3tage"
    "diabetes:Outcome" "tic-tac-toe:V10" "yeast:name"
    "vaccine:Vaccine_Hesitant" "Employee:LeaveOr1t" "brain-stroke:stroke"
)

mkdir -p results
echo "dataset,N,F,mode,threads,schedule,f1,acuracia,precisao,recall,reducao_pct,tempo_ms,iters,status" > "$OUT"

# Parseia métricas do stdout do binário e ecoa: N F ACC F1 PREC REC RED TEMPO ITERS
parse() {
    local R="$1"
    local N F ACC F1 PREC REC RED TEMPO ITERS
    N=$(echo "$R"     | grep "Dataset carregado:"      | awk '{print $3}')
    F=$(echo "$R"     | grep "Dataset carregado:"      | awk '{print $5}')
    ACC=$(echo "$R"   | grep "^Acuracia:"              | awk '{print $2}')
    F1=$(echo "$R"    | grep "^F1-Score:"              | awk '{print $2}')
    PREC=$(echo "$R"  | grep "^Precisao:"              | awk '{print $2}')
    REC=$(echo "$R"   | grep "^Recall:"                | awk '{print $2}')
    RED=$(echo "$R"   | grep "Melhor solucao:"         | sed 's/.*reducao ~//; s/%).*//')
    TEMPO=$(echo "$R" | grep "^Tempo ACO:"             | awk '{print $3}')
    ITERS=$(echo "$R" | grep "^Iteracoes executadas:"  | awk '{print $3}' | cut -d/ -f1)
    echo "${N:-} ${F:-} ${ACC:-0} ${F1:-0} ${PREC:-0} ${REC:-0} ${RED:-0} ${TEMPO:-0} ${ITERS:-0}"
}

# $1=mode $2=threads $3=schedule $4=raw_output
record() {
    local mode="$1" threads="$2" sched="$3" raw="$4" name="$5"
    local st="OK"; local N F ACC F1 PREC REC RED TEMPO ITERS
    read -r N F ACC F1 PREC REC RED TEMPO ITERS <<< "$(parse "$raw")"
    { [ "$F1" = "0" ] || [ -z "$F1" ] || [ "$TEMPO" = "0" ]; } && st="FAIL"
    local sched_csv="${sched//,/-}"   # evita virgula no CSV (static,2 -> static-2)
    echo "$name,$N,$F,$mode,$threads,$sched_csv,$F1,$ACC,$PREC,$REC,$RED,$TEMPO,$ITERS,$st" >> "$OUT"
    echo "  $name | $mode t=$threads sched=$sched -> ${TEMPO} ms, F1=$F1, iters=$ITERS [$st]"
}

idx=0
for pair in "${DATASETS[@]}"; do
    idx=$((idx+1))
    name="${pair%%:*}"; target="${pair##*:}"; path="data/baseline/${name}.csv"
    echo "[$idx/${#DATASETS[@]}] $name ($target)"
    [ -f "$path" ] || { echo "  [SKIP] $path ausente"; continue; }

    record seq 1 "-" "$($SEQ "$path" "$target" --ants "$ANTS" --iter "$ITER" --eval-sample 0 2>/dev/null)" "$name"
    record cuda GPU "-" "$($CUD "$path" "$target" --ants "$ANTS" --iter "$ITER" --patience "$PATIENCE" 2>/dev/null)" "$name"
    record omp 1 "default" "$(OMP_NUM_THREADS=1 $OMP "$path" "$target" --ants "$ANTS" --iter "$ITER" --patience "$PATIENCE" 2>/dev/null)" "$name"

    for t in $THREADS; do
        for s in $SCHEDULES; do
            record omp "$t" "$s" "$(OMP_NUM_THREADS="$t" OMP_SCHEDULE="$s" $OMP "$path" "$target" --ants "$ANTS" --iter "$ITER" --patience "$PATIENCE" 2>/dev/null)" "$name"
        done
    done
done

echo "DONE_BASELINE_SCHED -> $OUT"
