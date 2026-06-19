#!/usr/bin/env bash
# ===========================================================================
# run_sched_sweep.sh — Varredura de escalonadores OpenMP (chunk size) num
# nº de threads FIXO, no dataset CDC. Complementa o grid (que usou static e
# dynamic com chunk padrão) explorando a granularidade do escalonamento.
#
# OMP_SCHEDULE aceita "tipo,chunk" (ex.: static,2 / dynamic,8 / guided).
# Esses valores governam os laços schedule(runtime) — sobretudo o eval 1-NN.
# Sobrescreva qualquer parâmetro via env var.
# ===========================================================================
set -uo pipefail

DATASET="${DATASET:-data/cdc/cdc_diabetes.csv}"
TARGET="${TARGET:-Diabetes_binary}"
ANTS="${ANTS:-64}"
ITER="${ITER:-100}"
PATIENCE="${PATIENCE:-20}"
THREADS="${THREADS:-16}"
SCHEDULES="${SCHEDULES:-static,2 static,3 static,64 dynamic,2 dynamic,3 dynamic,64 guided}"
OUT="${OUT:-results/cdc_sched_sweep.csv}"
BIN=./build/aco_omp

mkdir -p results
echo "schedule,threads,tempo_aco_ms,tempo_eval_ms,gflops,iters,f1,reducao" > "$OUT"

for sched in $SCHEDULES; do
    echo "Running: OMP_SCHEDULE=$sched threads=$THREADS"
    out=$(OMP_NUM_THREADS="$THREADS" OMP_SCHEDULE="$sched" "$BIN" "$DATASET" "$TARGET" \
          --ants "$ANTS" --iter "$ITER" --patience "$PATIENCE" 2>/dev/null)
    t=$(echo "$out"  | grep "Tempo ACO:"            | awk '{print $3}')
    te=$(echo "$out" | grep -i "Tempo eval"         | awk '{print $4}')
    g=$(echo "$out"  | grep -i "throughput:"        | awk '{print $4}')
    it=$(echo "$out" | grep "Iteracoes executadas:" | awk '{print $3}' | cut -d/ -f1)
    f=$(echo "$out"  | grep "F1-Score:"             | awk '{print $2}')
    r=$(echo "$out"  | grep "reducao ~"             | sed -E 's/.*reducao ~?([0-9.]+).*/\1/')
    echo "$sched,$THREADS,$t,$te,$g,$it,$f,$r" >> "$OUT"
    echo "  -> $sched: ${t} ms, ${g} GFLOPS, iters=${it}"
done

echo "DONE_SCHED -> $OUT"
