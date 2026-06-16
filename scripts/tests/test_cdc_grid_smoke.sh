#!/usr/bin/env bash
# Smoke test do runner num dataset pequeno e grade reduzida (rápido).
set -uo pipefail
CSV=results/cdc_grid_results.csv
rm -f "$CSV" results/cdc_grid_raw.log

DATASET=data/baseline/diabetes.csv TARGET=Outcome ANTS=16 ITER=10 PATIENCE=2 \
THREAD_LIST="2 4" BLOCK_LIST="64 128" SCHED_LIST="static dynamic" \
    bash scripts/run_cdc_grid.sh

# Esperado: header + 2 CUDA (64,128) + 4 OpenMP (2,4 × static,dynamic) = 7 linhas
n=$(grep -c . "$CSV")
echo "linhas no CSV (com header): $n"
[ "$n" -eq 7 ] || { echo "FALHA: esperado 7 linhas"; cat "$CSV"; exit 1; }
grep -q "^CUDA,,,64," "$CSV"        || { echo "FALHA: faltou CUDA bs=64";  exit 1; }
grep -q "^CUDA,,,128," "$CSV"       || { echo "FALHA: faltou CUDA bs=128"; exit 1; }
grep -q "^OpenMP,2,static,," "$CSV" || { echo "FALHA: faltou OMP 2 static"; exit 1; }
grep -q "^OpenMP,4,dynamic,," "$CSV"|| { echo "FALHA: faltou OMP 4 dynamic"; exit 1; }
echo "OK: smoke test do grid passou"
