#!/usr/bin/env bash
# Verifica que reexecutar o runner pula configs já concluídos (resume).
set -uo pipefail
CSV=results/cdc_grid_results.csv
rm -f "$CSV" results/cdc_grid_raw.log

ENVS=(DATASET=data/baseline/diabetes.csv TARGET=Outcome ANTS=16 ITER=10 PATIENCE=2
      THREAD_LIST="2 4" BLOCK_LIST="64 128" SCHED_LIST="static dynamic")

echo "--- 1a execucao ---"
env "${ENVS[@]}" bash scripts/run_cdc_grid.sh > /dev/null
n1=$(grep -c . "$CSV")

echo "--- 2a execucao (deve pular tudo) ---"
out=$(env "${ENVS[@]}" bash scripts/run_cdc_grid.sh)
n2=$(grep -c . "$CSV")

echo "linhas apos 1a: $n1 | apos 2a: $n2"
[ "$n1" -eq "$n2" ] || { echo "FALHA: 2a execucao adicionou linhas (sem resume)"; exit 1; }
echo "$out" | grep -qi "pulando" || { echo "FALHA: nao indicou que pulou configs"; exit 1; }
echo "OK: resume funcionou (nenhuma linha duplicada)"
