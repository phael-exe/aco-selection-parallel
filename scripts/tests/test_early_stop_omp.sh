#!/usr/bin/env bash
# Testa o early stopping do binário OpenMP num dataset pequeno.
set -uo pipefail
BIN=./build/aco_omp
DS=data/baseline/diabetes.csv
TGT=Outcome
fail=0

# 1. --patience 2 deve disparar early stop antes de 100 iteracoes
out=$("$BIN" "$DS" "$TGT" --ants 16 --iter 100 --patience 2 2>/dev/null)
if echo "$out" | grep -q "(early stop)"; then
    echo "OK: early stop disparou com --patience 2"
else
    echo "FALHA: sem early stop com --patience 2"
    echo "$out" | grep -i "Iteracoes"
    fail=1
fi

# 2. --patience 0 deve rodar 100/100 sem early stop
out0=$("$BIN" "$DS" "$TGT" --ants 16 --iter 100 --patience 0 2>/dev/null)
if echo "$out0" | grep -q "100/100" && ! echo "$out0" | grep -q "(early stop)"; then
    echo "OK: --patience 0 rodou ate o teto"
else
    echo "FALHA: --patience 0 nao rodou 100/100"
    echo "$out0" | grep -i "Iteracoes"
    fail=1
fi

exit $fail
