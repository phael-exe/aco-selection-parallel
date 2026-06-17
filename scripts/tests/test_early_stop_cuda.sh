#!/usr/bin/env bash
# Testa o early stopping do binário CUDA num dataset pequeno (GPU local).
set -uo pipefail
BIN=./build/aco_cuda
DS=data/baseline/diabetes.csv
TGT=Outcome
fail=0

out=$("$BIN" "$DS" "$TGT" --ants 16 --iter 100 --patience 2 --block-size 64 2>/dev/null)
if echo "$out" | grep -q "(early stop)"; then
    echo "OK: early stop disparou com --patience 2"
else
    echo "FALHA: sem early stop com --patience 2"
    echo "$out" | grep -i "Iteracoes"
    fail=1
fi

out0=$("$BIN" "$DS" "$TGT" --ants 16 --iter 100 --patience 0 --block-size 64 2>/dev/null)
if echo "$out0" | grep -q "100/100" && ! echo "$out0" | grep -q "(early stop)"; then
    echo "OK: --patience 0 rodou ate o teto"
else
    echo "FALHA: --patience 0 nao rodou 100/100"
    echo "$out0" | grep -i "Iteracoes"
    fail=1
fi

exit $fail
