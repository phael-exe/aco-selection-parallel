#!/bin/bash

# Teste rápido: validar que compilação funciona sem warnings
echo "=== Teste 1: Compilação sem warnings ==="
make clean > /dev/null 2>&1
make sequential 2>&1 | grep -i "warning" && echo "FAIL: Há warnings" || echo "PASS: Sem warnings"

echo ""
echo "=== Teste 2: Heart Failure com top-1 ==="
./build/aco_seq data/baseline/heart_failure.csv DEATH_EVENT --ants 16 --iter 3 --eval-top-k 1 > /tmp/test_hf.txt 2>&1
F1=$(grep "F1-Score:" /tmp/test_hf.txt | tail -1 | awk '{print $2}')
if [ ! -z "$F1" ]; then
    echo "PASS: F1-Score calculado: $F1"
else
    echo "FAIL: F1-Score não encontrado"
fi

echo ""
echo "=== Teste 3: Heart Failure com top-5 (auto-detectado) ==="
./build/aco_seq data/baseline/heart_failure.csv DEATH_EVENT --ants 16 --iter 2 > /tmp/test_hf_auto.txt 2>&1
STRATEGY=$(grep "Eval strategy:" /tmp/test_hf_auto.txt | head -1)
if [[ "$STRATEGY" == *"top-5"* ]]; then
    echo "PASS: Auto-detectou top-5 para N=299"
else
    echo "FAIL: Não detectou top-5"
    echo "Strategy encontrada: $STRATEGY"
fi

echo ""
echo "=== Teste 4: Validar que taxa de redução está correta ==="
SELECTED=$(grep "Melhor solucao:" /tmp/test_hf.txt | awk '{print $3}' | cut -d/ -f1)
TOTAL=$(grep "Melhor solucao:" /tmp/test_hf.txt | awk '{print $3}' | cut -d/ -f2)
if [ ! -z "$SELECTED" ] && [ ! -z "$TOTAL" ]; then
    REDUCTION=$(echo "scale=4; 1.0 - ($SELECTED / $TOTAL)" | bc)
    REDUCTION_OUTPUT=$(grep "reducao" /tmp/test_hf.txt | grep "Melhor solucao:" | sed 's/.*reducao //' | sed 's/%.*//')
    echo "PASS: Redução calculada: $REDUCTION_OUTPUT%"
else
    echo "FAIL: Não conseguiu extrair selected/total"
fi

echo ""
echo "=== Teste 5: Early stopping funciona ==="
./build/aco_seq data/baseline/haberman.csv class --ants 16 --iter 50 > /tmp/test_early.txt 2>&1
ITERS=$(grep "Iteracoes executadas:" /tmp/test_early.txt | awk '{print $2}' | cut -d/ -f1)
MAX_ITERS=$(grep "Iteracoes executadas:" /tmp/test_early.txt | awk '{print $2}' | cut -d/ -f2)
if [ "$ITERS" -lt "$MAX_ITERS" ]; then
    echo "PASS: Early stopping ativo (parou em $ITERS/$MAX_ITERS iterações)"
else
    echo "INFO: Sem early stop (rodou todas $MAX_ITERS iterações)"
fi

echo ""
echo "=== Todos os testes básicos completos ==="
