#!/usr/bin/env bash
set -eu

# ===========================================================================
# run_validation.sh — Validação completa da versão sequencial refatorada
#
# Executa ./build/aco_seq em todos os 9 datasets do baseline, coleta métricas
# reais (acurácia, F1, precisão, recall, redução, tempo) e registra em CSV.
# Serve como gate de qualidade antes de EP02 (CUDA) / EP03 (OpenMP).
# ===========================================================================

BINARY="./build/aco_seq"
OUTPUT_CSV="results/validation_sequential.csv"
ANTS=64
ITER=100

DATASETS=(
    "data/baseline/heart_failure.csv:DEATH_EVENT"
    "data/baseline/haberman.csv:class"
    "data/baseline/cirrhosis.csv:3tage"
    "data/baseline/diabetes.csv:Outcome"
    "data/baseline/tic-tac-toe.csv:V10"
    "data/baseline/yeast.csv:name"
    "data/baseline/vaccine.csv:Vaccine_Hesitant"
    "data/baseline/Employee.csv:LeaveOr1t"
    "data/baseline/brain-stroke.csv:stroke"
)

# ---------------------------------------------------------------------------
# Pré-requisitos
# ---------------------------------------------------------------------------

echo "=========================================="
echo " Validação Sequencial — EP01 Gate"
echo "=========================================="
echo ""

if [ ! -x "$BINARY" ]; then
    echo "[ERRO] Binário não encontrado: $BINARY"
    echo "       Execute 'make sequential' antes."
    exit 1
fi

mkdir -p results

# ---------------------------------------------------------------------------
# Header do CSV
# ---------------------------------------------------------------------------

echo "Dataset,N,F,Acuracia,F1_Score,Precisao,Recall,Reducao_pct,Tempo_ms,Eval_TopK,Status" > "$OUTPUT_CSV"

# ---------------------------------------------------------------------------
# Contadores
# ---------------------------------------------------------------------------

TOTAL=0
PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0
TOTAL_TIME_MS=0

# ---------------------------------------------------------------------------
# Loop sobre datasets
# ---------------------------------------------------------------------------

for PAIR in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET TARGET <<< "$PAIR"
    NAME=$(basename "$DATASET" .csv)
    TOTAL=$((TOTAL + 1))

    echo "[$TOTAL/9] $NAME ($TARGET) ..."

    # Executar (stderr + stdout juntos para capturar tudo)
    RAW=$("$BINARY" "$DATASET" "$TARGET" --ants "$ANTS" --iter "$ITER" 2>&1) || {
        echo "  -> FAIL (crash/exit != 0)"
        echo "$NAME,,,,,,,,,FAIL" >> "$OUTPUT_CSV"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    }

    # Extrair métricas do output
    N=$(echo "$RAW" | grep "Dataset carregado:" | awk '{print $3}')
    F=$(echo "$RAW" | grep "Dataset carregado:" | awk '{print $5}')
    ACC=$(echo "$RAW" | grep "^Acuracia:" | awk '{print $2}')
    F1=$(echo "$RAW" | grep "^F1-Score:" | awk '{print $2}')
    PREC=$(echo "$RAW" | grep "^Precisao:" | awk '{print $2}')
    REC=$(echo "$RAW" | grep "^Recall:" | awk '{print $2}')
    TEMPO=$(echo "$RAW" | grep "^Tempo ACO:" | awk '{print $3}')
    TOPK=$(echo "$RAW" | grep "^Eval strategy:" | sed 's/.*top-//' | sed 's/ .*//')

    # Extrair redução do campo "reducao ~XX.XX%"
    RED=$(echo "$RAW" | grep "Melhor solucao:" | sed 's/.*reducao ~//' | sed 's/%).*//')

    # Verificar se extração funcionou
    if [ -z "$ACC" ] || [ -z "$F1" ] || [ -z "$RED" ] || [ -z "$TEMPO" ]; then
        echo "  -> FAIL (não conseguiu extrair métricas)"
        echo "$NAME,$N,$F,,,,,,,$TOPK,FAIL" >> "$OUTPUT_CSV"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Validar critérios
    STATUS="PASS"

    ACC_OK=$(echo "$ACC > 0.5" | bc -l 2>/dev/null || echo "0")
    F1_OK=$(echo "$F1 > 0.3" | bc -l 2>/dev/null || echo "0")
    RED_LOW=$(echo "$RED > 20" | bc -l 2>/dev/null || echo "0")
    RED_HIGH=$(echo "$RED < 80" | bc -l 2>/dev/null || echo "0")

    if [ "$ACC_OK" != "1" ]; then
        STATUS="FAIL"
    fi

    if [ "$F1_OK" != "1" ]; then
        if [ "$STATUS" = "PASS" ]; then STATUS="WARN"; fi
    fi

    if [ "$RED_LOW" != "1" ] || [ "$RED_HIGH" != "1" ]; then
        if [ "$STATUS" = "PASS" ]; then STATUS="WARN"; fi
    fi

    # Registrar no CSV
    echo "$NAME,$N,$F,$ACC,$F1,$PREC,$REC,$RED,$TEMPO,$TOPK,$STATUS" >> "$OUTPUT_CSV"

    # Acumular tempo
    TOTAL_TIME_INT=$(echo "$TEMPO" | cut -d. -f1)
    TOTAL_TIME_MS=$((TOTAL_TIME_MS + TOTAL_TIME_INT))

    # Contadores
    case $STATUS in
        PASS) PASS_COUNT=$((PASS_COUNT + 1)) ;;
        WARN) WARN_COUNT=$((WARN_COUNT + 1)) ;;
        FAIL) FAIL_COUNT=$((FAIL_COUNT + 1)) ;;
    esac

    echo "  -> $STATUS | Acc=$ACC F1=$F1 Red=${RED}% Tempo=${TEMPO}ms"
done

# ---------------------------------------------------------------------------
# Relatório Final
# ---------------------------------------------------------------------------

echo ""
echo "=========================================="
echo " Relatório Final"
echo "=========================================="
echo "  Datasets testados: $TOTAL"
echo "  PASS: $PASS_COUNT"
echo "  WARN: $WARN_COUNT"
echo "  FAIL: $FAIL_COUNT"
echo "  Tempo total: ${TOTAL_TIME_MS}ms"
echo "  CSV salvo em: $OUTPUT_CSV"
echo "=========================================="
echo ""

cat "$OUTPUT_CSV"

echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    echo "[RESULTADO] FALHOU — $FAIL_COUNT dataset(s) com FAIL"
    exit 1
else
    echo "[RESULTADO] APROVADO — Sequencial validado, pronto para EP02/EP03"
    exit 0
fi
