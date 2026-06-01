#!/usr/bin/env bash
set -eu

# ===========================================================================
# run_openmp_scaling.sh — Teste de escalabilidade da versão OpenMP (US11/EP03)
#
# Executa ./build/aco_omp variando OMP_NUM_THREADS em {1,2,4,8,16} sobre
# datasets representativos, mede o tempo do ACO e calcula o speedup T1/Tp.
# Registra tudo em results/scaling_openmp.csv.
#
# Uso:
#   scripts/run_openmp_scaling.sh            # datasets representativos
#   scripts/run_openmp_scaling.sh --cdc      # inclui o CDC Diabetes (N~253k, lento)
#
# Pré-requisito: make openmp  (gera ./build/aco_omp)
# ===========================================================================

BINARY="./build/aco_omp"
OUTPUT_CSV="results/scaling_openmp.csv"
ANTS=64
ITER=100
THREAD_LIST=(1 2 4 8 16)

# Datasets representativos (pequeno, médio, grande) — onde o 1-NN domina o tempo.
DATASETS=(
    "data/baseline/heart_failure.csv:DEATH_EVENT"
    "data/baseline/yeast.csv:name"
    "data/baseline/brain-stroke.csv:stroke"
)

# --cdc adiciona o dataset de larga escala (custoso)
if [ "${1:-}" = "--cdc" ]; then
    DATASETS+=("data/cdc/cdc_diabetes.csv:Diabetes_binary")
fi

# ---------------------------------------------------------------------------
# Pré-requisitos
# ---------------------------------------------------------------------------

if [ ! -x "$BINARY" ]; then
    echo "[ERRO] Binário não encontrado: $BINARY"
    echo "       Execute 'make openmp' antes."
    exit 1
fi

mkdir -p results

NPROC=$(nproc 2>/dev/null || echo "?")
echo "=========================================="
echo " Escalabilidade OpenMP — US11 / EP03"
echo " CPUs lógicas disponíveis: $NPROC"
echo "=========================================="
echo ""

echo "Dataset,N,Threads,Tempo_ms,Speedup,Acuracia,F1_Score" > "$OUTPUT_CSV"

# ---------------------------------------------------------------------------
# Loop principal
# ---------------------------------------------------------------------------

for entry in "${DATASETS[@]}"; do
    path="${entry%%:*}"
    target="${entry##*:}"
    name=$(basename "$path" .csv)

    if [ ! -f "$path" ]; then
        echo "[skip] $path não encontrado"
        continue
    fi

    echo "---- $name ($target) ----"

    base_time=""   # tempo com 1 thread (referência para speedup)

    for t in "${THREAD_LIST[@]}"; do
        # stdout traz as métricas; stderr (logs por iteração) é descartado
        out=$(OMP_NUM_THREADS="$t" "$BINARY" "$path" "$target" \
              --ants "$ANTS" --iter "$ITER" 2>/dev/null)

        # Extração por campo (awk) — robusta: evita capturar o "1" de "F1-Score"
        N=$(echo "$out"      | awk '/Dataset carregado:/{print $3; exit}')
        time_ms=$(echo "$out"| awk '/^Tempo ACO:/{print $3; exit}')
        acc=$(echo "$out"    | awk '/^Acuracia:/{print $2; exit}')
        f1=$(echo "$out"     | awk '/^F1-Score:/{print $2; exit}')

        # Speedup = T1 / Tp (calculado com awk para ponto flutuante)
        if [ "$t" = "1" ]; then
            base_time="$time_ms"
            speedup="1.00"
        else
            speedup=$(LC_ALL=C awk -v b="$base_time" -v c="$time_ms" \
                      'BEGIN { if (c > 0) printf "%.2f", b / c; else print "NA" }')
        fi

        printf "  %2s threads | %10s ms | speedup %5sx | acc %s\n" \
            "$t" "$time_ms" "$speedup" "$acc"

        echo "$name,$N,$t,$time_ms,$speedup,$acc,$f1" >> "$OUTPUT_CSV"
    done
    echo ""
done

echo "Resultados salvos em: $OUTPUT_CSV"
