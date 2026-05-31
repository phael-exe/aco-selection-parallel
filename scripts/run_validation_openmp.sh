#!/usr/bin/env bash
set -eu

# ===========================================================================
# run_validation_openmp.sh — Validação da versão OpenMP (US12 / EP03)
#
# Executa ./build/aco_omp em TODOS os 9 datasets do baseline + CDC Diabetes,
# variando OMP_NUM_THREADS em {1,2,4,8,16}. Para cada (dataset, threads) mede:
#   - tempo do ACO e speedup T1/Tp
#   - acurácia e F1 (devem ser IDÊNTICOS entre threads → prova de determinismo)
#   - delta de acurácia vs versão sequencial (results/validation_sequential.csv)
#     com critério de ±2% (Status OK/WARN)  [checklist US12 item 3]
#
# Saída: results/validation_openmp.csv (escrita incremental, sobrevive a kill).
#
# Uso:
#   scripts/run_validation_openmp.sh           # 9 datasets do baseline (rápido)
#   scripts/run_validation_openmp.sh --cdc     # SÓ o CDC (lento; anexa ao CSV)
#
# O CDC roda com `nice -n 10` para não travar a UI durante as horas de execução.
# Pré-requisitos: make sequential && make openmp ; scripts/run_validation.sh
# ===========================================================================

BINARY="./build/aco_omp"
OUTPUT_CSV="results/validation_openmp.csv"
SEQ_CSV="results/validation_sequential.csv"
ANTS=64
ITER=100
THREAD_LIST=(1 2 4 8 16)
TOL_PCT=2.0   # tolerância de acurácia OpenMP vs sequencial (US12)

BASELINE_DATASETS=(
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
CDC_DATASET="data/cdc/cdc_diabetes.csv:Diabetes_binary"

HEADER="Dataset,N,Threads,Tempo_ms,Speedup,Acuracia,F1_Score,Acuracia_Seq,Delta_Acc_pct,Status"

# ---------------------------------------------------------------------------
# Modo
# ---------------------------------------------------------------------------

MODE="baseline"
NICE=""
if [ "${1:-}" = "--cdc" ]; then
    MODE="cdc"
    NICE="nice -n 10"   # protege a responsividade da UI no run longo
fi

if [ ! -x "$BINARY" ]; then
    echo "[ERRO] Binário não encontrado: $BINARY — execute 'make openmp' antes." >&2
    exit 1
fi
mkdir -p results

# ---------------------------------------------------------------------------
# Helper: acurácia sequencial de referência para um dataset (col 4 do SEQ_CSV)
# ---------------------------------------------------------------------------

get_seq_acc() {
    local name="$1"
    [ -f "$SEQ_CSV" ] || { echo ""; return; }
    awk -F',' -v n="$name" '$1==n {print $4; exit}' "$SEQ_CSV"
}

# ---------------------------------------------------------------------------
# Processa um dataset: varre o thread sweep e anexa as linhas no CSV
# ---------------------------------------------------------------------------

run_dataset() {
    local path="$1" target="$2" seq_ref_mode="$3"
    local name; name=$(basename "$path" .csv)

    if [ ! -f "$path" ]; then
        echo "[skip] $path não encontrado" >&2
        return
    fi

    echo "---- $name ($target) ----"

    local base_time="" base_acc="" acc_seq=""
    if [ "$seq_ref_mode" = "csv" ]; then
        acc_seq=$(get_seq_acc "$name")
    fi

    for t in "${THREAD_LIST[@]}"; do
        local out
        out=$(OMP_NUM_THREADS="$t" $NICE "$BINARY" "$path" "$target" \
              --ants "$ANTS" --iter "$ITER" 2>/dev/null)

        local N time_ms acc f1
        N=$(echo "$out"      | awk '/Dataset carregado:/{print $3; exit}')
        time_ms=$(echo "$out"| awk '/^Tempo ACO:/{print $3; exit}')
        acc=$(echo "$out"    | awk '/^Acuracia:/{print $2; exit}')
        f1=$(echo "$out"     | awk '/^F1-Score:/{print $2; exit}')

        # Speedup = T1 / Tp
        local speedup
        if [ "$t" = "1" ]; then
            base_time="$time_ms"
            base_acc="$acc"
            speedup="1.00"
        else
            speedup=$(LC_ALL=C awk -v b="$base_time" -v c="$time_ms" \
                      'BEGIN { if (c+0 > 0) printf "%.2f", b / c; else print "NA" }')
        fi

        # Referência de acurácia: CSV sequencial (baseline) ou 1-thread OpenMP (CDC)
        local ref="$acc_seq"
        if [ "$seq_ref_mode" != "csv" ] || [ -z "$ref" ]; then
            ref="$base_acc"   # CDC: serial = OpenMP 1 thread (mesmo algoritmo, seed fixo)
        fi

        local delta status
        delta=$(LC_ALL=C awk -v a="$acc" -v r="$ref" \
                'BEGIN { d=a-r; if(d<0)d=-d; printf "%.4f", d*100 }')
        status=$(LC_ALL=C awk -v d="$delta" -v tol="$TOL_PCT" \
                'BEGIN { print (d+0 <= tol+0) ? "OK" : "WARN" }')

        printf "  %2s threads | %10s ms | speedup %5sx | acc %s | dAcc %s%% | %s\n" \
            "$t" "$time_ms" "$speedup" "$acc" "$delta" "$status"

        echo "$name,$N,$t,$time_ms,$speedup,$acc,$f1,$ref,$delta,$status" >> "$OUTPUT_CSV"
    done
    echo ""
}

# ---------------------------------------------------------------------------
# Execução
# ---------------------------------------------------------------------------

NPROC=$(nproc 2>/dev/null || echo "?")
echo "=========================================="
echo " Validação OpenMP — US12 / EP03  (modo: $MODE)"
echo " CPUs lógicas disponíveis: $NPROC | threads testados: ${THREAD_LIST[*]}"
echo "=========================================="
echo ""

if [ "$MODE" = "cdc" ]; then
    # Anexa ao CSV existente (não reescreve o header nem as 9 baselines)
    [ -f "$OUTPUT_CSV" ] || echo "$HEADER" > "$OUTPUT_CSV"
    run_dataset "${CDC_DATASET%%:*}" "${CDC_DATASET##*:}" "cdc"
else
    echo "$HEADER" > "$OUTPUT_CSV"   # recria do zero para as baselines
    for entry in "${BASELINE_DATASETS[@]}"; do
        run_dataset "${entry%%:*}" "${entry##*:}" "csv"
    done
fi

echo "Resultados salvos em: $OUTPUT_CSV"
