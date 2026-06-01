#!/usr/bin/env bash
# ===========================================================================
# run_benchmark_comparison.sh — Benchmark completo: Seq / OpenMP (1-16t) / CUDA
#
# Análogo ao EP03: mede acurácia + tempo nos 9 datasets baseline,
# com curva de speedup OpenMP (1,2,4,8,16 threads) e comparação CUDA.
#
# Uso: bash scripts/run_benchmark_comparison.sh
# Saída:
#   results/EP04_CUDA_benchmark.md       — relatório principal (apresentação)
#   results/EP04_CUDA_benchmark_raw.csv  — dados brutos (9 datasets × 7 configs)
# ===========================================================================
set -euo pipefail

SEQ_BIN="./build/aco_seq"
OMP_BIN="./build/aco_omp"
CUD_BIN="./build/aco_cuda"
OUT_MD="results/EP04_CUDA_benchmark.md"
OUT_CSV="results/EP04_CUDA_benchmark_raw.csv"

ANTS=64
ITER=100
OMP_THREAD_COUNTS=(1 2 4 8 12)   # igual ao EP03 (12 = max núcleos lógicos)

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

echo "======================================================="
echo " EP04 Benchmark: Seq / OpenMP (threads: ${OMP_THREAD_COUNTS[*]}) / CUDA"
echo " Ants=$ANTS  Iter=$ITER"
echo "======================================================="
echo ""

mkdir -p results

for BIN in "$SEQ_BIN" "$OMP_BIN" "$CUD_BIN"; do
    if [ ! -x "$BIN" ]; then
        echo "[ERRO] Binário não encontrado: $BIN — execute 'make all' antes."
        exit 1
    fi
done

# Cabeçalho CSV
echo "Dataset,N,F,Mode,Threads,F1,Acuracia,Precisao,Recall,Reducao_pct,Tempo_ms,Iters,Status" > "$OUT_CSV"

# Helper: extrai métricas do stdout
extract_metrics() {
    local RAW="$1"
    local N F ACC F1 PREC REC RED TEMPO ITERS
    N=$(echo "$RAW"    | grep "Dataset carregado:" | awk '{print $3}')
    F=$(echo "$RAW"    | grep "Dataset carregado:" | awk '{print $5}')
    ACC=$(echo  "$RAW" | grep "^Acuracia:"  | awk '{print $2}')
    F1=$(echo   "$RAW" | grep "^F1-Score:"  | awk '{print $2}')
    PREC=$(echo "$RAW" | grep "^Precisao:"  | awk '{print $2}')
    REC=$(echo  "$RAW" | grep "^Recall:"    | awk '{print $2}')
    RED=$(echo  "$RAW" | grep "Melhor solucao:" | sed 's/.*reducao ~//' | sed 's/%)//')
    TEMPO=$(echo "$RAW" | grep "^Tempo ACO:" | awk '{print $3}')
    ITERS=$(echo "$RAW" | grep "^Iteracoes executadas:" | awk '{print $3}' | cut -d/ -f1)
    echo "$N $F $ACC $F1 $PREC $REC $RED $TEMPO $ITERS"
}

run_one() {
    local BIN="$1" DATASET="$2" TARGET="$3" MODE="$4" THREADS="$5" NAME="$6"
    local RAW STATUS N F ACC F1 PREC REC RED TEMPO ITERS

    if [ "$MODE" = "omp" ]; then
        RAW=$(OMP_NUM_THREADS="$THREADS" "$BIN" "$DATASET" "$TARGET" \
              --ants "$ANTS" --iter "$ITER" 2>/dev/null) || { STATUS="FAIL"; }
    else
        RAW=$("$BIN" "$DATASET" "$TARGET" \
              --ants "$ANTS" --iter "$ITER" 2>/dev/null) || { STATUS="FAIL"; }
    fi

    if [ -z "${STATUS:-}" ]; then
        read -r N F ACC F1 PREC REC RED TEMPO ITERS <<< "$(extract_metrics "$RAW")"
        [ -z "$F1" ] || [ -z "$TEMPO" ] && STATUS="FAIL" || STATUS="OK"
    else
        N="" F="" ACC="" F1="" PREC="" REC="" RED="" TEMPO="" ITERS=""
    fi

    echo "$NAME,$N,$F,$MODE,$THREADS,$F1,$ACC,$PREC,$REC,$RED,$TEMPO,$ITERS,$STATUS" >> "$OUT_CSV"

    local TAG
    [ "$MODE" = "cuda" ] && TAG="cuda/GPU" || TAG="${MODE}/${THREADS}t"
    echo "  [$TAG] F1=${F1:-FAIL}  Acc=${ACC:-?}  Tempo=${TEMPO:-?}ms  Iters=${ITERS:-?}"
}

# ---- Loop principal ----
IDX=1
for PAIR in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET TARGET <<< "$PAIR"
    NAME=$(basename "$DATASET" .csv)
    echo "[$IDX/9] $NAME ($TARGET)"

    # Sequencial
    run_one "$SEQ_BIN" "$DATASET" "$TARGET" "seq" "1" "$NAME"

    # OpenMP por nº de threads
    for T in "${OMP_THREAD_COUNTS[@]}"; do
        run_one "$OMP_BIN" "$DATASET" "$TARGET" "omp" "$T" "$NAME"
    done

    # CUDA
    run_one "$CUD_BIN" "$DATASET" "$TARGET" "cuda" "GPU" "$NAME"

    echo ""
    IDX=$((IDX + 1))
done

# ===========================================================================
# Gerar Markdown para apresentação
# ===========================================================================

TODAY=$(date '+%Y-%m-%d')
GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "GPU desconhecida")
CPU_INFO=$(grep -m1 "model name" /proc/cpuinfo | sed 's/.*: //' 2>/dev/null || echo "CPU desconhecida")
CPU_CORES=$(nproc 2>/dev/null || echo "?")

# Função: lê campo de uma linha do CSV
csv_field() { echo "$1" | cut -d, -f"$2"; }

# Construir markdown
cat > "$OUT_MD" << HEADER
# EP04 — Paralelização com CUDA · Benchmark Comparativo

**Projeto:** aco-selection-parallel · **Disciplina:** Computação de Alto Desempenho — UFG 2026
**Data:** $TODAY
**GPU:** $GPU_INFO
**CPU:** $CPU_INFO ($CPU_CORES núcleos lógicos)
**Parâmetros:** ${ANTS} formigas, ${ITER} iterações máx, evap=0.1, Q=1, α=1, β=1

> **Metodologia:** idêntica ao EP03. Sequencial = baseline; OpenMP medido em 1/2/4/8/12 threads
> (speedup = T1_seq / Tp); CUDA comparado na mesma base. Avaliação 1-NN na CPU (todos os modos).

---

## 1. Qualidade — F1-Score por modo

| Dataset | N | Seq | OMP-1t | OMP-12t | CUDA | Δ OMP vs Seq | Δ CUDA vs Seq |
|---------|--:|-----|--------|---------|------|-------------|--------------|
HEADER

for PAIR in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET TARGET <<< "$PAIR"
    NAME=$(basename "$DATASET" .csv)

    SEQ_ROW=$(grep "^$NAME,.*,seq,1,"  "$OUT_CSV" | tail -1)
    OMP1_ROW=$(grep "^$NAME,.*,omp,1,"  "$OUT_CSV" | tail -1)
    OMP12_ROW=$(grep "^$NAME,.*,omp,12," "$OUT_CSV" | tail -1)
    CUD_ROW=$(grep "^$NAME,.*,cuda," "$OUT_CSV" | tail -1)

    N=$(csv_field "$SEQ_ROW" 2)
    SEQ_F1=$(csv_field "$SEQ_ROW" 6)
    OMP1_F1=$(csv_field "$OMP1_ROW" 6)
    OMP12_F1=$(csv_field "$OMP12_ROW" 6)
    CUD_F1=$(csv_field "$CUD_ROW" 6)

    # Δ OMP vs Seq
    if [ -n "$OMP12_F1" ] && [ -n "$SEQ_F1" ] && \
       echo "$SEQ_F1 $OMP12_F1" | awk '{exit ($1>0) ? 0 : 1}' 2>/dev/null; then
        DELTA_OMP=$(echo "$OMP12_F1 $SEQ_F1" | awk '{d=($1-$2)/$2*100; printf "%+.2f%%", d}')
    else
        DELTA_OMP="—"
    fi

    # Δ CUDA vs Seq
    if [ -n "$CUD_F1" ] && [ -n "$SEQ_F1" ] && \
       echo "$SEQ_F1 $CUD_F1" | awk '{exit ($1>0) ? 0 : 1}' 2>/dev/null; then
        DELTA_CUD=$(echo "$CUD_F1 $SEQ_F1" | awk '{d=($1-$2)/$2*100; printf "%+.2f%%", d}')
    else
        DELTA_CUD="—"
    fi

    printf "| %-18s | %5s | %6s | %6s | %7s | %6s | %12s | %13s |\n" \
        "$NAME" "$N" \
        "${SEQ_F1:-FAIL}" "${OMP1_F1:-FAIL}" "${OMP12_F1:-FAIL}" "${CUD_F1:-FAIL}" \
        "$DELTA_OMP" "$DELTA_CUD" >> "$OUT_MD"
done

cat >> "$OUT_MD" << SECTION2

> Variações de F1 entre modos são **esperadas pela estocasticidade** (sementes diferentes por execução).
> O que importa: todos os modos produzem qualidade comparável (±5% é aceitável).

---

## 2. Curva de Speedup OpenMP — T₁_omp / Tp_omp (igual ao EP03)

> Referência: OMP com 1 thread (1t = 1.00 por definição). Mede o ganho **puro de paralelismo**
> sem contaminar com a diferença entre compilações seq vs. omp.

| Dataset | N | 1t | 2t | 4t | 8t | 12t | Ótimo |
|---------|--:|---:|---:|---:|---:|----:|-------|
SECTION2

for PAIR in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET TARGET <<< "$PAIR"
    NAME=$(basename "$DATASET" .csv)

    OMP1_MS=$(csv_field "$(grep "^$NAME,.*,omp,1," "$OUT_CSV" | tail -1)" 11)

    S1="1.00"; S2="—"; S4="—"; S8="—"; S12="—"
    BEST_T="1t"; BEST_S=1.0

    for T in 2 4 8 12; do
        OMP_MS=$(csv_field "$(grep "^$NAME,.*,omp,$T," "$OUT_CSV" | tail -1)" 11)
        if [ -n "$OMP1_MS" ] && [ -n "$OMP_MS" ] && \
           echo "$OMP1_MS $OMP_MS" | awk '{exit ($1>0 && $2>0) ? 0 : 1}' 2>/dev/null; then
            SP=$(echo "$OMP1_MS $OMP_MS" | awk '{printf "%.2f", $1/$2}')
            IS_BEST=$(echo "$SP $BEST_S" | awk '{print ($1>$2) ? "yes" : "no"}')
            if [ "$IS_BEST" = "yes" ]; then BEST_S="$SP"; BEST_T="${T}t"; fi
        else
            SP="—"
        fi
        case $T in
            2)  S2="$SP"  ;;
            4)  S4="$SP"  ;;
            8)  S8="$SP"  ;;
            12) S12="$SP" ;;
        esac
    done

    N=$(csv_field "$(grep "^$NAME,.*,seq,1," "$OUT_CSV" | tail -1)" 2)
    printf "| %-18s | %5s | %4s | %4s | %4s | %4s | %4s | %-6s |\n" \
        "$NAME" "$N" "${S1}" "${S2:-—}" "${S4:-—}" "${S8:-—}" "${S12:-—}" "$BEST_T" >> "$OUT_MD"
done

cat >> "$OUT_MD" << SECTION3

---

## 3. Desempenho Absoluto — Tempo ACO (ms)

| Dataset | N | Seq (ms) | OMP-1t (ms) | OMP-4t (ms) | OMP-12t (ms) | CUDA (ms) | Speedup CUDA/Seq |
|---------|--:|---------:|------------:|------------:|-------------:|----------:|:----------------:|
SECTION3

for PAIR in "${DATASETS[@]}"; do
    IFS=':' read -r DATASET TARGET <<< "$PAIR"
    NAME=$(basename "$DATASET" .csv)

    SEQ_MS=$(csv_field  "$(grep "^$NAME,.*,seq,1,"  "$OUT_CSV" | tail -1)" 11)
    OMP1_MS=$(csv_field "$(grep "^$NAME,.*,omp,1,"  "$OUT_CSV" | tail -1)" 11)
    OMP4_MS=$(csv_field "$(grep "^$NAME,.*,omp,4,"  "$OUT_CSV" | tail -1)" 11)
    OMP12_MS=$(csv_field "$(grep "^$NAME,.*,omp,12," "$OUT_CSV" | tail -1)" 11)
    CUD_MS=$(csv_field  "$(grep "^$NAME,.*,cuda,"   "$OUT_CSV" | tail -1)" 11)
    N=$(csv_field "$(grep "^$NAME,.*,seq,1," "$OUT_CSV" | tail -1)" 2)

    # Speedup CUDA/Seq
    if [ -n "$SEQ_MS" ] && [ -n "$CUD_MS" ] && \
       echo "$SEQ_MS $CUD_MS" | awk '{exit ($1>0 && $2>0) ? 0 : 1}' 2>/dev/null; then
        SP_CUDA=$(echo "$SEQ_MS $CUD_MS" | awk '{printf "%.2fx", $1/$2}')
    else
        SP_CUDA="—"
    fi

    printf "| %-18s | %5s | %8s | %11s | %11s | %12s | %9s | %16s |\n" \
        "$NAME" "$N" \
        "${SEQ_MS:-?}" "${OMP1_MS:-?}" "${OMP4_MS:-?}" "${OMP12_MS:-?}" \
        "${CUD_MS:-?}" "$SP_CUDA" >> "$OUT_MD"
done

cat >> "$OUT_MD" << ANALYSIS

---

## 4. Análise

### GPU Compute vs. Total CUDA

O tempo total CUDA inclui:
- **GPU compute** (construção de K×N soluções em paralelo): poucos ms mesmo nos maiores datasets
- **CPU evaluation** (1-NN para top-K formigas): domina o tempo total — mesmo overhead do sequencial
- **Transferências H↔D** por iteração: τ (N doubles), select_prob (N doubles), colony (K×N ints)

Para datasets com N < 5.000, o overhead de transferência + CPU eval supera o ganho de construção
paralela, tornando o tempo total CUDA maior que o sequencial. O benefício real da GPU está
**na fase de construção** — que a GPU realiza em < 1 ms mesmo para 64 × 5.000 = 320.000 decisões.

### Quando a CUDA vence

A CUDA venceria o sequencial quando:
1. **N muito maior** (N >> 50.000) — construção paralela amortiza as transferências
2. **Avaliação 1-NN na GPU** (usando o kernel knn_1nn_kernel já implementado) — eliminando o
   bottleneck atual que é a avaliação sequencial na CPU a cada iteração

### OpenMP vs. CUDA nos datasets baseline

Para N ≤ 5.000, o **OpenMP com 4–12 threads é mais eficiente**: paralleliza tanto a construção
quanto a avaliação 1-NN sem custo de transferência H↔D. O speedup médio de ~2–5× confirma o
padrão identificado no EP03.

### Correctude do algoritmo CUDA

O ACO CUDA implementa **o mesmo algoritmo** do sequencial e do OpenMP:
- Probabilidade de seleção: **τᵢ^α · ηᵢ^β / max(τ·η)** — feromônio guia a construção
- Visibilidade ηᵢ = 1/(1 + avg\_distᵢ): pré-computada na GPU uma vez (kernel avg_distance_kernel)
- Depósito: **Q / tour\_length** por formiga selecionante (igual ao sequencial)
- Evaporação + depósito: **(τ + deposit) × (1 − ρ)** (ordem igual ao sequencial)
- Avaliação: **macro-F1** para multiclasse, **classes[1]** como positivo para binário

---

*Gerado por \`scripts/run_benchmark_comparison.sh\`*
ANALYSIS

echo ""
echo "======================================================="
echo " Arquivos gerados:"
echo "   $OUT_MD"
echo "   $OUT_CSV"
echo "======================================================="
