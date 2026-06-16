# CDC Grid Benchmark (OpenMP threads×escalonador + CUDA) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Habilitar early stopping (`--patience`) nos binários OpenMP e CUDA e criar um runner resumível que varre, no dataset CDC grande, CUDA por block size e OpenMP por nº de threads × escalonador, gerando CSV + relatório.

**Architecture:** Duas mudanças cirúrgicas de código (descomentar early stop já implementado + flag `--patience`, default desligado para não quebrar reprodutibilidade dos EP03/EP04), um script bash `run_cdc_grid.sh` (CUDA primeiro, depois OpenMP; append incremental em CSV/log; resumível por chave de config), e um script Python `build_cdc_report.py` que extrai o baseline T1 do log remoto existente e calcula speedup normalizado por iteração.

**Tech Stack:** C++17 (`g++ -fopenmp`), CUDA (`nvcc`, sm_89 local / sm_80+sm_89 no Makefile), Bash, Python 3. Toolchain local confirmado: nvcc 13.2 + GPU RTX 4050 (sm_89) + datasets em `data/baseline/`.

## Global Constraints

- ACO da grade: `--ants 64 --iter 100 --patience 20` (verbatim do spec §2).
- Threads OpenMP: `{2, 4, 8, 16, 32}` (1-thread removido — reaproveitado do log).
- Escalonadores: `{static, dynamic}` (`guided` removido).
- CUDA block sizes: `{32, 64, 128, 256, 512, 1024}`.
- `--patience` default = **0 (desligado)** em ambos os binários — só liga quando passado explicitamente.
- Dataset alvo: `data/cdc/cdc_diabetes.csv`, coluna `Diabetes_binary` (roda no remoto).
- Dataset de teste local: `data/baseline/diabetes.csv`, coluna `Outcome`, separador `;`.
- Execução isolada (nunca CPU+GPU juntos); CUDA antes do OpenMP.
- Runner deve ser resumível (pular configs com `status=OK` já no CSV) e tolerante a falha CUDA (registra `ERRO`, continua).
- Baseline T1 (per-iteração) extraído de `results_remote/hpc_raw_runs.log` (blocos `Threads: 1`): static = `2.6451e7 ms / 100`, dynamic = `2.67882e7 ms / 100`.
- Saídas em `results/`: `cdc_grid_results.csv`, `cdc_grid_raw.log`, `cdc_grid_report.md`.
- CSV header (verbatim):
  `modo,threads,escalonador,block_size,tempo_aco_ms,tempo_eval_ms,gflops,f1,acuracia,reducao_pct,iteracoes,status,timestamp`
  (speedup é calculado pelo report, não pelo runner.)

---

### Task 1: OpenMP — flag `--patience` + ativar early stop

**Files:**
- Modify: `src/openmp/main.cpp` (decl ~linha 40; parse ~linha 57; default `config.patience` linha 102)
- Modify: `src/openmp/aco_omp.cpp:407-414` (bloco comentado)
- Create: `scripts/tests/test_early_stop_omp.sh`

**Interfaces:**
- Consumes: `aco_omp.h` já tem `size_t patience;` em `ACOConfig`.
- Produces: binário `build/aco_omp` que aceita `--patience N`; com `N>0` para após N iterações sem melhora de F1 e imprime `(early stop)` + `Iteracoes executadas: K/100` com `K<100`. Com `N=0` roda até o teto.

- [ ] **Step 1: Escrever o teste que falha**

Create `scripts/tests/test_early_stop_omp.sh`:

```bash
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
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run:
```bash
chmod +x scripts/tests/test_early_stop_omp.sh
make openmp && ./scripts/tests/test_early_stop_omp.sh
```
Expected: FALHA no caso 1 — hoje `--patience` é ignorado (arg desconhecido) e o early stop está comentado, então roda 100/100 e não imprime `(early stop)`.

- [ ] **Step 3: Implementar — `main.cpp` (declarar + parsear + aplicar)**

Em `src/openmp/main.cpp`, após a linha `int    eval_top_k_explicit = 0;  // 0 = não especificado (auto-detectar)` adicione:
```cpp
    int    patience_arg        = 0;  // 0 = early stopping desligado (--patience N liga)
```

No loop de parse, após o bloco `--eval-top-k` (antes do `else` final), adicione:
```cpp
        } else if (std::strcmp(argv[i], "--patience") == 0) {
            patience_arg = std::atoi(argv[i + 1]);
```

Troque a linha `config.patience = 10;  // default` por:
```cpp
    config.patience = static_cast<size_t>(patience_arg);  // 0 = desligado
```

- [ ] **Step 4: Implementar — `aco_omp.cpp` (descomentar + gatear)**

Em `src/openmp/aco_omp.cpp`, substitua o bloco comentado:
```cpp
        // Early stopping desativado a pedido do usuário
        /*
        if (no_improve_count >= config.patience) {
            fprintf(stderr, "Early stopping: sem melhoria por %zu iterações\n", config.patience);
            result.iterations = iter + 1;
            break;
        }
        */
```
por:
```cpp
        // Early stopping: ativo apenas quando patience > 0 (via --patience)
        if (config.patience > 0 && no_improve_count >= config.patience) {
            fprintf(stderr, "Early stopping: sem melhoria por %zu iteracoes\n", config.patience);
            result.iterations = iter + 1;
            break;
        }
```

- [ ] **Step 5: Recompilar e rodar o teste — deve passar**

Run:
```bash
make openmp && ./scripts/tests/test_early_stop_omp.sh
```
Expected: PASS nos dois casos (`OK: early stop disparou...`, `OK: --patience 0 rodou ate o teto`).

- [ ] **Step 6: Commit**

```bash
git add src/openmp/main.cpp src/openmp/aco_omp.cpp scripts/tests/test_early_stop_omp.sh
git commit -m "feat(openmp): add --patience flag enabling early stopping (default off)"
```

---

### Task 2: CUDA — flag `--patience` + ativar early stop

**Files:**
- Modify: `src/cuda/main.cu` (decl ~linha 293; parse ~linha 303; remover `int patience = 10;` linha 385; bloco comentado 518-524)
- Create: `scripts/tests/test_early_stop_cuda.sh`

**Interfaces:**
- Produces: binário `build/aco_cuda` que aceita `--patience N`; com `N>0` para após N iterações sem melhora e imprime `(early stop)`. Com `N=0` roda até o teto.

- [ ] **Step 1: Escrever o teste que falha**

Create `scripts/tests/test_early_stop_cuda.sh`:

```bash
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
```

- [ ] **Step 2: Rodar o teste e confirmar que falha**

Run:
```bash
chmod +x scripts/tests/test_early_stop_cuda.sh
make cuda && ./scripts/tests/test_early_stop_cuda.sh
```
Expected: FALHA no caso 1 — `--patience` é ignorado e o early stop está comentado.

- [ ] **Step 3: Implementar — declarar + parsear `--patience`**

Em `src/cuda/main.cu`, após a linha `int    block_size       = 256;` adicione:
```cpp
    int    patience         = 0;    // 0 = early stopping desligado (--patience N liga)
```

No loop de parse, após a linha do `--block-size`, adicione:
```cpp
        else if (!strcmp(argv[i], "--patience"))   patience       = atoi(argv[i+1]);
```

- [ ] **Step 4: Implementar — remover redeclaração e gatear o break**

Ainda em `src/cuda/main.cu`, **remova** a linha (que agora redeclararia `patience`):
```cpp
    int    patience      = 10;
```

E substitua o bloco comentado:
```cpp
        // -- Early stopping desativado a pedido do usuário --
        /*
        if (no_improve >= patience) {
            fprintf(stderr, "Early stopping: sem melhoria por %d iteracoes\n", patience);
            break;
        }
        */
```
por:
```cpp
        // Early stopping: ativo apenas quando patience > 0 (via --patience)
        if (patience > 0 && no_improve >= patience) {
            fprintf(stderr, "Early stopping: sem melhoria por %d iteracoes\n", patience);
            break;
        }
```

- [ ] **Step 5: Recompilar e rodar o teste — deve passar**

Run:
```bash
make cuda && ./scripts/tests/test_early_stop_cuda.sh
```
Expected: PASS nos dois casos.

- [ ] **Step 6: Commit**

```bash
git add src/cuda/main.cu scripts/tests/test_early_stop_cuda.sh
git commit -m "feat(cuda): add --patience flag enabling early stopping (default off)"
```

---

### Task 3: Runner `run_cdc_grid.sh` — varredura completa + CSV/log incremental (sem resume)

**Files:**
- Create: `scripts/run_cdc_grid.sh`
- Create: `scripts/tests/test_cdc_grid_smoke.sh`

**Interfaces:**
- Consumes: `build/aco_cuda` e `build/aco_omp` com `--patience` (Tasks 1-2).
- Produces: `results/cdc_grid_results.csv` (header + 1 linha por config) e `results/cdc_grid_raw.log`. Variáveis de ambiente sobrescrevíveis: `DATASET, TARGET, ANTS, ITER, PATIENCE, THREAD_LIST, BLOCK_LIST, SCHED_LIST, OUT_PREFIX`. Linha CUDA: `CUDA,,,<bs>,...`; linha OpenMP: `OpenMP,<t>,<sched>,,...`.

- [ ] **Step 1: Escrever o teste de fumaça que falha**

Create `scripts/tests/test_cdc_grid_smoke.sh`:

```bash
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
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run:
```bash
chmod +x scripts/tests/test_cdc_grid_smoke.sh
./scripts/tests/test_cdc_grid_smoke.sh
```
Expected: FALHA — `scripts/run_cdc_grid.sh` ainda não existe.

- [ ] **Step 3: Implementar o runner**

Create `scripts/run_cdc_grid.sh`:

```bash
#!/usr/bin/env bash
# ===========================================================================
# run_cdc_grid.sh — Grade de experimentos no CDC grande.
#   1) CUDA por block size (roda primeiro; valida GPU no N grande)
#   2) OpenMP por nº de threads × escalonador
# Saídas incrementais (1 linha/run): results/cdc_grid_results.csv + raw log.
# Tolerante a falha de run (status=ERRO, segue). Resume: ver Task 4.
# Sobrescreva qualquer parâmetro via env var.
# ===========================================================================
set -uo pipefail

DATASET="${DATASET:-data/cdc/cdc_diabetes.csv}"
TARGET="${TARGET:-Diabetes_binary}"
ANTS="${ANTS:-64}"
ITER="${ITER:-100}"
PATIENCE="${PATIENCE:-20}"
THREAD_LIST="${THREAD_LIST:-2 4 8 16 32}"
BLOCK_LIST="${BLOCK_LIST:-32 64 128 256 512 1024}"
SCHED_LIST="${SCHED_LIST:-static dynamic}"
OUT_PREFIX="${OUT_PREFIX:-results/cdc_grid}"

CUD_BIN=./build/aco_cuda
OMP_BIN=./build/aco_omp
CSV="${OUT_PREFIX}_results.csv"
RAW="${OUT_PREFIX}_raw.log"
HEADER="modo,threads,escalonador,block_size,tempo_aco_ms,tempo_eval_ms,gflops,f1,acuracia,reducao_pct,iteracoes,status,timestamp"

mkdir -p results

if [ ! -f "$DATASET" ]; then
    echo "[ERRO] Dataset '$DATASET' nao encontrado." >&2
    exit 1
fi

echo "=== Compilando binarios (make all) ==="
make all || { echo "[ERRO] compilacao falhou" >&2; exit 1; }

# Cria CSV com header se ainda nao existir
[ -f "$CSV" ] || echo "$HEADER" > "$CSV"
[ -f "$RAW" ] || : > "$RAW"

# Extrai um campo numerico de um arquivo de saida. $1=arquivo $2=regex grep $3=indice awk
extract() {
    local val
    val=$(grep -iE "$2" "$1" | head -1 | awk "{print \$$3}")
    [ -z "$val" ] && val="0"
    echo "$val"
}

# Roda um config e grava 1 linha no CSV + bloco no RAW.
# $1=modo $2=threads $3=sched $4=block_size $5=comando
run_and_record() {
    local modo="$1" threads="$2" sched="$3" bs="$4" cmd="$5"
    local tmp; tmp=$(mktemp)
    echo "Running: modo=$modo threads=$threads sched=$sched bs=$bs"

    {
        echo "----- modo=$modo threads=$threads sched=$sched bs=$bs -----"
        echo "CMD: $cmd"
        echo "INICIO: $(date)"
    } >> "$RAW"

    local status="OK"
    eval "$cmd" > "$tmp" 2>&1 || status="ERRO"
    cat "$tmp" >> "$RAW"
    echo "FIM: $(date)" >> "$RAW"

    local tempo_aco tempo_eval gflops f1 acc reducao iters
    tempo_aco=$(extract "$tmp" "Tempo ACO:" 3)
    tempo_eval=$(grep -iE "Tempo eval \(1-NN\):|GPU eval \(1-NN\):" "$tmp" | head -1 | awk '{print $4}'); [ -z "$tempo_eval" ] && tempo_eval=0
    gflops=$(grep -iE "throughput:" "$tmp" | head -1 | awk '{print $4}'); [ -z "$gflops" ] && gflops=0
    f1=$(extract "$tmp" "F1-Score:" 2)
    acc=$(extract "$tmp" "Acuracia:" 2)
    reducao=$(grep -iE "reducao ~" "$tmp" | head -1 | sed -E 's/.*reducao ~?([0-9.]+).*/\1/'); [ -z "$reducao" ] && reducao=0
    iters=$(grep -iE "Iteracoes executadas:" "$tmp" | head -1 | awk '{print $3}' | cut -d/ -f1); [ -z "$iters" ] && iters=0

    echo "$modo,$threads,$sched,$bs,$tempo_aco,$tempo_eval,$gflops,$f1,$acc,$reducao,$iters,$status,$(date -Iseconds)" >> "$CSV"
    rm -f "$tmp"
    echo "  -> status=$status iters=$iters tempo_aco=$tempo_aco ms"
}

echo "=== 1) CUDA (block sizes) ==="
for bs in $BLOCK_LIST; do
    run_and_record "CUDA" "" "" "$bs" \
        "$CUD_BIN $DATASET $TARGET --ants $ANTS --iter $ITER --patience $PATIENCE --block-size $bs"
done

echo "=== 2) OpenMP (threads × escalonador) ==="
for sched in $SCHED_LIST; do
    for t in $THREAD_LIST; do
        run_and_record "OpenMP" "$t" "$sched" "" \
            "OMP_NUM_THREADS=$t OMP_SCHEDULE=$sched $OMP_BIN $DATASET $TARGET --ants $ANTS --iter $ITER --patience $PATIENCE"
    done
done

echo "=== Grade concluida. CSV: $CSV | RAW: $RAW ==="
```

- [ ] **Step 4: Rodar o teste — deve passar**

Run:
```bash
./scripts/tests/test_cdc_grid_smoke.sh
```
Expected: PASS (`OK: smoke test do grid passou`), CSV com 7 linhas, linhas CUDA e OpenMP presentes com números parseados.

- [ ] **Step 5: Commit**

```bash
git add scripts/run_cdc_grid.sh scripts/tests/test_cdc_grid_smoke.sh
git commit -m "feat: add run_cdc_grid.sh — CUDA-first then OpenMP threads×scheduler grid with incremental CSV"
```

---

### Task 4: Runner — resume (pular configs já concluídos)

**Files:**
- Modify: `scripts/run_cdc_grid.sh` (adicionar checagem de resume em `run_and_record`)
- Create: `scripts/tests/test_cdc_grid_resume.sh`

**Interfaces:**
- Consumes: `run_and_record` (Task 3) e o CSV existente.
- Produces: ao reexecutar, configs com `status=OK` no CSV são puladas (impressão `pulando (ja concluido)`); o CSV não ganha linhas duplicadas.

- [ ] **Step 1: Escrever o teste que falha**

Create `scripts/tests/test_cdc_grid_resume.sh`:

```bash
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
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run:
```bash
chmod +x scripts/tests/test_cdc_grid_resume.sh
./scripts/tests/test_cdc_grid_resume.sh
```
Expected: FALHA — sem resume, a 2ª execução duplica linhas (`n2 > n1`).

- [ ] **Step 3: Implementar o resume**

Em `scripts/run_cdc_grid.sh`, adicione esta função logo antes de `run_and_record`:
```bash
# Retorna 0 se a config (modo,threads,sched,bs) já está no CSV com status=OK.
already_done() {
    local modo="$1" threads="$2" sched="$3" bs="$4"
    [ -f "$CSV" ] || return 1
    awk -F, -v m="$modo" -v t="$threads" -v s="$sched" -v b="$bs" \
        '$1==m && $2==t && $3==s && $4==b && $12=="OK" {found=1} END{exit found?0:1}' "$CSV"
}
```

E no início de `run_and_record`, logo após a linha `local modo="$1" threads="$2" sched="$3" bs="$4" cmd="$5"`, adicione:
```bash
    if already_done "$modo" "$threads" "$sched" "$bs"; then
        echo "  -> pulando (ja concluido): modo=$modo threads=$threads sched=$sched bs=$bs"
        return 0
    fi
```

- [ ] **Step 4: Rodar o teste — deve passar**

Run:
```bash
./scripts/tests/test_cdc_grid_resume.sh
```
Expected: PASS (`OK: resume funcionou`), `n1 == n2`, saída contém `pulando`.

- [ ] **Step 5: Garantir que o smoke test (Task 3) ainda passa**

Run:
```bash
./scripts/tests/test_cdc_grid_smoke.sh
```
Expected: PASS (resume não quebra a primeira execução limpa, pois o smoke test apaga o CSV antes).

- [ ] **Step 6: Commit**

```bash
git add scripts/run_cdc_grid.sh scripts/tests/test_cdc_grid_resume.sh
git commit -m "feat: make run_cdc_grid.sh resumable (skip completed configs)"
```

---

### Task 5: Relatório — `build_cdc_report.py` (T1 do log + speedup + markdown)

**Files:**
- Create: `scripts/build_cdc_report.py`
- Create: `scripts/tests/test_build_cdc_report.py`

**Interfaces:**
- Consumes: `results/cdc_grid_results.csv` (Tasks 3-4) e um log com baseline 1-thread (`results_remote/hpc_raw_runs.log` no uso real).
- Produces: `extract_t1_per_iter(log_path) -> float` (ms/iteração do 1º bloco `Threads: 1`); `compute_speedup(t1_per_iter, tempo_aco_ms, iters) -> float`; e ao rodar como script, gera `results/cdc_grid_report.md`.

- [ ] **Step 1: Escrever o teste que falha**

Create `scripts/tests/test_build_cdc_report.py`:

```python
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from build_cdc_report import extract_t1_per_iter, compute_speedup


def test_extract_t1_per_iter():
    sample = (
        "=== Resultado ACO (OpenMP) ===\n"
        "Iteracoes executadas: 100/100\n"
        "Tempo ACO: 2.6451e+07 ms\n"
        "Threads: 1\n"
    )
    f = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    f.write(sample)
    f.close()
    t1 = extract_t1_per_iter(f.name)
    os.unlink(f.name)
    # 2.6451e7 / 100 = 264510 ms/iter
    assert abs(t1 - 264510.0) < 1.0, t1


def test_compute_speedup():
    # Tp/iter = metade do T1/iter -> speedup 2x
    # 1322550/10 = 132255 = 264510/2
    s = compute_speedup(264510.0, tempo_aco_ms=1322550.0, iters=10)
    assert abs(s - 2.0) < 1e-6, s


if __name__ == "__main__":
    test_extract_t1_per_iter()
    test_compute_speedup()
    print("OK: todos os testes passaram")
```

- [ ] **Step 2: Rodar e confirmar que falha**

Run:
```bash
python3 scripts/tests/test_build_cdc_report.py
```
Expected: FALHA — `ModuleNotFoundError: No module named 'build_cdc_report'` (script ainda não existe).

- [ ] **Step 3: Implementar `build_cdc_report.py`**

Create `scripts/build_cdc_report.py`:

```python
#!/usr/bin/env python3
"""Gera o relatório da grade CDC a partir do CSV do runner + baseline T1 do log.

Uso:
    python3 scripts/build_cdc_report.py \
        --csv results/cdc_grid_results.csv \
        --t1-log results_remote/hpc_raw_runs.log \
        --out results/cdc_grid_report.md
"""
import argparse
import csv
import re


def extract_t1_per_iter(log_path):
    """Tempo (ms) por iteração do baseline 1-thread.

    Procura o primeiro bloco '=== Resultado ACO' que contém 'Threads: 1' e
    divide o 'Tempo ACO' pelo nº de 'Iteracoes executadas'. O 1-thread é
    independente do escalonador (~1%), então serve de T1 para todos.
    """
    with open(log_path, encoding="utf-8", errors="ignore") as fh:
        text = fh.read()
    for block in re.split(r"=== Resultado ACO", text):
        if "Threads: 1" not in block:
            continue
        m_t = re.search(r"Tempo ACO:\s*([0-9.eE+]+)\s*ms", block)
        m_i = re.search(r"Iteracoes executadas:\s*(\d+)", block)
        if m_t and m_i and int(m_i.group(1)) > 0:
            return float(m_t.group(1)) / int(m_i.group(1))
    raise ValueError("Nenhum bloco 1-thread ('Threads: 1') encontrado em " + log_path)


def compute_speedup(t1_per_iter, tempo_aco_ms, iters):
    """Speedup normalizado por iteração: (T1/iter) / (Tp/iter)."""
    tempo_aco_ms = float(tempo_aco_ms)
    iters = int(iters)
    if iters <= 0 or tempo_aco_ms <= 0:
        return 0.0
    return t1_per_iter / (tempo_aco_ms / iters)


def build_report(csv_path, t1_log, out_path):
    t1 = extract_t1_per_iter(t1_log)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    cuda = [r for r in rows if r["modo"] == "CUDA"]
    omp = [r for r in rows if r["modo"] == "OpenMP"]

    lines = []
    lines.append("# Relatório — Grade CDC (OpenMP threads×escalonador + CUDA)\n")
    lines.append(f"**Baseline T1 (1 thread):** {t1:,.1f} ms/iteração "
                 f"(de `{t1_log}`)\n")
    lines.append("> Speedup = (T1/iteração) / (Tp/iteração) — normalizado por "
                 "iteração por causa do early stop (K varia). Ressalva: alguns "
                 "laços do ACO têm escalonador fixo; o laço quente (eval 1-NN) "
                 "respeita `OMP_SCHEDULE`.\n")

    lines.append("\n## CUDA (por block size)\n")
    lines.append("| Block size | Tempo ACO (ms) | Eval (ms) | GFLOPS | Speedup | Iter | Status |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for r in cuda:
        sp = compute_speedup(t1, r["tempo_aco_ms"], r["iteracoes"])
        lines.append(f"| {r['block_size']} | {r['tempo_aco_ms']} | {r['tempo_eval_ms']} "
                     f"| {r['gflops']} | {sp:.2f}x | {r['iteracoes']} | {r['status']} |")

    lines.append("\n## OpenMP (threads × escalonador)\n")
    lines.append("| Escalonador | Threads | Tempo ACO (ms) | Eval (ms) | GFLOPS | Speedup | Iter | Status |")
    lines.append("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |")
    for r in sorted(omp, key=lambda x: (x["escalonador"], int(x["threads"] or 0))):
        sp = compute_speedup(t1, r["tempo_aco_ms"], r["iteracoes"])
        lines.append(f"| {r['escalonador']} | {r['threads']} | {r['tempo_aco_ms']} "
                     f"| {r['tempo_eval_ms']} | {r['gflops']} | {sp:.2f}x "
                     f"| {r['iteracoes']} | {r['status']} |")

    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"Relatório escrito em {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results/cdc_grid_results.csv")
    ap.add_argument("--t1-log", default="results_remote/hpc_raw_runs.log")
    ap.add_argument("--out", default="results/cdc_grid_report.md")
    args = ap.parse_args()
    build_report(args.csv, args.t1_log, args.out)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Rodar o teste — deve passar**

Run:
```bash
python3 scripts/tests/test_build_cdc_report.py
```
Expected: PASS (`OK: todos os testes passaram`).

- [ ] **Step 5: Verificar geração ponta-a-ponta (com dados do smoke test)**

Run:
```bash
./scripts/tests/test_cdc_grid_smoke.sh >/dev/null 2>&1
python3 scripts/build_cdc_report.py \
    --csv results/cdc_grid_results.csv \
    --t1-log results_remote/hpc_raw_runs.log \
    --out results/cdc_grid_report.md
head -20 results/cdc_grid_report.md
```
Expected: arquivo `results/cdc_grid_report.md` com tabelas CUDA e OpenMP preenchidas e coluna Speedup numérica.

- [ ] **Step 6: Commit**

```bash
git add scripts/build_cdc_report.py scripts/tests/test_build_cdc_report.py
git commit -m "feat: add build_cdc_report.py — T1 from log + per-iteration speedup report"
```

---

## Execução real (no remoto, após implementar)

Não é tarefa de código — é o procedimento para você rodar o experimento de verdade:

1. Levar o código atualizado pro remoto (uma das opções):
   - `git push` da branch `feat/cdc-grid-benchmark` e `git pull` no remoto; ou
   - `scp` dos arquivos alterados (`src/openmp/main.cpp`, `src/openmp/aco_omp.cpp`,
     `src/cuda/main.cu`, `scripts/run_cdc_grid.sh`, `scripts/build_cdc_report.py`).
2. No remoto, em `~/aco-selection-parallel`:
   ```bash
   nohup bash scripts/run_cdc_grid.sh > results/cdc_grid_console.log 2>&1 &
   tail -f results/cdc_grid_console.log     # acompanhar
   # se cair: rode o mesmo comando de novo — ele pula o que já terminou
   ```
3. Trazer os resultados e gerar o relatório localmente:
   ```bash
   scp -r aluno@100.95.177.8:~/aco-selection-parallel/results/cdc_grid_* ./results/
   python3 scripts/build_cdc_report.py
   ```

## Self-Review (preenchido)

- **Cobertura do spec:** early stop §3.5 → Tasks 1-2; runner CUDA-first + OpenMP grid §5.1 →
  Task 3; resume §5.2 → Task 4; T1 do log + speedup §4 → Task 5; saídas §6 → Tasks 3+5;
  auto-encadeamento §5.2 → loop do Task 3. Co-execução híbrida e amostragem de eval ficam
  fora de escopo (§9), sem tarefa — correto.
- **Placeholders:** nenhum; todo passo de código mostra o código exato.
- **Consistência de tipos/nomes:** `--patience`, `config.patience`/`patience`, colunas do CSV,
  e as funções `extract_t1_per_iter`/`compute_speedup` batem entre Task 5 (def) e seu teste.
