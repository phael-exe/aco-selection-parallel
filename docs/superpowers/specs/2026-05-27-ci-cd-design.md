# CI/CD Design — aco-selection-parallel

**Data:** 2026-05-27  
**Projeto:** Paralelização ACO com CUDA e OpenMP — CAD UFG 2026  
**Abordagem escolhida:** Balanced (Build + Equivalência)

---

## Objetivo

Garantir que PRs para `main` não quebrem a compilação nem introduzam bugs de paralelismo, sem criar overhead que atrapalhe o trabalho acadêmico. CI deve rodar em ≤4 min.

---

## Gatilhos

```yaml
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
```

CI roda somente em PRs para `main` e pushs diretos em `main`. Branches de feature não disparam runs.

---

## Jobs (todos paralelos)

| Job | Runner | Depends on | Tempo estimado |
|-----|--------|------------|---------------|
| `build-sequential` | `ubuntu-latest` | — | ~2 min |
| `build-openmp` | `ubuntu-latest` | — | ~2 min |
| `build-cuda` | `ubuntu-latest` | — | ~2 min |
| `lint` | `ubuntu-latest` | — | ~30 s |
| `equivalence` | `ubuntu-latest` | `build-sequential`, `build-openmp` | ~1 min |

### `build-sequential`
1. Instala dependências (`g++`, `make`)
2. `make sequential`
3. Roda `./sequential --seed 42 data/baseline/haberman.csv` → `results/ci/seq_haberman.csv`
4. Roda `./sequential --seed 42 data/baseline/heart_failure.csv` → `results/ci/seq_heart.csv`
5. Faz upload dos CSVs como artifact para o job de equivalência

### `build-openmp`
1. Instala dependências (`g++`, `make` — OpenMP via libgomp já incluso)
2. `make openmp`
3. Roda `./openmp --seed 42 data/baseline/haberman.csv` → `results/ci/omp_haberman.csv`
4. Roda `./openmp --seed 42 data/baseline/heart_failure.csv` → `results/ci/omp_heart.csv`
5. Faz upload dos CSVs como artifact

### `build-cuda`
1. Instala `cuda-toolkit` via apt
2. `make cuda CI=1` — compila com `nvcc` sem linkar device code para execução
3. O Makefile deve respeitar a flag `CI=1` para pular o link final (ou usar `nvcc --compile`)
4. **Não executa nenhum kernel** — compile-only

### `lint`
1. `pip install flake8`
2. `flake8 scripts/ --max-line-length=120 --select=E9,F`
3. Só falha em erros reais (sintaxe, imports indefinidos), não em estilo

---

## Teste de Equivalência

Após `build-sequential` e `build-openmp` completarem, um job `equivalence` roda:

1. Baixa os artifacts de ambos os jobs
2. Executa `python3 scripts/compare_outputs.py results/ci/seq_haberman.csv results/ci/omp_haberman.csv`
3. Executa `python3 scripts/compare_outputs.py results/ci/seq_heart.csv results/ci/omp_heart.csv`
4. `compare_outputs.py` retorna exit 0 se equivalentes (tolerância `atol=1e-6`), exit 1 caso contrário

**Critério de equivalência:** mesmas instâncias selecionadas (índices das linhas no CSV de saída). O ACO deve ser determinístico via `--seed <int>` — essa é uma dependência de implementação necessária de qualquer forma para reprodutibilidade experimental.

---

## Dependência de Implementação

O CI exige que os binários aceitem `--seed <int>` como argumento CLI. Isso já é necessário para os experimentos do EP04 (reprodutibilidade). A issue T03 — "Parametrizar coluna-alvo via CLI" é o precedente para parametrização via CLI.

---

## Arquivos Criados

| Arquivo | Propósito |
|---------|-----------|
| `.github/workflows/ci.yml` | Workflow principal com 5 jobs |
| `scripts/compare_outputs.py` | Comparação tolerante de CSVs de saída |
| `README.md` (badge adicionado) | Badge de status do CI |

---

## Proteção de Branch (configuração manual)

Após o workflow estar verde, habilitar em GitHub → Settings → Branches → `main`:
- **Require status checks to pass before merging**
- Checks obrigatórios: `build-sequential`, `build-openmp`, `build-cuda`, `lint`, `equivalence`

---

## O que este CI não cobre (intencional)

- Execução de kernels CUDA (sem GPU no runner)
- Benchmarks de performance (muito lentos para CI)
- Datasets grandes (CDC Diabetes — 253k instâncias)
- Análise estática com cppcheck (fora do escopo, adicionável depois)
- Cobertura de código (overkill para contexto acadêmico)
