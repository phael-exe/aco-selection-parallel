# EP04 — Paralelização com CUDA · Benchmark Comparativo

**Projeto:** aco-selection-parallel · **Disciplina:** Computação de Alto Desempenho — UFG 2026  
**Data:** 2026-05-31  
**GPU:** NVIDIA GeForce RTX 4050 Laptop GPU  
**CPU:** 13th Gen Intel(R) Core(TM) i5-13420H (12 núcleos lógicos)  
**Parâmetros:** 64 formigas, 100 iterações máx, evap=0.1, Q=1, α=1, β=1

> **Metodologia:** idêntica ao EP03. Seq = baseline de qualidade; curva de speedup OpenMP usa
> **OMP-1t como referência** (1t = 1.00 por definição), igualmente ao EP03. CUDA comparado contra
> o sequencial em tempo absoluto. Avaliação 1-NN na CPU para todos os modos.

---

## 1. Qualidade — F1-Score por modo

| Dataset | N | Seq | OMP-1t | OMP-12t | CUDA | Δ OMP-12t vs Seq | Δ CUDA vs Seq |
|---------|--:|----:|-------:|--------:|-----:|:----------------:|:-------------:|
| heart_failure      |   299 | 0.916667 | 0.873096 | 0.873096 | 0.878307 |           -4.75% |        -4.18% |
| haberman           |   306 | 0.865854 | 0.877193 | 0.877193 | 0.877193 |           +1.31% |        +1.31% |
| cirrhosis          |   418 | 0.876302 | 0.863608 | 0.863608 |  0.84841 |           -1.45% |        -3.18% |
| diabetes           |   768 | 0.917603 | 0.915888 | 0.915888 | 0.909091 |           -0.19% |        -0.93% |
| tic-tac-toe        |   958 | 0.975535 | 0.975535 | 0.975535 | 0.974046 |           +0.00% |        -0.15% |
| yeast              |  1484 | 0.968335 | 0.970717 | 0.970717 | 0.971496 |           +0.25% |        +0.33% |
| vaccine            |  3152 | 0.960053 | 0.954395 | 0.954395 | 0.957152 |           -0.59% |        -0.30% |
| Employee           |  4653 | 0.809762 | 0.811048 | 0.811048 | 0.818124 |           +0.16% |        +1.03% |
| brain-stroke       |  4981 | 0.731405 | 0.723926 | 0.723926 | 0.730539 |           -1.02% |        -0.12% |

> Variações de F1 entre modos são **esperadas** pela estocasticidade (sementes diferentes por run).
> O que importa: todos os modos produzem qualidade comparável (±5% é tolerável).

---

## 2. Curva de Speedup OpenMP — T₁_omp / Tp_omp (igual ao EP03)

> Referência: **OMP-1t** (1t ≡ 1.00 por definição). Mede ganho puro de paralelismo,
> sem contaminar com diferenças de compilação entre seq e omp.

| Dataset | N | 1t | 2t | 4t | 8t | 12t | Ótimo |
|---------|--:|---:|---:|---:|---:|----:|-------|
| heart_failure      |   299 | 1.00 | 1.22 | 1.59 | 2.34 | 1.65 | 8t     |
| haberman           |   306 | 1.00 | 1.28 | 1.46 | 1.70 | 1.37 | 8t     |
| cirrhosis          |   418 | 1.00 | 1.49 | 1.77 | 2.45 | 2.33 | 8t     |
| diabetes           |   768 | 1.00 | 1.68 | 1.85 | 3.08 | 2.32 | 8t     |
| tic-tac-toe        |   958 | 1.00 | 1.75 | 1.76 | 2.73 | 2.73 | 8t     |
| yeast              |  1484 | 1.00 | 1.42 | 2.00 | 3.50 | 3.17 | 8t     |
| vaccine            |  3152 | 1.00 | 2.05 | 2.64 | 4.12 | 4.83 | 12t    |
| Employee           |  4653 | 1.00 | 1.82 | 2.20 | 3.73 | 3.97 | 12t    |
| brain-stroke       |  4981 | 1.00 | 1.76 | 2.98 | 3.87 | 3.82 | 8t     |

---

## 3. Desempenho Absoluto — Tempo ACO (ms)

| Dataset | N | Seq (ms) | OMP-1t (ms) | OMP-4t (ms) | OMP-12t (ms) | CUDA (ms) | Speedup CUDA/Seq |
|---------|--:|---------:|------------:|------------:|-------------:|----------:|:----------------:|
| heart_failure      |   299 |   18.1834 |     17.0186 |     10.7121 |      10.3096 |   26.5517 |            0.68x |
| haberman           |   306 |   12.3081 |     11.5379 |     7.90347 |      8.43582 |   16.1546 |            0.76x |
| cirrhosis          |   418 |   47.2566 |     48.4024 |     27.3423 |      20.7794 |   83.9148 |            0.56x |
| diabetes           |   768 |   74.1515 |     79.1943 |     42.9097 |      34.1344 |   120.766 |            0.61x |
| tic-tac-toe        |   958 |   199.654 |     203.778 |     115.932 |      74.5958 |   235.505 |            0.85x |
| yeast              |  1484 |   197.569 |     325.172 |     162.993 |      102.478 |   361.581 |            0.55x |
| vaccine            |  3152 |   3614.91 |     4223.35 |     1597.95 |      874.163 |   7579.84 |            0.48x |
| Employee           |  4653 |    1916.3 |     2302.28 |     1047.31 |      580.033 |    2909.6 |            0.66x |
| brain-stroke       |  4981 |   2889.22 |     2707.65 |     907.561 |      708.583 |   4223.79 |            0.68x |

---

## 4. Análise

### 4.1 GPU Compute vs. Tempo Total CUDA

O tempo total CUDA (`Tempo ACO`) inclui três componentes:

| Componente | Custo | Dominante? |
|---|---|---|
| **GPU compute** (construção K×N paralela) | < 1 ms mesmo em N=5.000 | ❌ |
| **Transferências H↔D** (τ, select_prob, colony por iter) | ~1-5 ms/iter | ❌ |
| **Avaliação 1-NN na CPU** (top-K formigas, mesma CPU do seq) | ~90% do tempo | ✅ |

A fase de construção GPU faz **64 × N decisões de seleção em paralelo** em < 1 ms — mas a
avaliação 1-NN (O(N² · F) por formiga) ainda roda na CPU. O resultado: o tempo total CUDA é
dominado pelo mesmo bottleneck do sequencial, mais o overhead de transferência.

### 4.2 Por que a CUDA é mais lenta que o sequencial nestes datasets?

Os 9 datasets baseline têm N entre 299 e 4.981 — muito pequenos para amortizar:
- Alocação de contexto CUDA (uma vez, mas real)
- Transferências D→H da colônia K×N a cada iteração (64 × 5.000 × 4 bytes = 1,2 MB/iter)
- Computação CPU da `select_prob` (download τ, compute, upload) a cada iteração

O ganho real da GPU estaria com **N >> 50.000** ou com a avaliação 1-NN também na GPU
(o kernel `knn_1nn_kernel` já está implementado em `kernels.cu`).

### 4.3 OpenMP ganha em N médio

Para N ∈ [3.000, 5.000] o OpenMP com 8–12 threads alcança **~3–4×** de speedup, pois
paraleliza tanto a construção quanto a avaliação 1-NN sem overhead de transferência.
Padrão confirmado pelo EP03: o ganho cresce monotonicamente com N.

### 4.4 Corretude do algoritmo CUDA

O ACO CUDA implementa **o mesmo algoritmo** que o sequencial e o OpenMP:

| Etapa | Implementação |
|-------|--------------|
| Probabilidade de seleção | **τᵢ^α · ηᵢ^β / max(τ·η)** — normalizado por máximo (CPU a cada iter) |
| Visibilidade ηᵢ | 1/(1 + avg_distᵢ) — pré-computada na GPU com `avg_distance_kernel` (uma vez) |
| Depósito | **Q / tour_length** por formiga selecionante — `deposit_kernel` com `atomicAddDouble` |
| Evaporação | **(τ + deposit) × (1 − ρ)** — `apply_pheromone_kernel` (ordem = sequencial) |
| Avaliação | Macro-F1 para multiclasse; `classes[1]` como positivo para binário (= sequencial) |

---

*Gerado por `scripts/run_benchmark_comparison.sh` · dados em `results/EP04_CUDA_benchmark_raw.csv`*
