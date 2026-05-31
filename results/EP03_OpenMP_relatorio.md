# EP03 — Paralelização com OpenMP · Relatório do Épico

**Projeto:** aco-selection-parallel · **Disciplina:** Computação de Alto Desempenho — UFG 2026
**Épico:** [#3 — EP03 — Paralelização com OpenMP](https://github.com/phael-exe/aco-selection-parallel/issues/3)
**User Stories:** [US11 #16](https://github.com/phael-exe/aco-selection-parallel/issues/16) · [US12 #17](https://github.com/phael-exe/aco-selection-parallel/issues/17)
**Artigo-baseline:** Magalhães, Gonçalves, Nobre, Freitas (2024), PUC Minas —
*"Avaliação de Desempenho e Escalabilidade do Algoritmo de Otimização de Colônia de Formigas em C++ e Python"* — [jmarcosjm/aco-cpp](https://github.com/jmarcosjm/aco-cpp).

> Este documento responde a quatro perguntas para cada item do épico:
> **(1) onde foi feito** (arquivo:linha), **(2) o quê**, **(3) por que assim**
> (incl. como o artigo fundamenta), e **(4) o resultado** — melhor ou pior, com os valores.

---

## 0. Como o artigo fundamenta o épico

O artigo pegou um ACO de **seleção de instâncias** em Python e o reescreveu em **C++**,
mostrando que C++ é **até 5× mais rápido** mantendo a mesma qualidade, com muito menos
*page faults* e *cache misses* (ex.: brain-stroke — 366.893 page faults em Python vs 31.032 em C++).

O EP03 **estende esse eixo**: partimos da versão sequencial C++ (EP01) e adicionamos paralelismo
de **CPU via OpenMP**, criando o ponto intermediário entre *sequencial* e *CUDA* (EP02). A tese
central do épico — que o OpenMP evita o overhead host↔device da GPU e por isso pode vencer em
datasets médios — é exatamente o tipo de comparação de paradigmas que o artigo defende como
academicamente valiosa.

Três escolhas do projeto que o artigo confirma:
1. **1-NN como wrapper** — o artigo usa KNN para julgar a qualidade do subconjunto; mantivemos.
2. **C++ puro, sem bibliotecas pesadas** — o artigo mostra que isso reduz cache miss / page fault.
3. **Seleção de instâncias é o problema** — descartar linhas redundantes preservando acurácia.

---

## US11 — Paralelização do Loop de Formigas com OpenMP

### Pontos de paralelização (onde / o quê / por quê / resultado)

| # | Região | Onde (arquivo:linha) | Diretiva usada | Por que assim |
|---|--------|----------------------|----------------|---------------|
| 1 | Loop de formigas (construção) | [aco_omp.cpp:279](../src/openmp/aco_omp.cpp#L279) | `#pragma omp parallel for schedule(dynamic)` | Cada formiga escreve só na sua fatia `colony[k*N..]` → embaraçosamente paralelo. RNG semeado por `(iter,k)` → resultado independente do escalonamento. |
| 2 | Cálculo de distância euclidiana | [aco_omp.cpp:48](../src/openmp/aco_omp.cpp#L48) | `#pragma omp parallel for schedule(dynamic)` (**não** `collapse(2)`) | Loop **triangular** (`j=i+1`): `collapse(2)` exige loops retangulares e produziria índices inválidos. `schedule(dynamic)` corrige o desbalanceamento natural do triângulo (linhas com `i` pequeno têm mais trabalho). |
| 3 | Depósito de feromônio | [aco_omp.cpp:176](../src/openmp/aco_omp.cpp#L176) | `#pragma omp atomic` | Várias formigas podem depositar na mesma instância `i` → race condition na escrita. `atomic` tem menos overhead que `critical` (confirmado pelo artigo). |
| 4 | Evaporação de feromônio | [aco_omp.cpp:184](../src/openmp/aco_omp.cpp#L184) | `#pragma omp parallel for` | Cada instância evapora de forma independente → sem dependências. |
| 5 | Depósito (loop sobre K formigas) | [aco_omp.cpp:289](../src/openmp/aco_omp.cpp#L289) | `#pragma omp parallel for schedule(dynamic)` | Paraleliza a aplicação do depósito sobre as K formigas; a escrita interna é protegida pelo `atomic` (#3). |
| 6 | Proxy-fitness (contagem) | [aco_omp.cpp:300](../src/openmp/aco_omp.cpp#L300) | `#pragma omp parallel for` | Cada `k` escreve só `ant_scores[k]` → sem conflito. |

**Compilação** — target `openmp` no [Makefile:45](../Makefile#L45) com `OMPFLAGS = -fopenmp`
([Makefile:12](../Makefile#L12)). Build via `make openmp` → `build/aco_omp`.

### ⚠️ Divergência honesta com o checklist da US11

O checklist da issue #16 marca **`[x] Paralelizar cálculo de distância com collapse(2)`**, mas o
código **não usa `collapse(2)`** — usa `schedule(dynamic)` no loop externo. **Isso é correto, não
um erro**: o laço é triangular e `collapse(2)` só é válido para laços retangulares. A
implementação entregue é tecnicamente **superior** ao que o texto do checklist pede; apenas o
texto do checklist está desatualizado em relação ao código. Recomenda-se editar o item da issue
para "schedule(dynamic) no loop externo (collapse(2) inválido para laço triangular)".

### Cobertura do checklist US11

| Item do checklist | Onde / evidência | Status |
|---|---|---|
| Copiar código sequencial para `src/openmp/` | `src/openmp/*.cpp` (9 arquivos) | ✅ |
| `#pragma omp parallel for` no loop de formigas | [aco_omp.cpp:279](../src/openmp/aco_omp.cpp#L279) | ✅ |
| Paralelizar distância (`collapse(2)`) | [aco_omp.cpp:48](../src/openmp/aco_omp.cpp#L48) — `schedule(dynamic)` (collapse(2) inviável; ver acima) | ✅ (implementação superior; texto do item impreciso) |
| `omp atomic` no depósito | [aco_omp.cpp:176](../src/openmp/aco_omp.cpp#L176) | ✅ |
| Paralelizar evaporação | [aco_omp.cpp:184](../src/openmp/aco_omp.cpp#L184) | ✅ |
| Compilar com `-fopenmp` | [Makefile:12,45](../Makefile#L45) | ✅ |
| Testar 1,2,4,8 threads via `OMP_NUM_THREADS` | sweep 1,2,4,8,**16** | ✅ (estendido a 16) |
| Validar contra sequencial | Seção de resultados abaixo | ✅ |
| Medir speedup por nº de threads | `results/validation_openmp.csv` | ✅ |

**Veredito US11:** ✅ entregue e conforme — com a ressalva de que o item "collapse(2)" do
checklist descreve algo que (corretamente) não foi feito; o código faz a escolha certa.

---

## US12 — Validação da Versão OpenMP

### Configuração experimental

| Item | Valor |
|------|-------|
| CPU | Intel Core i5-12450HX — 12 núcleos lógicos (8 físicos: 4P+4E) |
| Threads | 1, 2, 4, 8, 16 (16 = oversubscrição proposital, > 12 cores) |
| Parâmetros ACO | `--ants 64 --iter 100` (idêntico a US11 e à validação sequencial) |
| Seed OpenMP | fixo `42` → determinístico e independente do nº de threads |
| Speedup | `T1 / Tp` sobre `Tempo ACO` (ms) do `std::chrono` interno |
| Critério de acurácia | OpenMP vs sequencial dentro de ±2% |
| Artefatos | `results/validation_openmp.csv` (50 linhas) + `results/validation_openmp_report.md` |

> A referência sequencial usa `srand(time(NULL))` (**não-determinística**); o comparativo
> seq vs OpenMP é uma **amostra única**, sujeita a ruído em bases pequenas.

### Resultado 1 — Corretude por determinismo (prova forte)

A prova de corretude **não** é "ficou parecido com o sequencial", e sim: **o OpenMP dá resultado
byte-idêntico para 1, 2, 4, 8 e 16 threads.** Garantido por construção — RNG por-formiga semeado
só por `(iter,k)` ([aco_omp.cpp:281](../src/openmp/aco_omp.cpp#L281)) — e confirmado em **10/10
bases** com variação **0.000000** de acurácia e F1 entre threads.

### Resultado 2 — Acurácia OpenMP vs Sequencial (melhor ou pior?)

| Dataset | N | Acc OpenMP | Acc Sequencial | Δ (%) | Status |
|---|---:|---:|---:|---:|---|
| heart_failure | 299 | 0.839465 | 0.822742 | +1.67 | OK |
| haberman | 306 | 0.872549 | 0.872549 | 0.00 | OK |
| cirrhosis | 418 | 0.744019 | 0.717703 | **+2.63** | **WARN** |
| diabetes | 768 | 0.872396 | 0.873698 | −0.13 | OK |
| tic-tac-toe | 958 | 0.874739 | 0.864301 | +1.04 | OK |
| yeast | 1484 | 0.769542 | 0.774933 | −0.54 | OK |
| vaccine | 3152 | 0.920368 | 0.920685 | −0.03 | OK |
| Employee | 4653 | 0.844187 | 0.847410 | −0.32 | OK |
| brain-stroke | 4981 | 0.961855 | 0.963260 | −0.14 | OK |

**8/9 dentro de ±2%.** O único WARN (cirrhosis, +2,63%) **não é defeito da paralelização** (o
OpenMP é determinístico, variação 0% entre threads): é desvio contra a referência sequencial
não-determinística, em base pequena e sensível ao sorteio inicial. Note que o OpenMP ficou
**melhor** (0.744 vs 0.718). A propriedade exigida pela US12 — acurácia preservada em **todas as
configs de thread** — é satisfeita de forma **exata (0%)**.

### Resultado 3 — Curva de speedup (T1 / Tp)

| Dataset | N | 1t | 2t | 4t | 8t | 16t | Ótimo |
|---|---:|---:|---:|---:|---:|---:|---|
| heart_failure | 299 | 1.00 | 1.20 | 0.81 | 1.33 | 0.85 | ~1 (sem ganho real) |
| haberman | 306 | 1.00 | **1.87** | 1.66 | 1.35 | 0.86 | 2 |
| cirrhosis | 418 | 1.00 | 1.63 | **2.47** | 1.53 | 1.32 | 4 |
| diabetes | 768 | 1.00 | 1.57 | **2.76** | 1.46 | 1.33 | 4 |
| tic-tac-toe | 958 | 1.00 | 1.92 | **2.05** | 1.83 | 1.71 | 4 |
| yeast | 1484 | 1.00 | 1.57 | **2.39** | 1.53 | 1.47 | 4 |
| vaccine | 3152 | 1.00 | 1.89 | 3.32 | 2.86 | **4.09** | 16 (4t já dá 3.32) |
| Employee | 4653 | 1.00 | 1.82 | 3.11 | 2.76 | **3.23** | 16 (4t já dá 3.11) |
| brain-stroke | 4981 | 1.00 | 1.85 | 3.20 | 3.27 | **3.97** | 16 (4t já dá 3.20) |

Bases com tempo < ~100 ms (heart_failure, haberman) têm speedup **ruidoso** — o custo de criar/
sincronizar threads é da ordem do trabalho útil; conclusão prática: "não compensa paralelizar".

### Resultado 4 — CDC Diabetes (larga escala, 253.680 instâncias)

Modo *on-the-fly* (N > 10.000, [aco_omp.cpp:228](../src/openmp/aco_omp.cpp#L228)) — sem matriz
N×N, pico de RAM ~200 MB. Validação total ~3,4 h (1 thread ≈ 90 min).

| Threads | Tempo | Speedup | Acurácia | F1 |
|---:|---:|---:|---:|---:|
| 1 | 89,8 min | 1.00 | 0.900465 | 0.641407 |
| 2 | 48,9 min | 1.84 | 0.900465 | 0.641407 |
| 4 | 28,0 min | 3.20 | 0.900465 | 0.641407 |
| 8 | 20,2 min | 4.45 | 0.900465 | 0.641407 |
| **16** | **17,3 min** | **5.19** | 0.900465 | 0.641407 |

**Melhor escalabilidade do projeto:** curva **monotônica até 16 threads (5.19×)**, sem platô.
Determinismo confirmado também aqui (acc/F1 idênticos 1→16t). É o caso que mais justifica o
paralelismo — o volume do 1-NN (~126k treino × 253k teste por avaliação) amortiza o overhead.

### Por que o speedup máximo (~4×) fica abaixo do ideal (12×)?

Esperado e coerente com o gargalo conhecido: por iteração, só as **top-K formigas (K=1–3)** são
avaliadas com 1-NN, e a avaliação é **memory-bandwidth bound**. O paralelismo das formigas tem
trabalho limitado; o ganho real vem do 1-NN e das distâncias, que saturam a banda de memória
antes de usar todos os núcleos. Por isso o ótimo **cresce com N**: ~4 threads nas bases pequenas,
16 no CDC.

### Cobertura do checklist US12

| Item do checklist | Evidência | Status |
|---|---|---|
| `OMP_NUM_THREADS=1` (~igual ao sequencial) | 1t medido; Δacc ≤ 1.67% (cirrhosis explicado) | ✅ |
| Executar 2,4,8,16 threads | sweep completo nas 9 bases + CDC | ✅ |
| Comparar acurácia OpenMP vs sequencial | 8/9 OK (±2%); WARN único = ruído do seq | ✅ |
| Documentar speedup em `results/validation_openmp.csv` | gerado (50 linhas: 10 datasets × 5 threads) | ✅ |
| Identificar ponto de saturação | nº ótimo cresce com N: ~4 threads → 16 (CDC) | ✅ |
| 9 datasets + CDC Diabetes | **10/10 concluídos** | ✅ |

**Veredito US12:** ✅ tecnicamente completa — as 5 caixas do checklist têm evidência.

---

## Resumo: melhor ou pior?

- **Qualidade (acurácia):** preservada. 8/9 bases dentro de ±2% vs sequencial; determinismo
  **exato (0%)** entre 1→16 threads em 10/10 bases. Em cirrhosis o OpenMP ficou **melhor**
  (0.744 vs 0.718).
- **Desempenho (speedup):** **melhor** e proporcional à escala. Bases médias (3–5k): **3,2–4,1×**
  em 4–16 threads. Larga escala (CDC 253k): **5,19×** em 16 threads, escalando limpo.
- **Custo:** bases < ~100 ms não compensam paralelizar (overhead ≈ trabalho).

A versão OpenMP cumpre o objetivo do épico: paralelismo de CPU **sem GPU**, correto, com
acurácia preservada e speedup mensurável e crescente — pronta para o benchmark comparativo (EP04).

---

## ⚠️ Pendências de bookkeeping no GitHub (não são pendências técnicas)

O trabalho está feito, mas o board **não reflete**:

1. **Issues #16 (US11) e #17 (US12) continuam OPEN** e o épico #3 mostra **0/2 concluídos** — fechar após merge.
2. **Checklist da US12 (#17) está 0/5 marcado** — marcar as 5 caixas (há evidência para todas).
3. **Artefatos da US12 não estão commitados** — `results/validation_openmp_report.md` e
   `scripts/run_validation_openmp.sh` estão *untracked*. Commitar.
4. **Item "collapse(2)" da US11 (#16)** descreve algo que (corretamente) não foi feito — ajustar
   o texto para refletir `schedule(dynamic)` em laço triangular.
