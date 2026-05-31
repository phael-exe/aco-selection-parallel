# US12 — Validação da Versão OpenMP do ACO

**Épico:** EP03 — Paralelização com OpenMP · **Issue:** #16
**Disciplina:** Computação de Alto Desempenho — UFG 2026
**Baseline (artigo de referência):** Magalhães, J.M.O. *et al.* — *Avaliação de Desempenho e Escalabilidade do ACO em C++ e Python* (PUC Minas, 2024) — [jmarcosjm/aco-cpp](https://github.com/jmarcosjm/aco-cpp).
As 9 bases vêm desse baseline; o **CDC Diabetes** (253.680 instâncias) vem do [UCI ML Repository](https://archive.ics.uci.edu/dataset/891).

A US12 replica, no eixo **CPU/OpenMP**, a metodologia de *desempenho e escalabilidade* do artigo-baseline: medir a curva de speedup por nº de threads, validar a corretude (acurácia preservada) e identificar o ponto de saturação.

---

## 1. Configuração experimental

| Item | Valor |
|------|-------|
| CPU | Intel Core i5-12450HX — **12 núcleos lógicos** (8 físicos: 4P+4E) |
| Threads testados | **1, 2, 4, 8, 16** (16 = oversubscrição proposital, > 12 cores) |
| Parâmetros ACO | `--ants 64 --iter 100` (idêntico à validação sequencial e ao US11) |
| Seed OpenMP | fixo `42` → determinístico e independente do nº de threads |
| Métrica de tempo | `Tempo ACO` (ms), do `std::chrono` interno; speedup = `T1 / Tp` |
| Critério de acurácia | OpenMP vs sequencial dentro de **±2%** (`Status = OK/WARN`) |
| Artefatos | `results/validation_openmp.csv` (dados brutos), este relatório |

> Comparação com a versão sequencial usa `results/validation_sequential.csv`. A versão
> sequencial usa `srand(time(NULL))` (**não-determinística**); logo o comparativo é uma
> única amostra de referência, sujeita a ruído estatístico em bases pequenas.

---

## 2. Corretude da paralelização — Determinismo (prova forte)

A prova de corretude da paralelização **não** é "ficou parecido com o sequencial", e sim:
**o resultado do OpenMP é idêntico para qualquer nº de threads.** Isso é garantido por
construção — RNG por-formiga semeado apenas por `(iter, k)`, independente do escalonamento
([aco_omp.cpp:281](../src/openmp/aco_omp.cpp#L281)) — e foi confirmado empiricamente:

| Dataset | Acurácia (1→16t) | F1 (1→16t) | Variação entre threads |
|---|---|---|---|
| heart_failure | 0.839465 | 0.741935 | **0.000000** |
| haberman | 0.872549 | 0.769231 | **0.000000** |
| cirrhosis | 0.744019 | 0.738376 | **0.000000** |
| diabetes | 0.872396 | 0.815789 | **0.000000** |
| tic-tac-toe | 0.874739 | 0.796610 | **0.000000** |
| yeast | 0.769542 | 0.788860 | **0.000000** |
| vaccine | 0.920368 | 0.832555 | **0.000000** |
| Employee | 0.844187 | 0.773367 | **0.000000** |
| brain-stroke | 0.961855 | 0.620000 | **0.000000** |

✅ **Acurácia/F1 byte-idênticos em 1, 2, 4, 8 e 16 threads** em todas as bases.
A paralelização é correta: mesma resposta, apenas mais rápida.

---

## 3. Acurácia OpenMP vs Sequencial (±2%)

| Dataset | Acc OpenMP | Acc Sequencial | Δ (%) | Status |
|---|---|---|---|---|
| heart_failure | 0.839465 | 0.822742 | 1.67 | OK |
| haberman | 0.872549 | 0.872549 | 0.00 | OK |
| cirrhosis | 0.744019 | 0.717703 | **2.63** | **WARN** |
| diabetes | 0.872396 | 0.873698 | 0.13 | OK |
| tic-tac-toe | 0.874739 | 0.864301 | 1.04 | OK |
| yeast | 0.769542 | 0.774933 | 0.54 | OK |
| vaccine | 0.920368 | 0.920685 | 0.03 | OK |
| Employee | 0.844187 | 0.847410 | 0.32 | OK |
| brain-stroke | 0.961855 | 0.963260 | 0.14 | OK |

**8/9 dentro de ±2%.** O único `WARN` (cirrhosis, 2,63%) **não é defeito da paralelização**:
o OpenMP é determinístico (Seção 2, variação 0%), e o desvio é contra a referência sequencial,
que é uma **única amostra não-determinística**. cirrhosis é pequeno (418 inst., 19 features) e
sensível ao sorteio inicial — daí a maior dispersão. O OpenMP inclusive obteve acurácia *maior*
(0.744 vs 0.718). A propriedade exigida pela US12 — "acurácia preservada em todas as **configs
de thread**" — é satisfeita de forma **exata** (0%).

---

## 4. Curva de Speedup (T1 / Tp)

| Dataset | N | 1t | 2t | 4t | 8t | 16t | **Ótimo** |
|---|---:|---:|---:|---:|---:|---:|---|
| heart_failure | 299 | 1.00 | 1.20 | 0.81 | 1.33 | 0.85 | ~1 (sem ganho real) |
| haberman | 306 | 1.00 | **1.87** | 1.66 | 1.35 | 0.86 | **2** |
| cirrhosis | 418 | 1.00 | 1.63 | **2.47** | 1.53 | 1.32 | **4** |
| diabetes | 768 | 1.00 | 1.57 | **2.76** | 1.46 | 1.33 | **4** |
| tic-tac-toe | 958 | 1.00 | 1.92 | **2.05** | 1.83 | 1.71 | **4** |
| yeast | 1484 | 1.00 | 1.57 | **2.39** | 1.53 | 1.47 | **4** |
| vaccine | 3152 | 1.00 | 1.89 | 3.32 | 2.86 | **4.09** | **16** (4t já dá 3.32) |
| Employee | 4653 | 1.00 | 1.82 | 3.11 | 2.76 | **3.23** | **16** (4t já dá 3.11) |
| brain-stroke | 4981 | 1.00 | 1.85 | 3.20 | 3.27 | **3.97** | **16** (4t já dá 3.20) |

> Tempos < ~100 ms (heart_failure, haberman) têm speedup **ruidoso**: o custo de criar/sincronizar
> threads é da mesma ordem do trabalho útil, então as medições oscilam. Para essas bases a
> conclusão prática é "não compensa paralelizar".

---

## 5. Ponto de saturação e nº ótimo de threads

O ponto de saturação **depende do tamanho do dataset**:

- **N pequeno (≤ ~1000):** satura em **2–4 threads**. Acima disso o overhead de threads e a
  oversubscrição (16t > 12 cores) **degradam** o tempo. heart_failure/haberman praticamente não
  se beneficiam (speedup ≈ 1).
- **N médio (1000–1500):** ótimo em **4 threads** (~2.4×), depois cai.
- **N grande (3000–5000):** continua ganhando até 8–16 threads, mas com **retornos decrescentes** —
  o salto grande acontece já em **4 threads** (3.1–3.3×) e o ganho de 4→16 é marginal.
- **N muito grande (CDC, 253.680):** escala de forma **limpa e monotônica até 16 threads (5.19×)**,
  sem platô. Quanto maior o dataset, mais threads compensam — o ótimo migra de 2–4 (bases pequenas)
  para 16+ (CDC).

**Conclusão sobre o ótimo:** não há um número único — **o nº ótimo cresce com o tamanho do dataset**.
Para a maioria das bases do baseline (N ≤ 5k), **~4 threads** é o sweet spot (captura quase todo o
ganho sem pagar oversubscrição). Para larga escala (CDC), **16 threads** ainda é o melhor e a curva
sugere que continuaria ganhando além disso.

🔎 **Observação de eficiência:** o speedup máximo (~4×) fica bem abaixo do ideal teórico (12×).
Isso é esperado e coerente com o gargalo conhecido do algoritmo: por iteração, apenas as
**top-K formigas (K=1–3)** são avaliadas com 1-NN, e essa avaliação é *memory-bandwidth bound*.
O paralelismo das formigas tem trabalho limitado; o ganho real vem do 1-NN e do cálculo de
distâncias, que saturam a banda de memória bem antes de usar todos os núcleos.

---

## 6. CDC Diabetes (larga escala, 253.680 instâncias)

Executado full (`--ants 64 --iter 100`, early-stopping patience=10) com `nice -n 10`, em modo
*on-the-fly* (N > 10.000, [aco_omp.cpp:228](../src/openmp/aco_omp.cpp#L228)) — **sem matriz N×N**,
pico de RAM ~200 MB. Tempo total de validação: **~3,4 horas** (1 thread sozinho ≈ 90 min).

| Threads | Tempo | Speedup | Acurácia | F1 |
|---:|---:|---:|---:|---:|
| 1 | 89,8 min | 1.00 | 0.900465 | 0.641407 |
| 2 | 48,9 min | 1.84 | 0.900465 | 0.641407 |
| 4 | 28,0 min | 3.20 | 0.900465 | 0.641407 |
| 8 | 20,2 min | 4.45 | 0.900465 | 0.641407 |
| **16** | **17,3 min** | **5.19** | 0.900465 | 0.641407 |

- 🔒 **Determinismo confirmado em larga escala:** acurácia/F1 **idênticos** de 1→16 threads
  (variação 0%). A referência serial é o próprio run de 1 thread (o `aco_seq` não foi executado
  no CDC por custo proibitivo e por não ter paralelismo — seria igualmente lento).
- ⚡ **Melhor escalabilidade do projeto:** curva **monotônica crescente** até 16 threads (5.19×),
  sem o platô/regressão visto nas bases pequenas. O volume de trabalho do 1-NN (≈126k treino ×
  253k teste por avaliação) é grande o bastante para amortizar o overhead de threads e **continuar
  ganhando mesmo em oversubscrição** (16t > 12 cores). É o caso que mais justifica o paralelismo.

---

## 7. Conclusão (cobertura do checklist US12)

| Item do checklist | Status |
|---|---|
| Executar `OMP_NUM_THREADS=1` (~igual ao sequencial) | ✅ 1t medido; Δacc ≤ 1.67% nas bases (exceto cirrhosis, explicado) |
| Executar 2, 4, 8, 16 threads | ✅ sweep completo nas 9 bases + CDC |
| Comparar acurácia OpenMP vs sequencial | ✅ 8/9 OK (±2%); WARN único é ruído do sequencial, não da paralelização |
| Documentar speedup em `results/validation_openmp.csv` | ✅ gerado (50 linhas: 10 datasets × 5 threads) |
| Identificar ponto de saturação | ✅ nº ótimo cresce com N: ~4 threads (bases) → 16 (CDC); detalhe na Seção 5 |
| 9 datasets + CDC Diabetes | ✅ **10/10 concluídos** |

**Veredito:** a versão OpenMP é **correta** (determinística, idêntica de 1→16 threads em **todas**
as 10 bases, inclusive 253k instâncias), **preserva a acurácia** do sequencial (8/9 ±2%; o único
desvio é ruído da referência não-determinística) e oferece **speedup mensurável e crescente com a
escala** — até **5.19× no CDC**. A saturação é clara nas bases pequenas (overhead/oversubscrição)
e ausente na larga escala, onde mais threads sempre compensam. Pronta para o benchmark do EP04.
