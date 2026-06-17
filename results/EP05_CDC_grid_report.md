# Relatório — Grade CDC (OpenMP threads×escalonador + CUDA)

**Baseline T1 (1 thread):** 264,510.0 ms/iteração (de `results/cdc_t1_baseline.log`)

> Speedup = (T1/iteração) / (Tp/iteração) — normalizado por iteração por causa do early stop (K varia). Ressalva: alguns laços do ACO têm escalonador fixo; o laço quente (eval 1-NN) respeita `OMP_SCHEDULE`.

> ⚠️ **VERSÃO PRELIMINAR (v1) — speedups absolutos subestimados.** O baseline T1 vem de um run
> **anterior** (não da mesma leva): o 2-thread novo (297.809 ms/iter) é mais lento que o "1-thread"
> antigo (264.510 ms/iter), o que é impossível para paralelização real → os dois não são
> comparáveis. Efeito: **todas as speedups estão ~2× subestimadas** (por isso o 2-thread aparece
> <1×, e o CUDA ~49× quando o real deve ser ~100×). Um baseline de 1-thread da **mesma leva**
> (em execução) corrige isso na versão definitiva. **O sinal confiável agora é a escalabilidade
> dentro da leva — ver a última seção.**


## CUDA (por block size)

| Block size | Tempo ACO (ms) | Eval (ms) | GFLOPS | Speedup | Iter | Status |
| ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 32 | 119682 | 119507 | 573.043 | 48.62x | 22 | OK |
| 64 | 151400 | 151224 | 452.857 | 38.44x | 22 | OK |
| 128 | 149936 | 149759 | 457.286 | 38.81x | 22 | OK |
| 256 | 135250 | 135073 | 507.005 | 43.03x | 22 | OK |
| 512 | 127939 | 127763 | 536.016 | 45.48x | 22 | OK |
| 1024 | 118475 | 118296 | 578.911 | 49.12x | 22 | OK |

## OpenMP (threads × escalonador)

| Escalonador | Threads | Tempo ACO (ms) | Eval (ms) | GFLOPS | Speedup | Iter | Status |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| dynamic | 2 | 7.12464e+06 | 6.6827e+06 | 10.2481 | 0.82x | 22 | OK |
| dynamic | 4 | 3.47486e+06 | 3.27319e+06 | 20.9231 | 1.67x | 22 | OK |
| dynamic | 8 | 2.17273e+06 | 2.04268e+06 | 33.527 | 2.68x | 22 | OK |
| dynamic | 16 | 1.08579e+06 | 1.01585e+06 | 67.4165 | 5.36x | 22 | OK |
| dynamic | 32 | 514194 | 477918 | 143.299 | 11.32x | 22 | OK |
| static | 2 | 6.5518e+06 | 6.10642e+06 | 11.2153 | 0.89x | 22 | OK |
| static | 4 | 3.61234e+06 | 3.40815e+06 | 20.0945 | 1.61x | 22 | OK |
| static | 8 | 1.98051e+06 | 1.85478e+06 | 36.9236 | 2.94x | 22 | OK |
| static | 16 | 989271 | 918246 | 74.5825 | 5.88x | 22 | OK |
| static | 32 | 583868 | 548229 | 124.921 | 9.97x | 22 | OK |

## Escalabilidade OpenMP dentro da leva (confiável)

Speedup relativo ao **2-thread da mesma leva** (mesmo código, todos 22 iters) — não depende do
T1 antigo, então é o número honesto de escalabilidade:

| vs 2 threads | static | dynamic |
| :--- | ---: | ---: |
| 4t | 1.81× | 2.05× |
| 8t | 3.31× | 3.28× |
| 16t | 6.62× | 6.56× |
| **32t** | **11.22×** | **13.86×** |

Cada dobra de threads dá ~1.8–2.0× (escala quase linear até os 12 cores físicos, com ganho extra
em 32t por oversubscrição). Tempo absoluto mínimo: **dynamic 32t = 514.194 ms (~8,6 min)** para
22 iterações. O CUDA (melhor: bs=1024, ~118 s) é ainda ~4,3× mais rápido que a melhor config OpenMP.

## Notas

- **CUDA:** todas as 6 block sizes convergem idênticas (F1=0.7728, redução 23.11%, early stop em
  22 iters) — block size muda só a velocidade. Curva em U: melhor nos extremos (bs=1024 ⩰ bs=32),
  pior no meio (bs=64). GPU compute (construção) é ~6 ms/iter; ~99,9% do tempo é o eval 1-NN.
- **Qualidade idêntica entre OpenMP e CUDA:** ambos reduzem o dataset em ~23% mantendo F1 ~0.77.
- **Definitivo:** ao terminar o 1-thread fresco (~3 h), regenerar com
  `python3 scripts/build_cdc_report.py --csv results/EP05_CDC_grid_results.csv --t1-log results/cdc_t1_fresh.log --out results/EP05_CDC_grid_report.md`.
