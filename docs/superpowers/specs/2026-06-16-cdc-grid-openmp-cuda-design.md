# Design — Grade de Experimentos no CDC (OpenMP threads×escalonador + CUDA)

**Data:** 2026-06-16
**Autor:** phael-exe (com Claude)
**Status:** Aprovado (design) — aguardando revisão do spec

---

## 1. Contexto e motivação

Os resultados atuais em `results/` (EP03/EP04) vêm de **experimentos iniciais com datasets
pequenos** (ex.: `diabetes.csv`, 768 linhas, 2 formigas, 1 iteração) — números na casa de
**milissegundos**, dominados por ruído de criação/sincronização de threads, sem valor de
escalabilidade real.

A tentativa anterior de rodar no dataset grande (**CDC Diabetes, N=253.680**) via
`scripts/run_hpc_experiment.sh` **morreu na 4ª rodada** (~22h de execução), durante o
OpenMP 1-thread `guided`. Evidências cruzadas:

- `results_remote/experiment.log` para no meio do `guided` (1 thread);
- `results_remote/hpc_raw_runs.log` (653 linhas) termina no OpenMP 1-thread `dynamic`;
- `results_remote/temp_run.txt` vazio (morto escrevendo a rodada em andamento);
- **nenhuma** seção `Resultado ACO (CUDA)` no log → o CUDA **nunca rodou** (está por último
  no script, depois de 1 sequencial + 18 OpenMP, cada um ~7h).

Causa raiz: cada rodada no CDC completo (100 iter, sem amostragem) leva **~7,35h** porque o
eval 1-NN é O(N²) e o binário OpenMP **não tem** flag de amostragem (só `--eval-top-k`, já
auto-ajustado para 1 no N grande). O `run_hpc_experiment.sh` roda o CUDA por último → inviável.

**Objetivo deste projeto:** medir, no CDC grande, (a) a escalabilidade do OpenMP por nº de
threads e tipo de escalonador, e (b) o desempenho CUDA por block size — com execução
**isolada** e **à prova de queda**.

## 2. Decisões já tomadas (com o usuário)

| Decisão | Escolha |
|---|---|
| Fidelidade | **Modo cheio**: `--ants 64 --iter 100` (teto), sem amostragem |
| Early stopping | **Ativado**: para após **20 iterações sem melhora** (`--patience 20`) |
| Threads (OpenMP) | `{2, 4, 8, 16, 32}` (1-thread **removido** — já medido) |
| Escalonadores | `{static, dynamic}` (`guided` **removido** — não estudado) |
| 1-thread (T1) | **Não roda**; reaproveita medição existente do log remoto |
| CUDA | block sizes `{32, 64, 128, 256, 512, 1024}`, roda **1º** |
| Execução | **Isolada** (nunca CPU+GPU juntos); CUDA antes do OpenMP |
| Co-execução híbrida CPU+GPU | **Fora de escopo** (EP futuro) |
| Onde roda | Máquina remota `aluno@100.95.177.8` (script entregue; usuário executa) |

## 3. Validade do experimento de escalonador

Verificação dos pragmas em `src/openmp/` confirma que **o laço quente respeita `OMP_SCHEDULE`**:

- `src/openmp/knn.cpp:38` — eval 1-NN (**~98% do tempo**) → `schedule(runtime)` ✅
- `src/openmp/aco_omp.cpp:89, 323, 333` → `schedule(runtime)` ✅
- `src/openmp/aco_omp.cpp:48` — pré-cálculo distância/visibilidade (uma vez) → `schedule(dynamic)` fixo
- `src/openmp/aco_omp.cpp:182, 224, 344` → sem cláusula (default `static`)

Como o eval 1-NN domina e usa `schedule(runtime)`, variar `OMP_SCHEDULE` entre `static` e
`dynamic` mede um efeito **real**. Os poucos laços fixos são pré-cálculo de peso menor; isso
será **documentado no relatório** para não superinterpretar a diferença observada.

**Para o experimento de escalonador, nenhuma alteração de código é necessária.** (O early
stopping — seção 3.5 — exige uma mudança pequena e contida.)

## 3.5 Early stopping (mudança de código)

O early stopping **já existe implementado** nos dois binários, porém **comentado** de propósito:

- `src/openmp/aco_omp.cpp:407-414` — bloco `/* ... */`; `config.patience` default 10; **sem flag CLI**
- `src/cuda/main.cu:518-524` — bloco `/* ... */`; `patience = 10` hardcoded

**Mudança planejada (contida, nos dois binários):**

1. Descomentar o bloco de early stop.
2. Adicionar a flag CLI **`--patience N`**, com **default `0` = desligado**. Isso preserva a
   reprodutibilidade dos EP03/EP04 antigos (que rodaram sem early stop) — o comportamento só muda
   quando `--patience > 0` é passado explicitamente.
3. Gatear o `break` com `if (config.patience > 0 && no_improve >= config.patience)`.
4. Recompilar (`make all`). O grid passa `--patience 20`.

Critério "sem melhora": `no_improve` zera quando o F1 global melhora e incrementa caso contrário
(lógica já presente). Para após 20 iterações consecutivas sem melhora. O binário já imprime
`Iteracoes executadas: K/100` e `(early stop)` quando para antes do teto.

## 4. Baseline T1 (reaproveitado do log, não re-executado)

Do `results_remote/hpc_raw_runs.log` (CDC, 100 iter, 64 formigas):

| Config | Tempo ACO (ms) | Tempo eval (ms) | GFLOPS | F1 | Acur. | Redução |
|---|---:|---:|---:|---:|---:|---:|
| OpenMP 1t `static` | 2.6451e7 (~7,35h) | 2.58922e7 | 10.0956 | 0.773286 | 0.936798 | 23.05% |
| OpenMP 1t `dynamic` | 2.67882e7 (~7,44h) | 2.6231e7 | 9.96525 | 0.773286 | 0.936798 | 23.05% |

**Comparabilidade com early stop.** Os runs novos param em `K ≤ 100` iterações; o T1 logado rodou
100/100. Como o custo por iteração é ~constante e o ACO é **determinístico** (mesma trajetória em
toda config), o baseline é derivado **do próprio log**, sem re-rodar 1-thread:

- **Tempo por iteração:** `T1_iter = tempo_aco_ms(T1) / 100`.
- **K efetivo do T1:** lido da trajetória logada (`Iter i: F1=...`) — última iteração com melhora de
  F1 **+ 20** (paciência). Os runs novos param nesse mesmo K (determinismo).
- **T1 efetivo:** `T1_efetivo = T1_iter × K`.
- **Speedup:** `speedup_sched = T1_efetivo_sched / Tp_sched` (T1 de `static` e `dynamic` já medidos;
  `guided` removido, não é necessário).

Um script auxiliar de parsing extrai `T1_iter` e `K` do log remoto e injeta no relatório.

## 5. Arquitetura: script novo dedicado

Criar **`scripts/run_cdc_grid.sh`** (novo), em vez de alterar `run_hpc_experiment.sh`.
Justificativa: o script antigo mistura baseline sequencial (~7h), coleta `perf` e suposições de
dataset pequeno, e é o que gerou os relatórios EP03/EP04 — alterá-lo quebraria a
reprodutibilidade daqueles. O script novo é focado, resumível e sem baggage.

### 5.1 Fluxo

```
1. Pré-checagem: dataset existe; binários compilados (make all se faltar)
2. CUDA primeiro (também valida N grande na GPU — hipótese OOM/watchdog):
     para bs em {32,64,128,256,512,1024}:
        se (modo=CUDA, block=bs) já no CSV → pula (resume)
        roda ./build/aco_cuda <ds> <target> --ants 64 --iter 100 --patience 20 --block-size bs
        parseia métricas; grava 1 linha no CSV + bloco no log
        se falhar → status=ERRO, registra stderr, segue
3. OpenMP depois:
     para sched em {static,dynamic}:
       para t em {2,4,8,16,32}:
          se (modo=OpenMP, threads=t, sched) já no CSV → pula (resume)
          OMP_NUM_THREADS=t OMP_SCHEDULE=sched ./build/aco_omp <ds> <target> --ants 64 --iter 100 --patience 20
          parseia métricas; calcula speedup vs T1[sched]; grava linha + bloco
4. Gera/atualiza results/cdc_grid_report.md a partir do CSV
```

### 5.2 Robustez

- **Auto-encadeamento:** loop sequencial único — ao detectar o fim de um run (saída do binário +
  parse bem-sucedido), o próximo config é disparado automaticamente, sem intervenção. CUDA inteiro
  primeiro, depois toda a grade OpenMP.
- **Detached:** usuário invoca com `nohup ... &` (sobrevive a desconexão SSH).
- **Incremental:** CSV e log bruto recebem `append` **após cada run** (não no fim).
- **Resumível:** no início, lê o CSV existente; pula qualquer config (chave
  `modo,threads,escalonador,block_size`) já presente com `status=OK`. Queda na hora 20
  retoma de onde parou.
- **Tolerante a falha CUDA:** falha de uma config (OOM/watchdog) vira `status=ERRO` e o
  script **continua** — não usa `set -e` no laço de execução (usa `|| true` por run).

## 6. Saídas (em `results/`)

- **`cdc_grid_raw.log`** — stdout/stderr completo de cada execução (auditoria).
- **`cdc_grid_results.csv`** — uma linha por config. Colunas:
  ```
  modo,threads,escalonador,block_size,tempo_aco_ms,tempo_eval_ms,gflops,
  speedup_vs_t1,f1,acuracia,reducao_pct,iteracoes,status,timestamp
  ```
  (CUDA: `threads`/`escalonador` vazios; OpenMP: `block_size` vazio.)
- **`cdc_grid_report.md`** — tabela geral + leitura, no estilo EP03/EP04, incluindo a ressalva
  da seção 3 sobre laços com escalonador fixo.

### 6.1 Parsing das métricas

Reaproveita a lógica já validada em `run_hpc_experiment.sh`, que trata as duas saídas:
- OpenMP: `Tempo ACO:`, `Tempo eval (1-NN):`, `CPU eval throughput:`
- CUDA: `Tempo ACO:` (ou `GPU compute:`), `GPU eval (1-NN):`, `GPU eval throughput:`
- Comuns: `F1-Score:`, `Acuracia:`, `Melhor solucao: .../... (reducao ~XX%)`,
  `Iteracoes executadas:`

## 7. Estimativa de tempo

Âncora medida: OpenMP 1t = ~7,35h para 100 iter. Speedups do EP03 no CDC (2t≈1.84×, 4t≈3.2×,
8t≈4.45×, 16t≈5.19×, 32t≈plateau).

**Teto (se nunca houver early stop, K=100):**

```
CUDA (6 block sizes):     ~30-90 min   (+ valida GPU no N grande)
OpenMP static  {2..32}:   ~10,8h       (4,0 + 2,3 + 1,7 + 1,4 + 1,4)
OpenMP dynamic {2..32}:   ~10,8h
─────────────────────────────────────
TETO:                     ≈ 22h
```

**Esperado com early stop:** o tempo escala por `K/100`. Se a convergência cair em ~30-50 iter
(`--patience 20`), o total fica em **~7-11h**. O K real só se conhece ao rodar; o teto de 22h é o
pior caso (sem nenhuma parada antecipada).

## 8. Como executar (usuário, no remoto)

```bash
# na máquina remota, em ~/aco-selection-parallel:
nohup bash scripts/run_cdc_grid.sh > results/cdc_grid_console.log 2>&1 &
tail -f results/cdc_grid_console.log     # acompanhar
# se cair: basta rodar o mesmo comando de novo — ele pula o que já terminou
```

Depois, trazer os resultados:
```bash
scp -r aluno@100.95.177.8:~/aco-selection-parallel/results/cdc_grid_* ./results/
```

## 9. Fora de escopo (YAGNI / futuro)

- Co-execução híbrida CPU+GPU real (particionar trabalho entre dispositivos) — EP separado.
- Implementar amostragem do eval 1-NN no binário — não necessário (escolhido modo cheio).
- Re-rodar 1-thread — reaproveitado do log existente.
- Varredura `guided` — removida.
- Plots — os scripts de plot existentes (`scripts/plot_instance_reduction.py`) podem consumir o
  CSV depois; geração de figuras não faz parte deste script.

## 10. Critérios de sucesso

1. `cdc_grid_results.csv` com **16 linhas OK** (6 CUDA + 10 OpenMP) ou linhas `ERRO`
   explicáveis (ex.: CUDA OOM documentado).
2. Reexecutar o script após uma interrupção **não repete** runs já concluídos.
3. `cdc_grid_report.md` apresenta as curvas de speedup por escalonador e a tabela CUDA por
   block size, com a ressalva da seção 3.
4. CUDA confirmado funcionando (ou falha diagnosticada) no N grande logo no início.
5. Early stop funcionando: binários param em `K < 100` quando há 20 iterações sem melhora, e
   `iteracoes` (K efetivo) é registrado no CSV de cada run.
