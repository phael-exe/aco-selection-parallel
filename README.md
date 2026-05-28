# aco-selection-parallel

![CI](https://github.com/phael-exe/aco-selection-parallel/actions/workflows/ci.yml/badge.svg)

Paralelização do Algoritmo de Otimização de Colônia de Formigas (ACO) com CUDA e OpenMP para seleção de instâncias em larga escala.

**Disciplina:** Computação de Alto Desempenho — UFG 2026  
**Baseline:** [Magalhães et al. (PUC Minas)](https://github.com/jmarcosjm/aco-cpp)

## Equipe

| Nome | Matrícula |
|------|-----------|
| Wagner Victor Alves de Menezes | 202403929 |
| Daniel Rios Borges | 202403900 |
| Victor Gabriel Ribeiro Jácome | 202403926 |
| Raphael Alves de Lima Soares | 202403922 |

## Estrutura do Projeto

```
├── data/                  # Datasets (9 do baseline + CDC Diabetes)
├── src/
│   ├── sequential/        # C++ sequencial (adaptado do baseline)
│   ├── cuda/              # C++ + CUDA (paralelismo GPU)
│   └── openmp/            # C++ + OpenMP (paralelismo CPU)
├── scripts/               # Benchmark, profiling (Perf), plots
├── results/               # Saídas experimentais
├── docs/                  # Artigo final e apresentação
└── Makefile               # Build unificado
```

## Versões

| Versão | Tecnologia | Status |
|--------|------------|--------|
| Sequencial | C++ | 🔲 |
| GPU | C++ + CUDA | 🔲 |
| CPU Paralela | C++ + OpenMP | 🔲 |

## Datasets

| Categoria | Datasets | Instâncias |
|-----------|----------|------------|
| Pequenos (≤700) | Heart, Haberman, Cirrhosis | 299 – 418 |
| Médios (≤1500) | Diabetes, Tic-tac-toe, Yeast | 767 – 1.484 |
| Grandes (>1500) | Vaccine, Employee, Brain Stroke | 3.152 – 4.981 |
| **Larga escala** | **CDC Diabetes** | **253.680** |

## Como compilar

```bash
make sequential   # compila versão sequencial   -> build/aco_seq
make cuda         # compila versão CUDA          -> build/aco_cuda
make openmp       # compila versão OpenMP        -> build/aco_omp
make all          # compila as 3 versões
make benchmark    # compila e executa as versões disponíveis
make clean        # remove build/
```

> CUDA e OpenMP são épicos separados. Enquanto `src/cuda/` e `src/openmp/`
> estiverem vazios (ou `nvcc` não existir), esses targets são pulados sem erro.

## Referências

- Magalhães, J.M.O. et al. *Avaliação de Desempenho e Escalabilidade do ACO em C++ e Python*. PUC Minas, 2024.
- CDC Diabetes Health Indicators — [UCI ML Repository](https://archive.ics.uci.edu/dataset/891)
