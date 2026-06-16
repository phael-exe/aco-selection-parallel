# Relatório de Performance Científico - ACO Instance Reduction

*   **Data:** seg 15 jun 2026 21:11:17 -03
*   **Dataset:** data/cdc/cdc_diabetes.csv
*   **Parâmetros:** 64 formigas, 100 iterações (sem early-stopping)

## Tabela Geral de Resultados

| Modo | Configuração | Scheduler | Tempo ACO (ms) | Tempo Eval (ms) | Throughput (GFLOPS) | Speedup (vs Seq) | IPC | Cache Misses (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Sequencial | 1 thread | N/A | 2.6494e+07 | 2.59355e+07 | 10.0788 | 1.00x (ref) | N/A | N/A |
| OpenMP | 1 threads | static | 2.6451e+07 | 2.58922e+07 | 10.0956 | 1,00x | N/A | N/A |
| OpenMP | 1 threads | dynamic | 2.67882e+07 | 2.6231e+07 | 9.96525 | 0,99x | N/A | N/A |
