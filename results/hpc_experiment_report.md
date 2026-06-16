# Relatório de Performance Científico - ACO Instance Reduction

*   **Data:** Mon Jun 15 09:07:06 PM -03 2026
*   **Dataset:** data/baseline/diabetes.csv
*   **Parâmetros:** 2 formigas, 1 iterações (sem early-stopping)

## Tabela Geral de Resultados

| Modo | Configuração | Scheduler | Tempo ACO (ms) | Tempo Eval (ms) | Throughput (GFLOPS) | Speedup (vs Seq) | IPC | Cache Misses (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Sequencial | 1 thread | N/A | 5.9245 | 2.11378 | 10.3941 | 1.00x (ref) | N/A | N/A |
| OpenMP | 1 threads | static | 8.4771 | 3.53822 | 6.24086 | 0.70x | N/A | N/A |
| OpenMP | 1 threads | dynamic | 6.05571 | 2.0751 | 10.6412 | 0.98x | N/A | N/A |
| OpenMP | 1 threads | guided | 7.18741 | 3.46455 | 6.37357 | 0.82x | N/A | N/A |
| OpenMP | 2 threads | static | 6.74031 | 2.07194 | 10.6574 | 0.88x | N/A | N/A |
| OpenMP | 2 threads | dynamic | 5.75991 | 1.69916 | 12.9956 | 1.03x | N/A | N/A |
| OpenMP | 2 threads | guided | 7.17105 | 2.06328 | 10.7021 | 0.83x | N/A | N/A |
| OpenMP | 4 threads | static | 5.6383 | 1.38453 | 15.9488 | 1.05x | N/A | N/A |
| OpenMP | 4 threads | dynamic | 5.86259 | 1.00197 | 22.0381 | 1.01x | N/A | N/A |
| OpenMP | 4 threads | guided | 5.68246 | 1.19184 | 18.5273 | 1.04x | N/A | N/A |
| OpenMP | 8 threads | static | 6.29897 | 1.85166 | 11.9252 | 0.94x | N/A | N/A |
| OpenMP | 8 threads | dynamic | 4.98643 | 0.473276 | 46.6568 | 1.19x | N/A | N/A |
| OpenMP | 8 threads | guided | 4.7387 | 0.9828 | 22.468 | 1.25x | N/A | N/A |
| OpenMP | 16 threads | static | 6.59963 | 1.73327 | 12.7398 | 0.90x | N/A | N/A |
| OpenMP | 16 threads | dynamic | 9.1485 | 1.05284 | 20.9733 | 0.65x | N/A | N/A |
| OpenMP | 16 threads | guided | 5.09184 | 0.719731 | 30.6803 | 1.16x | N/A | N/A |
| OpenMP | 32 threads | static | 5.82732 | 1.67216 | 13.2054 | 1.02x | N/A | N/A |
| OpenMP | 32 threads | dynamic | 5.96054 | 0.681537 | 32.3996 | 0.99x | N/A | N/A |
| OpenMP | 32 threads | guided | 5.69433 | 0.589822 | 37.4376 | 1.04x | N/A | N/A |
| CUDA | Block size: 32 | N/A | 0.925519 | 0.777408 | 28.8545 | 6.40x | N/A | N/A |
| CUDA | Block size: 64 | N/A | 0.921628 | 0.783104 | 28.6447 | 6.43x | N/A | N/A |
| CUDA | Block size: 128 | N/A | 1.08837 | 0.950432 | 23.6016 | 5.44x | N/A | N/A |
| CUDA | Block size: 256 | N/A | 1.68825 | 1.53514 | 14.6122 | 3.51x | N/A | N/A |
| CUDA | Block size: 512 | N/A | 2.91048 | 2.76653 | 8.10827 | 2.04x | N/A | N/A |
| CUDA | Block size: 1024 | N/A | 4.28746 | 4.14106 | 5.41691 | 1.38x | N/A | N/A |
