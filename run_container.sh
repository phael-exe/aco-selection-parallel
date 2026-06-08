#!/usr/bin/env bash
# ===========================================================================
# aco-selection-parallel — Container Execution Helper
#
# Contains commands to build and run the 3 ACO versions (sequential, OpenMP,
# and CUDA) on the UCI CDC Diabetes dataset (~250k instances).
# ===========================================================================

set -euo pipefail

IMAGE_NAME="aco-selection-parallel"
SIF_NAME="aco.sif"
DATASET="data/cdc/cdc_diabetes.csv"
TARGET="Diabetes_binary"

print_usage() {
    echo "Uso: ./run_container.sh [opção]"
    echo ""
    echo "Opções:"
    echo "  build-docker      - Build da imagem Docker (com o dataset e executáveis)"
    echo "  build-singularity - Build do arquivo Singularity (.sif) para HPCs/maquinas sem root"
    echo "  run-seq           - Executa o ACO sequencial no dataset CDC (usa amostragem)"
    echo "  run-omp <threads> - Executa o ACO OpenMP com o número de threads especificado"
    echo "  run-cuda          - Executa o ACO CUDA na GPU RTX 4090"
    echo "  run-benchmark     - Roda o benchmark completo e salva relatórios em ./results/"
    echo ""
}

if [ $# -lt 1 ]; then
    print_usage
    exit 1
fi

ACTION="$1"

case "$ACTION" in
    build-docker)
        echo "=== [DOCKER] Construindo Imagem ==="
        docker build -t "$IMAGE_NAME" .
        echo "Pronto! Imagem '$IMAGE_NAME' construída com sucesso."
        ;;

    build-singularity)
        echo "=== [SINGULARITY] Construindo arquivo .sif ==="
        if command -v singularity &> /dev/null; then
            singularity build --fakeroot "$SIF_NAME" Dockerfile
        elif command -v apptainer &> /dev/null; then
            apptainer build --fakeroot "$SIF_NAME" Dockerfile
        else
            echo "Erro: Singularity ou Apptainer não encontrado no PATH."
            exit 1
        fi
        echo "Pronto! Arquivo '$SIF_NAME' gerado com sucesso."
        ;;

    run-seq)
        echo "=== [DOCKER] Executando ACO Sequencial (Amostragem Padrão = 5k) ==="
        docker run --rm "$IMAGE_NAME" ./build/aco_seq "$DATASET" "$TARGET" --ants 64 --iter 10
        ;;

    run-omp)
        THREADS="${2:-16}"
        echo "=== [DOCKER] Executando ACO OpenMP com $THREADS threads (Dataset completo) ==="
        docker run --rm -e OMP_NUM_THREADS="$THREADS" "$IMAGE_NAME" ./build/aco_omp "$DATASET" "$TARGET" --ants 64 --iter 10
        ;;

    run-cuda)
        echo "=== [DOCKER] Executando ACO CUDA na GPU RTX 4090 (Dataset completo) ==="
        docker run --gpus all --rm "$IMAGE_NAME" ./build/aco_cuda "$DATASET" "$TARGET" --ants 64 --iter 10
        ;;

    run-benchmark)
        echo "=== [DOCKER] Executando benchmark completo ==="
        echo "Os relatórios gerados serão salvos em $(pwd)/results/"
        mkdir -p results
        docker run --gpus all --rm -v "$(pwd)/results:/workspace/results" "$IMAGE_NAME" bash scripts/run_benchmark_comparison.sh
        ;;

    *)
        print_usage
        exit 1
        ;;
esac
