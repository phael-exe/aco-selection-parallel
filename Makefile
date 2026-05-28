# Makefile unificado — aco-selection-parallel
# Targets: sequential | cuda | openmp | all | clean
#
# Compila qualquer uma das 3 versoes com um comando padronizado.
# CUDA/OpenMP sao epicos separados: enquanto suas pastas estao vazias
# (ou nvcc nao existe), os targets pulam de forma graciosa, sem erro.

CXX       = g++
NVCC      = nvcc
CXXFLAGS  = -std=c++17 -O2 -Wall -Wextra
NVCCFLAGS = -std=c++17 -O2
OMPFLAGS  = -fopenmp
BUILD_DIR = build

SEQ_SRC  = $(wildcard src/sequential/*.cpp)
SEQ_HDR  = $(wildcard src/sequential/*.h)
OMP_SRC  = $(wildcard src/openmp/*.cpp)
OMP_HDR  = $(wildcard src/openmp/*.h)
CUDA_SRC = $(wildcard src/cuda/*.cu)

SEQ_BIN  = $(BUILD_DIR)/aco_seq
OMP_BIN  = $(BUILD_DIR)/aco_omp
CUDA_BIN = $(BUILD_DIR)/aco_cuda

# Detecta se o nvcc esta disponivel (portabilidade entre maquinas).
NVCC_AVAILABLE := $(shell command -v $(NVCC) 2>/dev/null)

.PHONY: all sequential cuda openmp clean

all: sequential cuda openmp

# ---- Sequencial -------------------------------------------------------------
sequential: $(SEQ_BIN)

$(SEQ_BIN): $(SEQ_SRC) $(SEQ_HDR)
ifeq ($(SEQ_SRC),)
	@echo "[sequential] nenhum fonte em src/sequential/ — pulando."
else
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -o $@ $(SEQ_SRC)
	@echo "[sequential] OK -> $@"
endif

# ---- OpenMP (epico separado) ------------------------------------------------
openmp: $(OMP_BIN)

$(OMP_BIN): $(OMP_SRC) $(OMP_HDR)
ifeq ($(OMP_SRC),)
	@echo "[openmp] nenhum fonte em src/openmp/ — pulando (epico OpenMP)."
else
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(OMPFLAGS) -o $@ $(OMP_SRC)
	@echo "[openmp] OK -> $@"
endif

# ---- CUDA (epico separado) --------------------------------------------------
cuda: $(CUDA_BIN)

$(CUDA_BIN): $(CUDA_SRC)
ifeq ($(CUDA_SRC),)
	@echo "[cuda] nenhum fonte em src/cuda/ — pulando (epico CUDA)."
else ifeq ($(NVCC_AVAILABLE),)
	@echo "[cuda] nvcc nao encontrado — pulando."
else
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -o $@ $(CUDA_SRC)
	@echo "[cuda] OK -> $@"
endif

# ---- Limpeza ----------------------------------------------------------------
clean:
	rm -rf $(BUILD_DIR)
