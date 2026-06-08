# Use NVIDIA CUDA 12.2.2 Developer Image on Ubuntu 22.04
FROM nvidia/cuda:12.2.2-devel-ubuntu22.04

# Disable interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    python3-dev \
    bc \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /workspace

# Copy Python requirements first for caching
COPY requirements.txt /workspace/

# Install Python packages
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the entire workspace into the container
COPY . /workspace/

# Compile all three ACO implementations (sequential, openmp, cuda)
# Note: Makefile targets are optimized for sm_89 (RTX 4090)
RUN make clean && make all

# Download and bake the CDC Diabetes dataset (250k+ instances) into the image
RUN python3 scripts/download_cdc.py

# Default command starts a bash shell
CMD ["/bin/bash"]
