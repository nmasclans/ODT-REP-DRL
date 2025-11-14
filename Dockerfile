# ODT-REP-DRL Dockerfile
# This Dockerfile sets up an environment for building and running the ODT-REP-DRL code.

# Use the specified NVIDIA CUDA base image, which includes the CUDA toolkit and cuDNN.
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during package installation.
ENV DEBIAN_FRONTEND=noninteractive
ENV CXX=/usr/bin/g++

# Install base dependencies required for ODT-REP-DRL and its build process.
# - software-properties-common is needed for add-apt-repository.
# - We clean up apt lists to keep the image size down.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    git \
    doxygen \
    libboost-all-dev \
    libyaml-cpp-dev \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

# Build and install the fmt library from source as a static library.
# The version in the Ubuntu repository is a shared library, which causes link errors.
RUN cd /usr/src/ && \
    git clone --depth 1 https://github.com/fmtlib/fmt.git && \
    cd fmt && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_POSITION_INDEPENDENT_CODE=TRUE .. && \
    make -j$(nproc)

# Add the Cantera PPA and install the Cantera library and its development headers.
RUN add-apt-repository ppa:cantera-team/cantera && \
    apt-get update && \
    apt-get install -y cantera-python3 libcantera-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory for the project.
WORKDIR /ODT-REP-DRL

