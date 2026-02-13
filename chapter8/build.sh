#!/bin/bash

# Build script for AutoEncoder projects
echo "Building AutoEncoder projects..."

# Create build directory
mkdir -p build
cd build

# Configure with CMake
cmake -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" ..

# Build
make -j$(nproc)

echo "Build complete!"
echo "Run original autoencoder with: ./build/autoencoder"
echo "Run variational autoencoder with: ./build/vae"