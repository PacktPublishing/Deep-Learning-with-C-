# Deep Learning with C++

A comprehensive collection of deep learning implementations in C++ covering neural networks, CNNs, RNNs, LSTMs, autoencoders, GANs, and transformer-based models.

## Table of Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation](#installation)
  - [1. Install Eigen](#1-install-eigen)
  - [2. Install LibTorch](#2-install-libtorch)
  - [3. Install CUDA (Optional)](#3-install-cuda-optional)
- [Environment Setup](#environment-setup)
- [Running Tests](#running-tests)
- [Chapter Overview](#chapter-overview)
- [Troubleshooting](#troubleshooting)

## Overview

This repository contains implementations of deep learning algorithms and architectures in C++, organized by chapter:

- **Chapter 5**: Neural Networks and Optimizers
- **Chapter 6**: Convolutional Neural Networks (CNNs) and Image Processing
- **Chapter 7**: Recurrent Neural Networks (RNNs) and LSTMs
- **Chapter 8**: Autoencoders, GANs, and Transformer Models

## System Requirements

- **OS**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL
- **Compiler**: g++ with C++17 support (g++ 7.0+)
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (optional, for GPU-accelerated programs)
- **Disk Space**: ~5GB for dependencies

## Installation

### 1. Install Eigen

Eigen is a C++ template library for linear algebra.

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install libeigen3-dev
```

**macOS:**
```bash
brew install eigen
```

**Verify Installation:**
```bash
ls /usr/include/eigen3  # Ubuntu/Debian
ls /usr/local/include/eigen3  # macOS
```

### 2. Install LibTorch

LibTorch is the C++ distribution of PyTorch.

**Download LibTorch (CPU version):**
```bash
cd ~
wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cpu.zip
```

**Download LibTorch (GPU version with CUDA 12.1):**
```bash
cd ~
wget https://download.pytorch.org/libtorch/cu121/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu121.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu121.zip
```

**Note**: Choose CPU or GPU version based on your system. GPU version requires CUDA installation (see next section).

### 3. Install CUDA (Optional)

Required for GPU-accelerated programs.

**Check if CUDA is already installed:**
```bash
nvcc --version
```

**Install CUDA Toolkit (Ubuntu):**
```bash
# For CUDA 12.1
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1
```

**Install cuBLAS:**
```bash
sudo apt-get install libcublas-12-1
```

**Add CUDA to PATH:**
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## Environment Setup

Set the `LIBTORCH_PATH` environment variable to point to your LibTorch installation:

**For CPU version:**
```bash
echo 'export LIBTORCH_PATH=~/libtorch' >> ~/.bashrc
source ~/.bashrc
```

**Verify Setup:**
```bash
echo $LIBTORCH_PATH
ls $LIBTORCH_PATH/lib
```

You should see files like `libtorch.so`, `libtorch_cpu.so`, and `libc10.so`.

## Running Tests

Each chapter has a test script that compiles and runs all programs.

### Chapter 5: Neural Networks and Optimizers

```bash
cd tests/chapter5
./test.sh
```

**Programs tested (4 total):**
- Basic neural network with backpropagation
- CUDA-accelerated MLP
- Optimizers (SGD, Momentum, Adam, RMSprop)
- LibTorch neural network

### Chapter 6: CNNs and Image Processing

```bash
cd tests/chapter6
./test.sh
```

**Programs tested (18 total):**
- Basic convolution layers
- Matrix-based convolution
- CUDA convolution implementations
- Image classification with CNNs
- U-Net for image segmentation
- Image rotation with CUDA
- And more...

**Expected Results:** 17/18 programs working (94%)

See `tests/chapter6/README.md` for detailed results and troubleshooting.

### Chapter 7: RNNs and LSTMs

```bash
cd tests/chapter7
./test.sh
```

**Programs tested (23 items):**
- RNN cells and BPTT implementations
- LSTM networks (simple and matrix-based)
- Text prediction and preprocessing
- Word embeddings
- Sequence-to-sequence models
- GPU-accelerated RNN with CUDA
- And more...

### Chapter 8: Autoencoders, GANs, and Transformers

```bash
cd tests/chapter8
./test.sh
```

**Programs tested:**
- Variational Autoencoder (VAE)
- Sequence-to-Sequence Autoencoders (LSTM and RNN)
- Vision GANs (standard and structured training)
- Autoregressive Language Model (Transformer-based)

## Chapter Overview

### Chapter 5: Neural Networks and Optimizers

Implementations of basic neural networks with various optimization algorithms.

**Key Programs:**
- `neural_network.cpp` - Basic feedforward network with backpropagation
- `cuda_mlp.cu` - GPU-accelerated multi-layer perceptron
- `optimizers.cpp` - SGD, Momentum, Adam, RMSprop optimizers
- `libtorch_nn.cpp` - Neural network using LibTorch

**Dependencies:** Eigen, LibTorch, CUDA (optional)

### Chapter 6: CNNs and Image Processing

Convolutional neural networks for computer vision tasks.

**Key Programs:**
- `BasicConvLayer.cpp` - Basic convolution implementation
- `CudaConvLayer.cu` - GPU-accelerated convolution
- `ImageClassifier.cpp` - CNN for image classification
- `UNet.cpp` - U-Net architecture for segmentation
- `ImageRotation.cpp` - CUDA-based image rotation

**Dependencies:** Eigen, LibTorch, CUDA (optional)

### Chapter 7: RNNs and LSTMs

Recurrent neural networks for sequence modeling.

**Key Programs:**
- `RNNCell.cpp` - Basic RNN cell implementation
- `RNNCell_libtorch.cpp` - RNN cell using LibTorch (CPU/GPU)
- `lstm_simple.cpp` - Simple LSTM network
- `rnn_bptt_eigen.cpp` - RNN with backpropagation through time
- `text_prediction.cpp` - Character-level text prediction
- `rnn_bptt_gpu.cu` - GPU-accelerated RNN

**Dependencies:** Eigen, LibTorch, CUDA (optional)

### Chapter 8: Autoencoders, GANs, and Transformers

Advanced generative models and transformer architectures.

**Key Programs:**
- `vae.cpp` - Variational Autoencoder
- `seq2seq_autoencoder.cpp` - Sequence-to-sequence with LSTM
- `vision_gan.cpp` - Generative Adversarial Network for images
- `autoregressive_llm.cpp` - Transformer-based language model

**Dependencies:** LibTorch (required), CUDA (optional)

## Troubleshooting

### Common Issues

**1. "Eigen not found" error:**
```bash
# Verify Eigen installation
ls /usr/include/eigen3

# If not found, install:
sudo apt-get install libeigen3-dev
```

**2. "LibTorch not found" error:**
```bash
# Check LIBTORCH_PATH
echo $LIBTORCH_PATH

# Set it if not set:
export LIBTORCH_PATH=~/libtorch
```

**3. "cannot find -ltorch" error:**
```bash
# Verify LibTorch libraries exist
ls $LIBTORCH_PATH/lib/libtorch.so

# If missing, re-download LibTorch
```

**4. "nvcc: command not found" error:**
```bash
# Install CUDA toolkit
sudo apt-get install nvidia-cuda-toolkit

# Or add CUDA to PATH
export PATH=/usr/local/cuda/bin:$PATH
```

**5. "cannot find -lcublas" error:**
```bash
# Install cuBLAS
sudo apt-get install libcublas-dev
```

**6. Compilation errors with LibTorch:**
- Ensure you're using LibTorch 2.1.0 (compatible with C++17)
- Use the cxx11 ABI version
- Check that all include paths are correct

**7. Runtime "undefined symbol" errors:**
```bash
# Add LibTorch to library path
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH

# Or use -Wl,-rpath during compilation
```

### Manual Compilation Examples

**Basic C++ program:**
```bash
g++ -std=c++17 program.cpp -o program
```

**With Eigen:**
```bash
g++ -std=c++17 -I/usr/include/eigen3 program.cpp -o program
```

**With LibTorch:**
```bash
g++ -std=c++17 program.cpp -o program \
  -I$LIBTORCH_PATH/include \
  -I$LIBTORCH_PATH/include/torch/csrc/api/include \
  -L$LIBTORCH_PATH/lib \
  -ltorch -ltorch_cpu -lc10 \
  -Wl,-rpath,$LIBTORCH_PATH/lib
```

**CUDA program:**
```bash
nvcc program.cu -o program -lcublas
```

**CUDA with Eigen:**
```bash
nvcc -I/usr/include/eigen3 program.cu -o program -lcublas
```

### Getting Help

For detailed chapter-specific information, see:
- `tests/chapter5/README.md`
- `tests/chapter6/README.md`
- `tests/chapter7/README.md`

## Performance Notes

- **CPU vs GPU**: GPU-accelerated programs (CUDA) can be 10-100x faster for large models
- **LibTorch**: Provides automatic GPU acceleration when CUDA is available
- **Eigen**: Optimized for CPU operations, uses SIMD instructions
- **Batch Size**: Larger batches improve GPU utilization but require more memory

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please ensure:
- Code compiles with C++17
- Test scripts pass
- Documentation is updated

## Acknowledgments

This repository demonstrates deep learning concepts using:
- **Eigen**: Linear algebra library
- **LibTorch**: PyTorch C++ API
- **CUDA**: NVIDIA GPU computing platform
