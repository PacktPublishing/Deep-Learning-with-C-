# Chapter 9: Transformer & Distributed Training

Complete implementation of Transformer architecture with distributed training (DDP, FSDP) and model compression techniques.

📖 **See [CHAPTER9_GUIDE.md](CHAPTER9_GUIDE.md) for comprehensive documentation**

## Quick Start

### Prerequisites
- LibTorch (PyTorch C++)
- CUDA Toolkit + NCCL
- C++17 compiler
- CMake 3.18+

### Install LibTorch
```bash
# Download pre-built (CUDA 11.8)
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip

# Or use Python installation
pip install torch
```

### Build All Targets
```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make

# Or with Python's torch
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..
make
```

### Build Specific Target
```bash
make transformer          # Core transformer
make ddp_training         # DDP training
make fsdp_training        # FSDP training
make knowledge_distillation
make magnitude_pruning
make gradient_pruning
make wanda_pruning
make quantization
```

### Run Examples

**Single GPU:**
```bash
./transformer
./bert_train
./knowledge_distillation
```

**Multi-GPU DDP (2 GPUs):**
```bash
./ddp_training 0 2 &
./ddp_training 1 2 &
```

**Multi-GPU FSDP (2 GPUs):**
```bash
./fsdp_training 0 2 &
./fsdp_training 1 2 &
```

## File Structure

**Core:** encoder, decoder, transformer, attention, positional encoding  
**Distributed:** ddp_training, fsdp_training, ddp_transformer, fsdp_transformer  
**Compression:** knowledge_distillation, magnitude_pruning, gradient_pruning, wanda_pruning, quantization, low_rank_approximation  
**Utils:** training_utils, data_loader  
**Examples:** bert_train, train_encoder, train_decoder, train_encoder_decoder