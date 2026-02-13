# Chapter 9: Transformer Architecture & Distributed Training

## Overview
This chapter implements the complete Transformer architecture with distributed training (DDP, FSDP) and model compression techniques.

---

## Core Architecture Files

### Attention Mechanisms
- **multihead_attention.cpp** - Multi-head attention with linear transformations
- **attention.cpp** - Basic scaled dot-product attention
- **rope.cpp** - Rotary Position Embeddings (RoPE)

### Positional Encoding
- **positional_encoding.h** - Sinusoidal positional encoding (LibTorch)
- **simple_positional_encoding.cpp** - Standalone implementation

### Encoder & Decoder
- **encoder.cpp/h** - EncoderBlock with self-attention, FFN, layer norm
- **decoder.cpp/h** - DecoderBlock with masked self-attention, cross-attention
- **transformer.cpp** - Complete encoder-decoder architecture

---

## Distributed Training

### DDP (Distributed Data Parallel)
**Files:** `ddp_training.cpp`, `ddp_transformer.cpp`

**Key Concepts:**
- Each GPU maintains full model copy
- Data is partitioned across GPUs
- Gradients are averaged via all-reduce

**Communication Pattern:**
```
Forward:  Independent (no communication)
Backward: All-Reduce gradients → Average → Update
```

**Operations:**
1. **Broadcast Parameters** - Sync initial weights from rank 0
2. **All-Reduce Gradients** - Sum gradients, divide by world_size
3. **Optimizer Step** - All ranks update identically

**Use Case:** Models that fit in single GPU memory

---

### FSDP (Fully Sharded Data Parallel)
**Files:** `fsdp_training.cpp`, `fsdp_transformer.cpp`

**Key Concepts:**
- Parameters sharded across GPUs (each GPU stores 1/N)
- Reduces memory by N times
- All-gather before forward/backward, reduce-scatter after

**Communication Pattern:**
```
Forward:  All-Gather params → Compute → Free params
Backward: All-Gather params → Compute grads → Reduce-Scatter grads
```

**Operations:**
1. **Shard Parameters** - Split model across ranks
2. **All-Gather** - Temporarily reconstruct full model
3. **Reduce-Scatter** - Aggregate and distribute gradient shards

**Use Case:** Large models that don't fit in single GPU

---

### DDP vs FSDP Comparison

| Aspect | DDP | FSDP |
|--------|-----|------|
| Memory | Full model per GPU | Sharded (1/N per GPU) |
| Communication | All-reduce gradients | All-gather + reduce-scatter |
| Speed | Faster (less comm) | Slower (more comm) |
| Model Size | Limited by GPU memory | Can train huge models |

---

### NCCL (NVIDIA Collective Communications Library)

**Collective Operations:**
- **Broadcast** - One-to-all (rank 0 → all ranks)
- **All-Reduce** - All-to-all reduction (sum/avg gradients)
- **All-Gather** - Collect shards from all ranks
- **Reduce-Scatter** - Reduce and distribute results

**Used in:**
- DDP: All-reduce for gradient synchronization
- FSDP: All-gather (forward) + reduce-scatter (backward)

---

## Model Compression

### Knowledge Distillation
**File:** `knowledge_distillation.cpp`

**Concept:** Transfer knowledge from large teacher to small student

**Key Components:**
- Temperature-scaled softmax for soft targets
- KL divergence loss between teacher/student
- Combined loss: α×soft_loss + (1-α)×hard_loss

**Formula:**
```
soft_loss = KL(student_logits/T, teacher_logits/T) × T²
total_loss = α×soft_loss + (1-α)×CE(student, labels)
```

---

### Pruning
**File:** `pruning.cpp`

**Types:**
1. **Unstructured** - Remove individual weights (creates sparse matrix)
2. **Structured** - Remove entire filters/neurons (maintains dense structure)

**Method:** Magnitude-based (remove smallest weights by L1 norm)

**Iterative Pruning:**
- Gradually increase pruning ratio
- Fine-tune after each pruning step
- Achieves better accuracy than one-shot pruning

---

### Quantization
**File:** `quantization.cpp`

**Types:**
1. **K-means** - Cluster weights into k centroids (codebook)
2. **Uniform** - Map to fixed-point integers (2^n levels)

**Compression:**
- 32-bit float → 4-bit (16 clusters) = 8× compression
- Store: codebook + indices instead of full weights

---

### Low-Rank Approximation
**File:** `low_rank_approximation.cpp`

**SVD Decomposition:**
```
W (m×n) ≈ U (m×r) × S (r×r) × V^T (r×n)
```
- Replace Linear(m,n) with Linear(m,r) → Linear(r,n)
- Parameters: m×n → m×r + r×n (compression when r << min(m,n))

**Tucker Decomposition:**
- For Conv2d tensors (4D)
- Decomposes along multiple modes

---

## Training Utilities

### Training Utils
**File:** `training_utils.cpp`

**Functions:**
- `train_step()` - Forward, backward, optimize
- `eval_step()` - Validation without gradients
- `train_epoch()` - Full epoch with timing
- `validate()` - Compute loss and accuracy
- `LRScheduler` - Reduce learning rate on plateau

---

### Data Loader
**File:** `data_loader.cpp`

**Components:**
- `SimpleDataset` - Store data and labels
- `DataLoader` - Batch creation with shuffling
- `DistributedSampler` - Partition data across GPUs (in DDP/FSDP files)

---

## Training Examples

### BERT Training
**File:** `bert_train.cpp`

**Tasks:**
1. Masked Language Modeling (MLM) - Predict masked tokens
2. Next Sentence Prediction (NSP) - Binary classification

**Special Tokens:** [CLS], [SEP], [MASK], [PAD]

---

## Build Instructions

### Prerequisites
- LibTorch (PyTorch C++ API)
- CUDA Toolkit 11.0+
- NCCL library
- CMake 3.18+
- C++17 compiler (g++/clang++)

### Install LibTorch

**Option 1: Download Pre-built**
```bash
# CUDA 11.8
wget https://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcu118.zip
unzip libtorch-cxx11-abi-shared-with-deps-2.1.0+cu118.zip
export LIBTORCH_PATH=/path/to/libtorch
export LD_LIBRARY_PATH=$LIBTORCH_PATH/lib:$LD_LIBRARY_PATH
```

**Option 2: Use Python Installation**
```bash
pip install torch torchvision torchaudio
```

### Build All Targets

```bash
mkdir build && cd build

# With downloaded LibTorch
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..

# With Python's torch
cmake -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'` ..

make -j$(nproc)
```

### Build Specific Targets

```bash
# Core architecture
make transformer
make attention
make multihead_attention

# Distributed training
make ddp_training
make fsdp_training
make ddp_transformer
make fsdp_transformer

# Model compression
make knowledge_distillation
make pruning
make quantization
make low_rank_approximation

# Training examples
make bert_train
make train_encoder
make train_decoder
```

### Run Examples

**Single GPU:**
```bash
./transformer
./bert_train
./pruning
./quantization
```

**Multi-GPU (2 GPUs):**
```bash
# DDP
./ddp_training 0 2 &
./ddp_training 1 2 &

# FSDP
./fsdp_training 0 2 &
./fsdp_training 1 2 &
```

---

## File Summary

**Core Architecture (8 files):**
- attention.cpp, multihead_attention.cpp, rope.cpp
- positional_encoding.h, simple_positional_encoding.cpp
- encoder.cpp/h, decoder.cpp/h, transformer.cpp

**Distributed Training (4 files):**
- ddp_training.cpp, ddp_transformer.cpp
- fsdp_training.cpp, fsdp_transformer.cpp

**Model Compression (4 files):**
- knowledge_distillation.cpp
- pruning.cpp
- quantization.cpp
- low_rank_approximation.cpp

**Utilities (2 files):**
- training_utils.cpp
- data_loader.cpp

**Training Examples (4 files):**
- bert_train.cpp, train_encoder.cpp
- train_decoder.cpp, train_encoder_decoder.cpp

---

## Key Takeaways

1. **DDP** - Fast, full model replication, limited by GPU memory
2. **FSDP** - Memory efficient, sharded model, more communication
3. **Compression** - Distillation, pruning, quantization, low-rank for deployment
4. **NCCL** - Efficient GPU-to-GPU communication primitives
5. **Transformer** - Encoder-decoder with attention, positional encoding, FFN
