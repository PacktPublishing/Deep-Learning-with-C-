# Chapter 7: Recurrent Neural Networks in C++ - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [RNN Implementations](#rnn-implementations)
3. [LSTM Architecture](#lstm-architecture)
4. [Text Processing & Applications](#text-processing--applications)
5. [Compilation & Setup](#compilation--setup)
6. [Performance Comparison](#performance-comparison)

---

## Overview

This chapter demonstrates the progression from basic RNN implementations to production-ready LSTM networks in C++, covering:

- **Vanilla RNN** with Backpropagation Through Time (BPTT)
- **LSTM Networks** with gating mechanisms
- **Text Processing** pipelines and word embeddings
- **Applications**: Text prediction and neural machine translation
- **Optimization Levels**: Manual loops → std::valarray → Eigen → LibTorch → CUDA

---

## RNN Implementations

### Architecture: Vanilla RNN (Elman RNN)

**Forward Pass:**
```
h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
y_t = W_hy * h_t + b_y
p_t = σ(y_t)
```

**Backward Pass (BPTT):**
```
∂L/∂y_t = p_t - target_t
∂L/∂W_hy = ∂L/∂y_t * h_t^T
∂L/∂h_t = W_hy^T * ∂L/∂y_t + ∂L/∂h_{t+1}
∂L/∂z_t = ∂L/∂h_t ⊙ tanh'(z_t)
∂L/∂W_xh = ∂L/∂z_t * x_t^T
∂L/∂W_hh = ∂L/∂z_t * h_{t-1}^T
```

### Implementation Variants

| File | Library | Performance | Complexity | GPU | Use Case |
|------|---------|-------------|------------|-----|----------|
| `RNNCell.cpp` | Manual | 1x | Simple | No | Learning basics |
| `RNNCell_simple.cpp` | Manual | 1x | Simple | No | Minimal example |
| `rnn_bptt.cpp` | Manual loops | 1x | Simple | No | Understanding BPTT |
| `rnn_bptt_matrix.cpp` | std::valarray | 3-5x | Medium | No | Standard library |
| `rnn_bptt_valarray.cpp` | Custom Matrix | 3-5x | Medium | No | Modular design |
| `rnn_bptt_eigen.cpp` | Eigen | 10-50x | Low | No | Production CPU |
| `rnn_bptt_libtorch.cpp` | LibTorch | 50-200x | Low | Yes | Production GPU |
| `rnn_bptt_gpu.cu` | CUDA/cuBLAS | 10-100x | High | Yes | Custom GPU |

### File Descriptions

**`RNNCell.cpp` / `RNNCell_simple.cpp`**
- Basic RNN cell implementation with weight matrices
- Element-wise operations for clarity
- Educational focus on forward pass mechanics

**`rnn_bptt.cpp`**
- Complete BPTT implementation with nested loops
- Clear mathematical correspondence to equations
- Best for understanding gradient flow

**`rnn_bptt_matrix.cpp`**
- Uses `std::valarray` for vectorization
- Built-in math functions (tanh, exp)
- No external dependencies

**`rnn_bptt_valarray.cpp`**
- Modular design with `matrix_ops.h`
- Custom Matrix class with operator overloading
- Reusable linear algebra library

**`rnn_bptt_eigen.cpp`**
- High-performance Eigen library
- SIMD vectorization and cache optimization
- Expression templates for memory efficiency

**`rnn_bptt_libtorch.cpp`**
- PyTorch C++ API with automatic differentiation
- GPU support and production optimizations
- Framework-level abstractions

**`rnn_bptt_gpu.cu`**
- Custom CUDA kernels with cuBLAS
- Explicit GPU memory management
- Maximum control over parallelization

### Training Data

XOR-like sequence problem used across all implementations:
```
Input: {{1,0}, {0,1}, {1,1}} → Target: {1, 1, 0}
Input: {{0,1}, {1,0}, {0,0}} → Target: {1, 1, 0}
Input: {{1,1}, {0,0}, {1,0}} → Target: {0, 0, 1}
```

### Limitations of Vanilla RNN

- **Vanishing gradients**: Poor performance on long sequences (>10 steps)
- **No gating mechanisms**: Cannot selectively forget information
- **Sequential processing**: Limited parallelization across time steps
- **Short-term memory**: Struggles with long-term dependencies

**Solution**: Use LSTM or GRU architectures for practical applications.

---

## LSTM Architecture

### Mathematical Foundation

LSTM solves vanishing gradients through four gating mechanisms:

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)    // Forget gate
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)    // Input gate  
C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C) // Candidate values
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)    // Output gate

C_t = f_t * C_{t-1} + i_t * C̃_t       // Update cell state
h_t = o_t * tanh(C_t)                  // Update hidden state
```

### Implementation Files

| File | Approach | Features |
|------|----------|----------|
| `lstm_simple.cpp` | Vector-based | Separate gate matrices, educational |
| `lstm_matrix.cpp` | Eigen-based | Combined matrices, optimized |
| `lstm_libtorch.cpp` | LibTorch | Multi-layer, dropout, GPU support |

### Core LSTM Cell Design

```cpp
class EigenLSTMCell {
private:
    // Combined weight matrices for all four gates
    Eigen::MatrixXf W_combined;  // [4*hidden_size, input_size]
    Eigen::MatrixXf U_combined;  // [4*hidden_size, hidden_size]
    Eigen::VectorXf b_combined;  // [4*hidden_size]
    
public:
    std::pair<Eigen::VectorXf, Eigen::VectorXf> forward(
        const Eigen::VectorXf& input,
        const Eigen::VectorXf& prev_hidden,
        const Eigen::VectorXf& prev_cell) {
        
        // Single matrix multiplication for all gates
        Eigen::VectorXf gates = W_combined * input + 
                               U_combined * prev_hidden + 
                               b_combined;
        
        // Extract and activate gates
        Eigen::VectorXf forget_gate = sigmoid(gates.segment(0, hidden_size));
        Eigen::VectorXf input_gate = sigmoid(gates.segment(hidden_size, hidden_size));
        Eigen::VectorXf candidate = tanh(gates.segment(2*hidden_size, hidden_size));
        Eigen::VectorXf output_gate = sigmoid(gates.segment(3*hidden_size, hidden_size));
        
        // Update states
        Eigen::VectorXf new_cell = forget_gate.cwiseProduct(prev_cell) + 
                                  input_gate.cwiseProduct(candidate);
        Eigen::VectorXf new_hidden = output_gate.cwiseProduct(tanh(new_cell));
        
        return {new_hidden, new_cell};
    }
};
```

### Key Design Decisions

**Combined Weight Matrices**
- Reduces memory access patterns
- Enables vectorization
- Single matrix multiplication for all gates

**Xavier Initialization**
```cpp
W_combined = Eigen::MatrixXf::Random(4*hidden_size, input_size) * 
            std::sqrt(2.0f / (input_size + hidden_size));
```
- Prevents gradient explosion/vanishing
- Maintains gradient flow during training

**Forget Gate Bias = 1.0**
```cpp
b_combined.segment(0, hidden_size).setOnes();
```
- Allows information to flow initially
- Prevents early gradient death

**Gradient Clipping**
```cpp
forget_gate = forget_gate.unaryExpr([](float x) { 
    return 1.0f / (1.0f + std::exp(-std::max(-50.0f, std::min(50.0f, x)))); 
});
```
- Prevents numerical instability
- Essential for training stability

### LSTM vs Vanilla RNN

| Aspect | Vanilla RNN | LSTM |
|--------|-------------|------|
| Memory | Single hidden state | Hidden + cell states |
| Gradients | Vanishing problem | Controlled via gates |
| Long sequences | Poor (<10 steps) | Excellent (>100 steps) |
| Complexity | O(h²) per step | O(4h²) per step |
| Training stability | Unstable | Much more stable |
| Parameters | 3 weight matrices | 8 weight matrices |

---

## Text Processing & Applications

### Text Processing Pipeline

**Files:**
- `text_preprocessor.cpp` - Text cleaning and tokenization
- `text_utf8_handling.cpp` - UTF-8 character handling
- `word_embeddings.cpp` - Word2Vec implementation
- `word_embeddings_eigen.cpp` - Optimized Word2Vec

**Core Components:**

```cpp
class TextPredictor {
    // Vocabulary building
    void buildVocabulary(const std::vector<std::string>& corpus, int min_freq = 2) {
        // Count word frequencies
        // Filter by minimum frequency
        // Add special tokens: <START>, <END>, <UNK>
    }
    
    // One-hot encoding
    Matrix createOneHot(int word_id) {
        Matrix one_hot(vocab_size, 1);
        one_hot(word_id, 0) = 1.0f;
        return one_hot;
    }
    
    // Text preprocessing
    std::string preprocess(const std::string& text) {
        // Lowercase conversion
        // Punctuation removal
        // Tokenization
    }
};
```

### Word Embeddings (Word2Vec)

**Skip-gram Model:**
```cpp
class Word2Vec {
    // Input: center word → Output: context words
    torch::nn::Embedding input_embeddings{nullptr};
    torch::nn::Linear output_layer{nullptr};
    
    torch::Tensor forward(torch::Tensor center_words) {
        auto embedded = input_embeddings->forward(center_words);
        auto logits = output_layer->forward(embedded);
        return torch::log_softmax(logits, -1);
    }
};
```

### Applications

#### 1. Text Prediction

**Files:**
- `text_prediction.cpp` - Manual LSTM implementation
- `text_prediction_libtorch.cpp` - LibTorch version

**Architecture:**
```
Input Sequence → LSTM Layers → Output Layer → Softmax → Next Word
```

**Training Loop:**
```cpp
for (int epoch = 0; epoch < epochs; ++epoch) {
    for (const auto& sequence : sequences) {
        auto input_ids = torch::tensor(sequence).to(device);
        auto target_ids = torch::tensor(targets).to(device);
        
        optimizer.zero_grad();
        auto lstm_out = lstm->forward(one_hot(input_ids));
        auto logits = output_layer->forward(lstm_out);
        auto loss = cross_entropy(logits, target_ids);
        
        loss.backward();
        optimizer.step();
    }
}
```

#### 2. Neural Machine Translation

**Files:**
- `rnn_translator.cpp` - RNN-based seq2seq
- `lstm_translator.cpp` - LSTM-based seq2seq

**Encoder-Decoder Architecture:**
```
Source Sequence → Encoder LSTM → Context Vector → Decoder LSTM → Target Sequence
```

**Key Components:**
- **Encoder**: Processes source language, produces context vector
- **Decoder**: Generates target language from context
- **Attention** (optional): Focuses on relevant source words

---

## Compilation & Setup

### Prerequisites

```bash
# Eigen (for optimized implementations)
sudo apt-get install libeigen3-dev  # Ubuntu/Debian
brew install eigen                   # macOS

# LibTorch (for production implementations)
# Download from: https://pytorch.org/cppdocs/installing.html
wget https://download.pytorch.org/libtorch/cpu/libtorch-macos-2.1.0.zip
unzip libtorch-macos-2.1.0.zip

# CUDA Toolkit (for GPU implementations)
# Download from: https://developer.nvidia.com/cuda-downloads
```

### Compilation Commands

**Basic C++ (no dependencies):**
```bash
g++ -O3 -std=c++17 RNNCell.cpp -o rnn_cell
g++ -O3 -std=c++17 rnn_bptt.cpp -o rnn_bptt
g++ -O3 -std=c++17 lstm_simple.cpp -o lstm_simple
```

**With std::valarray:**
```bash
g++ -O3 -std=c++17 rnn_bptt_matrix.cpp -o rnn_matrix
g++ -O3 -std=c++17 rnn_bptt_valarray.cpp -o rnn_valarray
```

**With Eigen:**
```bash
g++ -O3 -std=c++17 -I/usr/include/eigen3 rnn_bptt_eigen.cpp -o rnn_eigen
g++ -O3 -std=c++17 -I/usr/include/eigen3 lstm_matrix.cpp -o lstm_matrix
g++ -O3 -std=c++17 -I/usr/include/eigen3 word_embeddings_eigen.cpp -o word_embeddings
```

**With LibTorch:**
```bash
g++ -O3 -std=c++17 rnn_bptt_libtorch.cpp \
    -I./libtorch/include \
    -L./libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,./libtorch/lib \
    -o rnn_libtorch

g++ -O3 -std=c++17 lstm_libtorch.cpp \
    -I./libtorch/include \
    -L./libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,./libtorch/lib \
    -o lstm_libtorch

g++ -O3 -std=c++17 text_prediction_libtorch.cpp \
    -I./libtorch/include \
    -L./libtorch/lib \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,./libtorch/lib \
    -o text_prediction
```

**With CUDA:**
```bash
nvcc -O3 -std=c++17 rnn_bptt_gpu.cu \
    -lcublas -lcudnn \
    -o rnn_gpu
```

### Running Examples

```bash
# Basic RNN
./rnn_cell
./rnn_bptt

# LSTM variants
./lstm_simple
./lstm_matrix
./lstm_libtorch

# Text applications
./text_prediction
./lstm_translator

# Word embeddings
./word_embeddings
```

---

## Performance Comparison

### Computational Performance

| Implementation | Relative Speed | Memory Usage | Compilation Time |
|---------------|----------------|--------------|------------------|
| Manual loops | 1x (baseline) | Minimal | Fast |
| std::valarray | 3-5x | Medium | Medium |
| Eigen | 10-50x | Optimized | Slow |
| LibTorch CPU | 50-100x | Medium | Slow |
| LibTorch GPU | 100-500x | High | Slow |
| CUDA Custom | 10-100x | High | Medium |

### Memory Characteristics

**Basic C++:**
- Manual arrays, minimal overhead
- Poor cache locality
- No vectorization

**std::valarray:**
- Vectorized storage
- Better cache utilization
- Standard library guarantees

**Eigen:**
- Column-major storage (cache-friendly)
- Expression templates (zero-copy)
- SIMD instructions

**LibTorch:**
- Automatic memory management
- GPU memory pooling
- Tensor operations optimized

### When to Use Each Implementation

**Learning & Education:**
- `RNNCell.cpp`, `rnn_bptt.cpp` - Understanding fundamentals
- `lstm_simple.cpp` - LSTM gate mechanics

**Prototyping:**
- `rnn_bptt_matrix.cpp` - Quick experiments, no dependencies
- `rnn_bptt_eigen.cpp` - Fast CPU prototyping

**Production:**
- `rnn_bptt_libtorch.cpp` - Automatic differentiation, GPU support
- `lstm_libtorch.cpp` - Multi-layer, dropout, production features
- `rnn_bptt_gpu.cu` - Custom GPU kernels for specialized needs

**Research:**
- Eigen versions - Custom algorithms, CPU-focused
- LibTorch versions - Rapid experimentation, GPU acceleration

### Best Practices

**Initialization:**
- Use Xavier/Glorot for weights
- Set forget gate bias to 1.0 in LSTM
- Zero initialization for biases (except forget gate)

**Training:**
- Gradient clipping (threshold: 1.0-5.0)
- Learning rate: 0.001-0.01 (Adam optimizer)
- Batch size: 32-128 for text tasks
- Sequence length: 20-50 for LSTM

**Architecture:**
- Hidden size: 64-512 (start with 128)
- Layers: 1-3 for most tasks
- Dropout: 0.2-0.5 between layers
- Embedding size: 100-300 for word embeddings

**Regularization:**
- Dropout between LSTM layers
- Weight decay (L2): 1e-5 to 1e-4
- Early stopping based on validation loss

---

## Supporting Files

**`matrix_ops.h`** - Custom matrix operations library
- Matrix class with std::valarray backend
- Operator overloading for mathematical notation
- Vectorized element-wise operations

**`core_matrix_operations.cpp`** - Core matrix utilities
- Basic linear algebra operations
- Helper functions for matrix manipulation

**`sample_text.txt`** - Sample training data
- Text corpus for training examples
- Used in text prediction demonstrations

---

## Summary

This chapter provides a complete progression from basic RNN concepts to production-ready LSTM implementations:

1. **Educational Path**: Manual → std::valarray → Eigen
2. **Production Path**: LibTorch (CPU/GPU) or Custom CUDA
3. **Applications**: Text prediction, machine translation, word embeddings

**Key Takeaways:**
- Vanilla RNNs suffer from vanishing gradients
- LSTMs solve this through gating mechanisms
- Modern frameworks (LibTorch) provide production-ready solutions
- Understanding fundamentals helps debug and optimize models

**Next Steps:**
- Experiment with different architectures (GRU, Transformer)
- Implement attention mechanisms
- Scale to larger datasets
- Deploy models in production environments
