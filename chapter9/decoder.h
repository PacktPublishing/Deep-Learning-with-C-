#pragma once
#include <torch/torch.h>

/**
 * @brief Single decoder layer with masked self-attention, cross-attention, and feedforward network
 */
struct DecoderBlock : torch::nn::Module {
    torch::nn::MultiheadAttention self_attn{nullptr}, cross_attn{nullptr};
    torch::nn::LayerNorm norm1{nullptr}, norm2{nullptr}, norm3{nullptr};
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    
    DecoderBlock(int d_model, int nhead, int d_ff);
    torch::Tensor forward(torch::Tensor x, torch::Tensor enc_out, torch::Tensor mask);
};

/**
 * @brief Transformer decoder with stacked decoder layers
 */
struct Decoder : torch::nn::Module {
    torch::nn::ModuleList layers{nullptr};
    
    Decoder(int num_layers, int d_model, int nhead, int d_ff);
    torch::Tensor forward(torch::Tensor x, torch::Tensor enc_out, torch::Tensor mask);
};
