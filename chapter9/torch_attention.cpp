#include <torch/torch.h>
#include <iostream>

class TorchAttention {
public:
    // Scaled Dot-Product Attention using LibTorch
    static torch::Tensor scaledDotProductAttention(
        const torch::Tensor& Q,  // queries [seq_len, d_k]
        const torch::Tensor& K,  // keys [seq_len, d_k]  
        const torch::Tensor& V   // values [seq_len, d_v]
    ) {
        auto d_k = K.size(1);
        auto scale = 1.0 / std::sqrt(d_k);
        
        // Attention(Q,K,V) = softmax(QK^T/√d_k)V
        auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale;
        auto attention_weights = torch::softmax(scores, -1);
        return torch::matmul(attention_weights, V);
    }
    
    // Multi-Head Attention with Linear Transformations
    static torch::Tensor multiHeadAttention(
        const torch::Tensor& input,  // input [seq_len, d_model]
        int d_model,
        int num_heads
    ) {
        auto seq_len = input.size(0);
        auto d_k = d_model / num_heads;
        
        // Linear transformation weights
        auto W_q = torch::randn({d_model, d_model});
        auto W_k = torch::randn({d_model, d_model});
        auto W_v = torch::randn({d_model, d_model});
        auto W_o = torch::randn({d_model, d_model});
        
        // Apply linear transformations
        auto Q = torch::matmul(input, W_q);
        auto K = torch::matmul(input, W_k);
        auto V = torch::matmul(input, W_v);
        
        // Reshape to [seq_len, num_heads, d_k]
        auto Q_heads = Q.view({seq_len, num_heads, d_k}).transpose(0, 1);  // [num_heads, seq_len, d_k]
        auto K_heads = K.view({seq_len, num_heads, d_k}).transpose(0, 1);
        auto V_heads = V.view({seq_len, num_heads, d_k}).transpose(0, 1);
        
        // Apply attention to each head
        std::vector<torch::Tensor> head_outputs;
        for (int i = 0; i < num_heads; i++) {
            auto head_out = scaledDotProductAttention(Q_heads[i], K_heads[i], V_heads[i]);
            head_outputs.push_back(head_out);
        }
        
        // Concatenate heads and apply output projection
        auto concat = torch::cat(head_outputs, -1);
        return torch::matmul(concat, W_o);
    }
};

int main() {
    // Example usage - Single Head Attention
    int seq_len = 4;
    int d_k = 5;
    int d_v = 5;
    
    auto Q = torch::randn({seq_len, d_k});
    auto K = torch::randn({seq_len, d_k});
    auto V = torch::randn({seq_len, d_v});
    
    auto output = TorchAttention::scaledDotProductAttention(Q, K, V);
    std::cout << "Single Head Attention:" << std::endl;
    std::cout << "Input shapes: Q" << Q.sizes() << ", K" << K.sizes() << ", V" << V.sizes() << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;
    
    // Multi-Head Attention Example
    int d_model = 8;  // Must be divisible by num_heads
    int num_heads = 2;
    
    auto Q_multi = torch::randn({seq_len, d_model});
    auto K_multi = torch::randn({seq_len, d_model});
    auto V_multi = torch::randn({seq_len, d_model});
    
    auto multi_output = TorchAttention::multiHeadAttention(Q_multi, K_multi, V_multi, num_heads);
    std::cout << "\nMulti-Head Attention:" << std::endl;
    std::cout << "Input shapes: Q" << Q_multi.sizes() << ", K" << K_multi.sizes() << ", V" << V_multi.sizes() << std::endl;
    std::cout << "Output shape: " << multi_output.sizes() << std::endl;
    std::cout << "Number of heads: " << num_heads << std::endl;
    
    return 0;
}