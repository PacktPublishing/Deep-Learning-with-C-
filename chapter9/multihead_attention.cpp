#include <torch/torch.h>
#include <iostream>

class MultiHeadAttention {
public:
    // Scaled Dot-Product Attention
    static torch::Tensor scaledDotProductAttention(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        const torch::Tensor& V
    ) {
        auto d_k = K.size(-1);
        auto scale = 1.0 / std::sqrt(d_k);
        
        auto scores = torch::matmul(Q, K.transpose(-2, -1)) * scale;
        auto attention_weights = torch::softmax(scores, -1);
        return torch::matmul(attention_weights, V);
    }
    
    // Multi-Head Attention with Linear Transformations
    static torch::Tensor forward(
        const torch::Tensor& input,  // [seq_len, d_model]
        int d_model,
        int num_heads
    ) {
        auto seq_len = input.size(0);
        auto d_k = d_model / num_heads;
        
        // Linear transformation weights (W_q, W_k, W_v, W_o)
        auto W_q = torch::randn({d_model, d_model});
        auto W_k = torch::randn({d_model, d_model});
        auto W_v = torch::randn({d_model, d_model});
        auto W_o = torch::randn({d_model, d_model});
        
        // Linear projections: Q = XW_q, K = XW_k, V = XW_v
        auto Q = torch::matmul(input, W_q);
        auto K = torch::matmul(input, W_k);
        auto V = torch::matmul(input, W_v);
        
        // Reshape to multi-head: [seq_len, num_heads, d_k]
        Q = Q.view({seq_len, num_heads, d_k});
        K = K.view({seq_len, num_heads, d_k});
        V = V.view({seq_len, num_heads, d_k});
        
        // Apply attention to each head
        std::vector<torch::Tensor> head_outputs;
        for (int h = 0; h < num_heads; h++) {
            auto head_out = scaledDotProductAttention(Q.select(1, h), K.select(1, h), V.select(1, h));
            head_outputs.push_back(head_out);
        }
        
        // Concatenate heads and apply output projection
        auto concat = torch::cat(head_outputs, -1);
        return torch::matmul(concat, W_o);
    }
};

int main() {
    int seq_len = 4;
    int d_model = 8;
    int num_heads = 2;
    
    auto input = torch::randn({seq_len, d_model});
    auto output = MultiHeadAttention::forward(input, d_model, num_heads);
    
    std::cout << "Multi-Head Attention with Linear Transformations:" << std::endl;
    std::cout << "Input shape: " << input.sizes() << std::endl;
    std::cout << "Output shape: " << output.sizes() << std::endl;
    std::cout << "Heads: " << num_heads << ", d_k: " << d_model/num_heads << std::endl;
    
    return 0;
}