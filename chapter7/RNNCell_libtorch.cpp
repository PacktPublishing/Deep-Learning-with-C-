#include <torch/torch.h>
#include <iostream>

class RNNCell : public torch::nn::Module {
private:
    int input_size, hidden_size, output_size;
    torch::Tensor W_xh, W_hh, W_hy, b_h, b_y;

public:
    RNNCell(int input_sz, int hidden_sz, int output_sz)
        : input_size(input_sz), hidden_size(hidden_sz), output_size(output_sz) {

        W_xh = register_parameter("W_xh", torch::randn({hidden_sz, input_sz}) * 0.1);
        W_hh = register_parameter("W_hh", torch::randn({hidden_sz, hidden_sz}) * 0.1);
        W_hy = register_parameter("W_hy", torch::randn({output_sz, hidden_sz}) * 0.1);
        b_h  = register_parameter("b_h", torch::zeros({hidden_sz}));
        b_y  = register_parameter("b_y", torch::zeros({output_sz}));
    }

    // h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)
    // y_t = sigmoid(W_hy * h_t + b_y)
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x, const torch::Tensor& h_prev) {
        torch::Tensor h = torch::tanh(torch::mv(W_xh, x) + torch::mv(W_hh, h_prev) + b_h);
        torch::Tensor y = torch::sigmoid(torch::mv(W_hy, h) + b_y);
        return {h, y};
    }

    std::pair<std::vector<torch::Tensor>, std::vector<torch::Tensor>>
    forward_sequence(const std::vector<torch::Tensor>& sequence) {
        std::vector<torch::Tensor> hidden_states, outputs;
        torch::Tensor h = torch::zeros({hidden_size}, W_xh.options());

        hidden_states.push_back(h);
        for (const auto& x : sequence) {
            auto [new_h, y] = forward(x, h);
            h = new_h;
            hidden_states.push_back(h);
            outputs.push_back(y);
        }
        return {hidden_states, outputs};
    }
};

int main() {
    auto device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << "Using device: " << (device == torch::kCUDA ? "CUDA" : "CPU") << std::endl;

    RNNCell rnn(2, 3, 1);
    rnn.to(device);

    std::vector<torch::Tensor> sequence = {
        torch::tensor({0.5f, -0.3f}).to(device),
        torch::tensor({0.2f,  0.8f}).to(device),
        torch::tensor({-0.1f, 0.4f}).to(device)
    };

    auto [hidden_states, outputs] = rnn.forward_sequence(sequence);

    std::cout << "Sequence processing results:" << std::endl;
    for (size_t t = 0; t < outputs.size(); ++t) {
        std::cout << "Time " << t
                  << ": Input=" << sequence[t]
                  << " Output=" << outputs[t] << std::endl;
    }

    return 0;
}
