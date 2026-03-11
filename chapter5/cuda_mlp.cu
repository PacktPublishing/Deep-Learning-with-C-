#include <cuda_runtime.h>
#include <curand.h>
#include <cublas_v2.h>
#include <iostream>
#include <vector>

// CUDA kernel for ReLU activation
__global__ void reluKernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// CUDA kernel for ReLU derivative
__global__ void reluDerivativeKernel(float* data, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = (data[idx] > 0) ? 1.0f : 0.0f;
    }
}

class CudaLayer {
private:
    int input_size, output_size;
    float *d_weights, *d_biases;
    float *d_input, *d_output;
    float *d_activation;
    cublasHandle_t handle;

public:
    CudaLayer(int in_size, int out_size) : input_size(in_size), output_size(out_size) {
        cublasCreate(&handle);
        
        // Allocate memory on GPU
        cudaMalloc(&d_weights, output_size * input_size * sizeof(float));
        cudaMalloc(&d_biases, output_size * sizeof(float));
        cudaMalloc(&d_input, input_size * sizeof(float));
        cudaMalloc(&d_output, output_size * sizeof(float));
        cudaMalloc(&d_activation, output_size * sizeof(float));

        // Initialize weights randomly
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandGenerateUniform(gen, d_weights, output_size * input_size);
        curandDestroyGenerator(gen);
        
        // Initialize biases to zero
        cudaMemset(d_biases, 0, output_size * sizeof(float));
    }

    // Prevent copying
    CudaLayer(const CudaLayer&) = delete;
    CudaLayer& operator=(const CudaLayer&) = delete;

    // Allow moving
    CudaLayer(CudaLayer&& other) noexcept 
        : input_size(other.input_size), output_size(other.output_size),
          d_weights(other.d_weights), d_biases(other.d_biases),
          d_input(other.d_input), d_output(other.d_output),
          d_activation(other.d_activation), handle(other.handle) {
        other.d_weights = nullptr;
        other.d_biases = nullptr;
        other.d_input = nullptr;
        other.d_output = nullptr;
        other.d_activation = nullptr;
        other.handle = nullptr;
    }

    void forward(float* input, int batch_size) {
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Matrix multiplication: activation = weights * input + biases
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    output_size, batch_size, input_size,
                    &alpha,
                    d_weights, output_size,
                    input, input_size,
                    &beta,
                    d_activation, output_size);

        // Add biases
        for (int i = 0; i < batch_size; ++i) {
            cublasSaxpy(handle, output_size, &alpha, 
                       d_biases, 1, 
                       d_activation + i * output_size, 1);
        }

        // Apply ReLU
        int block_size = 256;
        int num_blocks = (output_size * batch_size + block_size - 1) / block_size;
        reluKernel<<<num_blocks, block_size>>>(d_activation, output_size * batch_size);
    }

    void backward(float* grad_output, float learning_rate, int batch_size) {
        const float alpha = learning_rate;
        const float beta = 1.0f;

        // Compute gradient through ReLU
        int block_size = 256;
        int num_blocks = (output_size * batch_size + block_size - 1) / block_size;
        float* d_grad_activation;
        cudaMalloc(&d_grad_activation, output_size * batch_size * sizeof(float));
        reluDerivativeKernel<<<num_blocks, block_size>>>(d_activation, d_grad_activation, 
                                                        output_size * batch_size);

        // Update weights
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_T,
                    output_size, input_size, batch_size,
                    &alpha,
                    d_grad_activation, output_size,
                    d_input, input_size,
                    &beta,
                    d_weights, output_size);

        // Update biases
        cublasSgemv(handle, CUBLAS_OP_N,
                    output_size, batch_size,
                    &alpha,
                    d_grad_activation, output_size,
                    d_input, 1,
                    &beta,
                    d_biases, 1);

        cudaFree(d_grad_activation);
    }

    ~CudaLayer() {
        if (d_weights) cudaFree(d_weights);
        if (d_biases) cudaFree(d_biases);
        if (d_input) cudaFree(d_input);
        if (d_output) cudaFree(d_output);
        if (d_activation) cudaFree(d_activation);
        if (handle) cublasDestroy(handle);
    }
};

class CudaMLP {
private:
    std::vector<CudaLayer> layers;
    float learning_rate;

public:
    CudaMLP(const std::vector<int>& layer_sizes, float lr = 0.01) 
        : learning_rate(lr) {
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            layers.emplace_back(layer_sizes[i], layer_sizes[i + 1]);
        }
    }

    void train(float* X, float* y, int batch_size, int epochs) {
        float *d_prediction, *d_grad;
        cudaMalloc(&d_prediction, batch_size * sizeof(float));
        cudaMalloc(&d_grad, batch_size * sizeof(float));
        
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        for (int epoch = 0; epoch < epochs; ++epoch) {
            // Forward pass
            float* current_input = X;
            for (auto& layer : layers) {
                layer.forward(current_input, batch_size);
                // Note: Need to update current_input to layer output in real implementation
            }
            
            // Compute loss gradient: grad = 2 * (prediction - y) / batch_size
            const float alpha = 2.0f / batch_size;
            const float neg_alpha = -alpha;
            cudaMemcpy(d_grad, d_prediction, batch_size * sizeof(float), cudaMemcpyDeviceToDevice);
            cublasSaxpy(handle, batch_size, &neg_alpha, y, 1, d_grad, 1);
            
            // Backward pass with proper gradient flow
            float* current_grad = d_grad;
            for (int i = layers.size() - 1; i >= 0; --i) {
                layers[i].backward(current_grad, learning_rate, batch_size);
                // Note: Need to update current_grad to layer's gradient output in real implementation
            }
        }
        
        cudaFree(d_prediction);
        cudaFree(d_grad);
        cublasDestroy(handle);
    }
};

int main() {
    // XOR problem setup
    const int input_size = 2;
    const int batch_size = 4;
    
    float h_X[8] = {0, 0, 0, 1, 1, 0, 1, 1};  // Input data
    float h_y[4] = {0, 1, 1, 0};              // Target output
    
    float *d_X, *d_y;
    cudaMalloc(&d_X, sizeof(float) * input_size * batch_size);
    cudaMalloc(&d_y, sizeof(float) * batch_size);
    
    cudaMemcpy(d_X, h_X, sizeof(float) * input_size * batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, sizeof(float) * batch_size, cudaMemcpyHostToDevice);

    std::vector<int> layer_sizes = {2, 4, 3, 1};
    CudaMLP mlp(layer_sizes, 0.1f);
    
    std::cout << "Training CUDA MLP..." << std::endl;
    mlp.train(d_X, d_y, batch_size, 1000);
    std::cout << "Training completed" << std::endl;

    cudaFree(d_X);
    cudaFree(d_y);
    
    return 0;
}
