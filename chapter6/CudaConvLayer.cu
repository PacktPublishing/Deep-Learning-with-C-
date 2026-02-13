#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <vector>
using namespace std;

__global__ void im2col_relu_kernel(const float* input, float* output, 
                                  int inputRows, int inputCols, int kernelSize, 
                                  int stride, int outputRows, int outputCols) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalOutputs = outputRows * outputCols;
    
    if (idx < totalOutputs) {
        int outRow = idx / outputCols, outCol = idx % outputCols;
        int inputStartRow = outRow * stride, inputStartCol = outCol * stride;
        
        for (int ki = 0; ki < kernelSize; ki++) {
            for (int kj = 0; kj < kernelSize; kj++) {
                output[(ki * kernelSize + kj) * totalOutputs + idx] = 
                    input[(inputStartRow + ki) * inputCols + (inputStartCol + kj)];
            }
        }
    }
}

__global__ void relu_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

class CudaConvolutionalLayer {
    float *d_filters, *d_input, *d_im2col_output, *d_conv_output;
    int inputChannels, outputChannels, kernelSize, stride;
    cublasHandle_t cublasHandle;
    
public:
    CudaConvolutionalLayer(int inChannels, int outChannels, int kSize, int strideSize) 
        : inputChannels(inChannels), outputChannels(outChannels), kernelSize(kSize), stride(strideSize) {
        
        cublasCreate(&cublasHandle);
        int filterSize = outChannels * inChannels * kernelSize * kernelSize;
        cudaMalloc(&d_filters, filterSize * sizeof(float));
        
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
        curandGenerateUniform(gen, d_filters, filterSize);
        
        float alpha = 0.2f, beta = -0.1f;
        cublasSscal(cublasHandle, filterSize, &alpha, d_filters, 1);
        cublasSaxpy(cublasHandle, filterSize, &beta, d_filters, 1, d_filters, 1);
        curandDestroyGenerator(gen);
    }
    
    ~CudaConvolutionalLayer() {
        cudaFree(d_filters); cudaFree(d_input); cudaFree(d_im2col_output); cudaFree(d_conv_output);
        cublasDestroy(cublasHandle);
    }
    
    vector<float> forward(const vector<float>& input, int inputRows, int inputCols) {
        int outputRows = (inputRows - kernelSize) / stride + 1;
        int outputCols = (inputCols - kernelSize) / stride + 1;
        int totalOutputs = outputRows * outputCols;
        int outputSize = outputChannels * totalOutputs;
        
        cudaMalloc(&d_input, input.size() * sizeof(float));
        cudaMalloc(&d_im2col_output, kernelSize * kernelSize * totalOutputs * sizeof(float));
        cudaMalloc(&d_conv_output, outputSize * sizeof(float));
        
        cudaMemcpy(d_input, input.data(), input.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        int blockSize = 256, gridSize = (totalOutputs + blockSize - 1) / blockSize;
        im2col_relu_kernel<<<gridSize, blockSize>>>(d_input, d_im2col_output, inputRows, inputCols, kernelSize, stride, outputRows, outputCols);
        
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, totalOutputs, outputChannels, kernelSize * kernelSize,
                   &alpha, d_im2col_output, totalOutputs, d_filters, kernelSize * kernelSize, &beta, d_conv_output, totalOutputs);
        
        gridSize = (outputSize + blockSize - 1) / blockSize;
        relu_kernel<<<gridSize, blockSize>>>(d_conv_output, outputSize);
        
        vector<float> result(outputSize);
        cudaMemcpy(result.data(), d_conv_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
        return result;
    }
};

int main() {
    CudaConvolutionalLayer conv(1, 64, 3, 1);
    
    vector<float> input(64);
    for(int i = 0; i < 64; i++) input[i] = i + 1;
    
    vector<float> output = conv.forward(input, 8, 8);
    
    cout << "Input: 8x8\nOutput: 64x36\nCUDA convolution completed successfully!" << endl;
    return 0;
}