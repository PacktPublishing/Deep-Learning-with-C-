#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <Eigen/Dense>
#include <iostream>

using namespace Eigen;
using namespace std;

class CudaEigenConvLayer {
private:
    float* d_filters;
    float* d_patches;
    float* d_output;
    cublasHandle_t handle;
    int inputChannels, outputChannels, kernelSize, stride;
    
public:
    CudaEigenConvLayer(int inCh, int outCh, int kSize, int str) 
        : inputChannels(inCh), outputChannels(outCh), kernelSize(kSize), stride(str) {
        cublasCreate(&handle);
        
        // Initialize filters on CPU with Eigen, then copy to GPU
        MatrixXf filters_cpu = MatrixXf::Random(outCh, inCh * kSize * kSize) * 0.2f - 0.1f;
        cudaMalloc(&d_filters, filters_cpu.size() * sizeof(float));
        cudaMemcpy(d_filters, filters_cpu.data(), filters_cpu.size() * sizeof(float), cudaMemcpyHostToDevice);
    }
    
    ~CudaEigenConvLayer() {
        cudaFree(d_filters);
        cudaFree(d_patches);
        cudaFree(d_output);
        cublasDestroy(handle);
    }
    
    MatrixXf forward(const MatrixXf& input) {
        // CPU im2col with Eigen
        int outputRows = (input.rows() - kernelSize) / stride + 1;
        int outputCols = (input.cols() - kernelSize) / stride + 1;
        int totalPatches = outputRows * outputCols;
        
        MatrixXf patches(kernelSize * kernelSize, totalPatches);
        int patchIdx = 0;
        for (int i = 0; i <= input.rows() - kernelSize; i += stride) {
            for (int j = 0; j <= input.cols() - kernelSize; j += stride) {
                int idx = 0;
                for (int ki = 0; ki < kernelSize; ki++) {
                    for (int kj = 0; kj < kernelSize; kj++) {
                        patches(idx++, patchIdx) = input(i + ki, j + kj);
                    }
                }
                patchIdx++;
            }
        }
        
        // GPU matrix multiplication
        cudaMalloc(&d_patches, patches.size() * sizeof(float));
        cudaMalloc(&d_output, outputChannels * totalPatches * sizeof(float));
        
        cudaMemcpy(d_patches, patches.data(), patches.size() * sizeof(float), cudaMemcpyHostToDevice);
        
        const float alpha = 1.0f, beta = 0.0f;
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                   totalPatches, outputChannels, kernelSize * kernelSize,
                   &alpha, d_patches, totalPatches,
                   d_filters, kernelSize * kernelSize,
                   &beta, d_output, totalPatches);
        
        // Copy back and apply ReLU with Eigen
        MatrixXf result(outputChannels, totalPatches);
        cudaMemcpy(result.data(), d_output, result.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        return result.cwiseMax(0.0f);
    }
};