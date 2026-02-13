#include <cuda_runtime.h> 
#include <torch/torch.h> 
 
class CUDAConvolutionalLayer { 
private: 
float* device_filters; 
int inputChannels; 
int outputChannels; 
int kernelSize; 
 
public: 
CUDAConvolutionalLayer(int inChannels = 512, int outChannels = 512, int kSize = 3) { 
inputChannels = inChannels; 
outputChannels = outChannels; 
kernelSize = kSize; 
 
// Allocate memory on GPU for filters 
cudaMalloc(&device_filters, 
outputChannels * inputChannels * kernelSize * kernelSize * sizeof(float)); 
} 
 
__global__ void convolution_kernel(float* input, float* filters, float* output, 
int input_size, int kernel_size, int stride) { 
int idx = blockIdx.x * blockDim.x + threadIdx.x; 
 
// Compute convolution for each thread 
for (int i = 0; i < kernel_size; i++) { 
for (int j = 0; j < kernel_size; j++) { 
// Perform element-wise multiplication and accumulation 
atomicAdd(&output[idx], input[i] * filters[j]); 
} 
} 
} 
 
torch::Tensor forward(torch::Tensor input) { 
// Move input to GPU 
auto cuda_input = input.cuda(); 
 
// Launch CUDA kernel 
int threadsPerBlock = 256; 
int numBlocks = (input.size(0) + threadsPerBlock - 1) / threadsPerBlock; 
 
float* output; 
cudaMalloc(&output, input.size(0) * sizeof(float)); 
 
convolution_kernel<<<numBlocks, threadsPerBlock>>>( 
cuda_input.data_ptr<float>(), 
device_filters, 
output, 
input.size(0), 
kernelSize, 
2  // stride 
); 
 
// Apply ReLU and max pooling 
auto result = torch::from_blob(output, input.sizes()); 
result = torch::relu(result); 
result = torch::max_pool2d(result, {2, 2}, {2, 2}); 
 
return result; 
} 
 
~CUDAConvolutionalLayer() { 
cudaFree(device_filters); } };
