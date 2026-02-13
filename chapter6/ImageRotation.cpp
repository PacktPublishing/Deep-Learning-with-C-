#include <cuda_runtime.h>
#include <torch/torch.h>

class CUDAImageRotation {
private:
    float min_angle;
    float max_angle;
    
    __global__ void rotate_kernel(
        float* input, 
        float* output, 
        int height, 
        int width, 
        float sin_theta, 
        float cos_theta
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        int idy = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (idx < width && idy < height) {
            // Calculate source coordinates
            float x = idx - width/2;
            float y = idy - height/2;
            
            int src_x = round(x * cos_theta - y * sin_theta + width/2);
            int src_y = round(x * sin_theta + y * cos_theta + height/2);
            
            // Check bounds and copy pixel
            if (src_x >= 0 && src_x < width && src_y >= 0 && src_y < height) {
                output[idy * width + idx] = input[src_y * width + src_x];
            }
        }
    }

public:
    CUDAImageRotation(float min = -180.0f, float max = 180.0f) 
        : min_angle(min), max_angle(max) {}
    
    torch::Tensor rotate(torch::Tensor input, float angle) {
        // Ensure input is on CUDA
        input = input.cuda();
        
        // Create output tensor
        auto output = torch::zeros_like(input);
        
        // Calculate rotation parameters
        float rad = angle * M_PI / 180.0f;
        float sin_theta = sin(rad);
        float cos_theta = cos(rad);
        
        // Set up CUDA kernel parameters
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks(
            (input.size(1) + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (input.size(0) + threadsPerBlock.y - 1) / threadsPerBlock.y
        );
        
        // Launch kernel
        rotate_kernel<<<numBlocks, threadsPerBlock>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            input.size(0),
            input.size(1),
            sin_theta,
            cos_theta
        );
        
        return output;
    }
    
    torch::Tensor random_rotate(torch::Tensor input) {
        float angle = min_angle + 
            static_cast<float>(rand()) / 
            (static_cast<float>(RAND_MAX/(max_angle-min_angle)));
        return rotate(input, angle);
    }
};