#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

using namespace std;
using namespace Eigen;

// Matrix utility functions
MatrixXf createMatrix(int rows, int cols) {
    MatrixXf mat = MatrixXf::Random(rows, cols) * 0.1f;
    return mat;
}

void applyReLU(MatrixXf& mat) {
    mat = mat.cwiseMax(0.0f);
}

class OptimizedConvolutionalLayer { 
private: 
    MatrixXf filters;  // Store filters as a matrix 
    int inputChannels;  // Number of input channels 
    int outputChannels; // Number of output channels 
    int kernelSize;
    int stride; 
 
public: 
    OptimizedConvolutionalLayer(int inChannels, int outChannels, int kSize, int strideSize) { 
        inputChannels = inChannels; 
        outputChannels = outChannels; 
        kernelSize = kSize;
        stride = strideSize; 
        // Initialize filters as matrix 
        filters = createMatrix(outChannels, inChannels * kernelSize * kernelSize); 
    } 
 
    MatrixXf forward(const MatrixXf& input) { 
        // Convert input to matrix format for efficient computation 
        MatrixXf inputMatrix = im2col(input, kernelSize, stride); 
 
        // Perform matrix multiplication instead of explicit convolution 
        MatrixXf output = filters * inputMatrix; 
 
        // Apply ReLU activation
        applyReLU(output);
        
        return output; 
    }

private:
    MatrixXf im2col(const MatrixXf& input, int kSize, int stride) {
        int inputRows = input.rows();
        int inputCols = input.cols();
        int outputRows = (inputRows - kSize) / stride + 1;
        int outputCols = (inputCols - kSize) / stride + 1;
        MatrixXf result(kSize * kSize, outputRows * outputCols);
        
        int colIdx = 0;
        for(int i = 0; i <= inputRows - kSize; i += stride) {
            for(int j = 0; j <= inputCols - kSize; j += stride) {
                int rowIdx = 0;
                for(int ki = 0; ki < kSize; ki++) {
                    for(int kj = 0; kj < kSize; kj++) {
                        result(rowIdx++, colIdx) = input(i + ki, j + kj);
                    }
                }
                colIdx++;
            }
        }
        return result;
    }


};

int main() {
    OptimizedConvolutionalLayer conv(1, 64, 3, 1);
    
    MatrixXf input(8, 8);
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            input(i, j) = i * 8 + j + 1;
        }
    }
    
    MatrixXf output = conv.forward(input);
    
    cout << "Input: " << input.rows() << "x" << input.cols() << endl;
    cout << "Output: " << output.rows() << "x" << output.cols() << endl;
    
    return 0;
} 
