#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <cmath>

using namespace std;
using namespace Eigen;
using Matrix = MatrixXf;

// Matrix utility functions
Matrix createMatrix(int rows, int cols) {
    Matrix mat = Matrix::Random(rows, cols) * 0.1f;
    return mat;
}

void applyReLU(Matrix& mat) {
    mat = mat.cwiseMax(0.0f);
}

class OptimizedConvolutionalLayer { 
private: 
    Matrix filters;  // Store filters as a matrix 
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
 
    Matrix forward(const Matrix& input) { 
        // Convert input to matrix format for efficient computation 
        Matrix inputMatrix = im2col(input, kernelSize, stride); 
 
        // Perform matrix multiplication instead of explicit convolution 
        Matrix output = filters * inputMatrix; 
 
        // Apply ReLU activation
        applyReLU(output);
        
        return output; 
    }

private:
    Matrix im2col(const Matrix& input, int kSize, int stride) {
        int inputRows = input.rows();
        int inputCols = input.cols();
        int outputRows = (inputRows - kSize) / stride + 1;
        int outputCols = (inputCols - kSize) / stride + 1;
        Matrix result(kSize * kSize, outputRows * outputCols);
        
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
    OptimizedConvolutionalLayer conv(3, 64, 3, 1);
    
    Matrix input(8, 8);
    for(int i = 0; i < 8; i++) {
        for(int j = 0; j < 8; j++) {
            input(i, j) = i * 8 + j + 1;
        }
    }
    
    Matrix output = conv.forward(input);
    
    cout << "Input: " << input.rows() << "x" << input.cols() << endl;
    cout << "Output: " << output.rows() << "x" << output.cols() << endl;
    
    return 0;
} 
