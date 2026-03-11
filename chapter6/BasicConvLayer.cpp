#include <vector>
#include <random>
#include <algorithm>
#include <iostream>
#include <cmath>

using namespace std;

class ConvolutionalLayer {
private:
    vector<vector<float>> filters;  // Convolution filters/kernels
    int stride;                     // Filter stride
    int padding;                    // Padding size
 
public:
    ConvolutionalLayer(int numFilters, int filterSize, int strideSize, int paddingSize) {
        // Initialize filters, stride and padding parameters
        this->stride = strideSize;
        this->padding = paddingSize;
        initializeFilters(numFilters, filterSize);
    }
 
    vector<vector<float>> forward(const vector<vector<float>>& input) {
        // Apply convolution operation
        vector<vector<float>> featureMaps;
 
        // For each filter
        for(auto& filter : filters) {
            // Slide filter across input
            vector<float> featureMap;
            int filterSize = (int)sqrt(filter.size());
            for(int i = 0; i < input.size() - filterSize + 1; i += stride) {
                for(int j = 0; j < input.size() - filterSize + 1; j += stride) {
                    // Calculate convolution at current position
                    float sum = 0;
                    for(int k = 0; k < filterSize; k++) {
                        for(int l = 0; l < filterSize; l++) {
                            sum += input[i+k][j+l] * filter[k * filterSize + l];
                        }
                    }
                    // Apply ReLU activation
                    featureMap.push_back(max(0.0f, sum));
                }
            }
            featureMaps.push_back(featureMap);
        }
        return featureMaps;
    }

private:
    void initializeFilters(int numFilters, int filterSize) {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dist(0.0f, 0.1f);
        
        filters.resize(numFilters);
        for(int i = 0; i < numFilters; i++) {
            filters[i].resize(filterSize * filterSize);
            for(int j = 0; j < filterSize * filterSize; j++) {
                filters[i][j] = dist(gen);
            }
        }
    }
};

int main() {
    // Create 3x3 convolutional layer with 2 filters
    ConvolutionalLayer conv(2, 3, 1, 0);
    
    // Create sample 5x5 input
    vector<vector<float>> input = {
        {1, 2, 3, 4, 5},
        {6, 7, 8, 9, 10},
        {11, 12, 13, 14, 15},
        {16, 17, 18, 19, 20},
        {21, 22, 23, 24, 25}
    };
    
    // Forward pass
    auto output = conv.forward(input);
    
    cout << "Input size: " << input.size() << "x" << input[0].size() << endl;
    cout << "Number of feature maps: " << output.size() << endl;
    cout << "Feature map size: " << output[0].size() << endl;
    
    return 0;
}
