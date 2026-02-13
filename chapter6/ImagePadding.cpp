#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    
    int top = 50, bottom = 50, left = 50, right = 50;
    cv::Mat padded;
    
    // Different padding types
    cv::copyMakeBorder(image, padded, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    
    cv::imwrite("padded.jpg", padded);
    std::cout << "Padding completed: " << image.size() << " -> " << padded.size() << std::endl;
    
    return 0;
}
