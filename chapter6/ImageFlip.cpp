#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    
    cv::Mat flipped_horizontal, flipped_vertical, flipped_both;
    
    cv::flip(image, flipped_horizontal, 1);   // Horizontal flip
    cv::flip(image, flipped_vertical, 0);     // Vertical flip
    cv::flip(image, flipped_both, -1);        // Both axes
    
    cv::imwrite("flipped_horizontal.jpg", flipped_horizontal);
    cv::imwrite("flipped_vertical.jpg", flipped_vertical);
    cv::imwrite("flipped_both.jpg", flipped_both);
    
    std::cout << "Flip completed: horizontal, vertical, and both" << std::endl;
    
    return 0;
}
