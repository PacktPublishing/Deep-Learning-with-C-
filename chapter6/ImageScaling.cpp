#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    
    // Scale by factor
    double scale_factor = 0.5;
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
    
    cv::imwrite("scaled.jpg", scaled);
    std::cout << "Scaling completed: " << image.size() << " -> " << scaled.size() << std::endl;
    
    return 0;
}
