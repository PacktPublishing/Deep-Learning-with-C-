#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    
    cv::Size new_size(300, 300);
    cv::Mat resampled_nearest, resampled_linear, resampled_cubic, resampled_lanczos;
    
    cv::resize(image, resampled_nearest, new_size, 0, 0, cv::INTER_NEAREST);
    cv::resize(image, resampled_linear, new_size, 0, 0, cv::INTER_LINEAR);
    cv::resize(image, resampled_cubic, new_size, 0, 0, cv::INTER_CUBIC);
    cv::resize(image, resampled_lanczos, new_size, 0, 0, cv::INTER_LANCZOS4);
    
    cv::imwrite("resampled_nearest.jpg", resampled_nearest);
    cv::imwrite("resampled_linear.jpg", resampled_linear);
    cv::imwrite("resampled_cubic.jpg", resampled_cubic);
    cv::imwrite("resampled_lanczos.jpg", resampled_lanczos);
    
    std::cout << "Resampling completed with 4 interpolation methods" << std::endl;
    
    return 0;
}
