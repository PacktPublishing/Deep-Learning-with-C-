#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    
    // Define ROI (Region of Interest)
    cv::Rect roi(50, 50, 200, 200);  // x, y, width, height
    cv::Mat cropped = image(roi);
    
    cv::imwrite("cropped.jpg", cropped);
    std::cout << "Cropping completed: " << cropped.size() << std::endl;
    
    return 0;
}
