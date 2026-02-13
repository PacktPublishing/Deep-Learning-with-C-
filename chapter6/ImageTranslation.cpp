#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    
    // Translation matrix: shift by (tx, ty)
    int tx = 50, ty = 30;
    cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);
    
    cv::Mat translated;
    cv::warpAffine(image, translated, translation_matrix, image.size());
    
    cv::imwrite("translated.jpg", translated);
    std::cout << "Translation completed: shifted by (" << tx << ", " << ty << ")" << std::endl;
    
    return 0;
}
