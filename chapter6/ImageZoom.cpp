#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    
    // Zoom into center region
    int cx = image.cols / 2, cy = image.rows / 2;
    int zoom_width = image.cols / 2, zoom_height = image.rows / 2;
    
    cv::Rect zoom_roi(cx - zoom_width / 2, cy - zoom_height / 2, zoom_width, zoom_height);
    cv::Mat zoomed_crop = image(zoom_roi);
    
    cv::Mat zoomed;
    cv::resize(zoomed_crop, zoomed, image.size(), 0, 0, cv::INTER_LINEAR);
    
    cv::imwrite("zoomed.jpg", zoomed);
    std::cout << "Zoom completed: 2x magnification" << std::endl;
    
    return 0;
}
