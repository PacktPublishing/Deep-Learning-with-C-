#include <opencv2/opencv.hpp>
#include <iostream>

cv::Mat applyTranslation(const cv::Mat& image, int tx, int ty) {
    cv::Mat translation_matrix = (cv::Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);
    cv::Mat translated;
    cv::warpAffine(image, translated, translation_matrix, image.size());
    return translated;
}

cv::Mat applyCropping(const cv::Mat& image, int x, int y, int width, int height) {
    cv::Rect roi(x, y, width, height);
    return image(roi).clone();
}

cv::Mat applyScaling(const cv::Mat& image, double scale_factor) {
    cv::Mat scaled;
    cv::resize(image, scaled, cv::Size(), scale_factor, scale_factor, cv::INTER_LINEAR);
    return scaled;
}

cv::Mat applyZoom(const cv::Mat& image, double zoom_factor) {
    int cx = image.cols / 2, cy = image.rows / 2;
    int zoom_width = image.cols / zoom_factor, zoom_height = image.rows / zoom_factor;
    cv::Rect zoom_roi(cx - zoom_width / 2, cy - zoom_height / 2, zoom_width, zoom_height);
    cv::Mat zoomed_crop = image(zoom_roi);
    cv::Mat zoomed;
    cv::resize(zoomed_crop, zoomed, image.size(), 0, 0, cv::INTER_LINEAR);
    return zoomed;
}

cv::Mat applyFlip(const cv::Mat& image, int flip_code) {
    cv::Mat flipped;
    cv::flip(image, flipped, flip_code);
    return flipped;
}

cv::Mat applyPadding(const cv::Mat& image, int top, int bottom, int left, int right) {
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, top, bottom, left, right, 
                       cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
    return padded;
}

cv::Mat applyResampling(const cv::Mat& image, cv::Size new_size, int interpolation) {
    cv::Mat resampled;
    cv::resize(image, resampled, new_size, 0, 0, interpolation);
    return resampled;
}

int main() {
    cv::Mat image = cv::imread("input.jpg");
    if (image.empty()) {
        std::cerr << "Error loading image" << std::endl;
        return -1;
    }
    
    std::cout << "Original size: " << image.size() << std::endl;
    
    // Apply combined transforms
    cv::Mat result = image.clone();
    
    result = applyScaling(result, 0.8);
    std::cout << "After scaling: " << result.size() << std::endl;
    
    result = applyTranslation(result, 20, 15);
    std::cout << "After translation: " << result.size() << std::endl;
    
    result = applyFlip(result, 1);
    std::cout << "After horizontal flip: " << result.size() << std::endl;
    
    result = applyPadding(result, 30, 30, 30, 30);
    std::cout << "After padding: " << result.size() << std::endl;
    
    result = applyResampling(result, cv::Size(400, 400), cv::INTER_CUBIC);
    std::cout << "After resampling: " << result.size() << std::endl;
    
    cv::imwrite("combined_transforms.jpg", result);
    std::cout << "Combined transforms completed successfully!" << std::endl;
    
    return 0;
}
