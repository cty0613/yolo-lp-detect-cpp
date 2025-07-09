#ifndef PLATE_HPP
#define PLATE_HPP

#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <string>
#include <vector>

struct OcrResult {
    int number;
    int reliability;    // 0: 하단만 정확, 1: 상+하단 정확, -1: 실패
    std::string lpNum;
    bool success;
};

class PlatePrep {
public:
    PlatePrep(const std::string& preprocess_dir = "preprocess");
    cv::Mat preprocess_plate(const cv::Mat& input_img, int index);

private:
    std::string preprocess_dir;

    void save(const cv::Mat& img, const std::string& filename);
    std::vector<cv::Point2f> order_points(const std::vector<cv::Point>& pts);
    void remove_side_dots(cv::Mat& img);
    cv::Mat preprocess(const cv::Mat& input, float resize_factor = 3);
    // std::string ocr_eng(const cv::Mat& image, const std::string& whitelist);
    // std::string ocr_kor(const cv::Mat& image, const std::string& whitelist);
    bool extract_plate_region(const cv::Mat& input, cv::Mat& output_plate, const std::string& prefix);
};

#endif // PLATE_OCR_HPP
