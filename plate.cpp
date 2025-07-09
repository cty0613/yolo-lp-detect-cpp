#include "plate.hpp"
#include <iostream>
#include <sys/stat.h>

PlatePrep::PlatePrep(const std::string& preprocess_dir) : preprocess_dir(preprocess_dir) {
    mkdir(this->preprocess_dir.c_str(), 0777);
}

void PlatePrep::save(const cv::Mat& img, const std::string& filename) {
    std::string full_path = preprocess_dir + "/" + filename;
    cv::imwrite(full_path, img);
}

std::vector<cv::Point2f> PlatePrep::order_points(const std::vector<cv::Point>& pts) {
    std::vector<cv::Point2f> src(4);

    auto sum = [](const cv::Point2f& p) { return p.x + p.y; };
    auto diff = [](const cv::Point2f& p) { return p.y - p.x; };

    src[0] = *std::min_element(pts.begin(), pts.end(), [&](const cv::Point2f& a, const cv::Point2f& b) { return sum(a) < sum(b); }); // TL
    src[2] = *std::max_element(pts.begin(), pts.end(), [&](const cv::Point2f& a, const cv::Point2f& b) { return sum(a) < sum(b); }); // BR
    src[1] = *std::min_element(pts.begin(), pts.end(), [&](const cv::Point2f& a, const cv::Point2f& b) { return diff(a) < diff(b); }); // TR
    src[3] = *std::max_element(pts.begin(), pts.end(), [&](const cv::Point2f& a, const cv::Point2f& b) { return diff(a) < diff(b); }); // BL

    return src;
}

void PlatePrep::remove_side_dots(cv::Mat& img) {
    int img_width = img.cols;
    int img_height = img.rows;

    int margin_px = static_cast<int>(img_width * (70.0 / 335.0));
    cv::rectangle(img, cv::Point(0, 0), cv::Point(img_width / 2, img_height), cv::Scalar(255, 255, 255), cv::FILLED);
    cv::rectangle(img, cv::Point(img_width - margin_px, 0), cv::Point(img_width, img_height), cv::Scalar(255, 255, 255), cv::FILLED);

    save(img, "08_upper_cover.png");
}

cv::Mat PlatePrep::preprocess(const cv::Mat& input, float resize_factor) {
    cv::Mat gray, resized, binary, kernel;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(), resize_factor, resize_factor, cv::INTER_CUBIC);
    cv::threshold(resized, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
    kernel = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    cv::dilate(binary, binary, kernel, cv::Point(-1, -1), 1);
    return binary;
}

// std::string PlatePrep::ocr_eng(const cv::Mat& image, const std::string& whitelist) {
//     tesseract::TessBaseAPI tess;
//     tess.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
//     tess.SetVariable("tessedit_char_whitelist", whitelist.c_str());
//     tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
//     tess.SetImage(image.data, image.cols, image.rows, 1, image.step);
//     std::string out = tess.GetUTF8Text();
//     tess.End();
//     return out;
// }

// std::string PlatePrep::ocr_kor(const cv::Mat& image, const std::string& whitelist) {
//     tesseract::TessBaseAPI tess;
//     tess.Init(NULL, "kor", tesseract::OEM_LSTM_ONLY);
//     tess.SetVariable("tessedit_char_whitelist", whitelist.c_str());
//     tess.SetPageSegMode(tesseract::PSM_SINGLE_LINE);
//     tess.SetImage(image.data, image.cols, image.rows, 1, image.step);
//     std::string out = tess.GetUTF8Text();
//     tess.End();
//     return out;
// }

bool PlatePrep::extract_plate_region(const cv::Mat& input, cv::Mat& output_plate, const std::string& prefix) {
    cv::Mat hsv;
    cv::cvtColor(input, hsv, cv::COLOR_BGR2HSV);
    save(hsv, prefix + "01_hsv.png");

    cv::Scalar lower_yellow(15, 100, 100);
    cv::Scalar upper_yellow(35, 255, 255);
    cv::Mat mask;
    cv::inRange(hsv, lower_yellow, upper_yellow, mask);
    save(mask, prefix + "02_yellow_mask.png");

    cv::Mat masked;
    cv::bitwise_and(input, input, masked, mask);
    save(masked, prefix + "03_yellow_region.png");

    cv::Mat gray, blur, edges;
    cv::cvtColor(masked, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blur, cv::Size(5, 5), 0);
    cv::Canny(blur, edges, 50, 150);
    save(edges, prefix + "04_edges.png");

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(edges, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<cv::Point> candidate;

    for (const auto& cnt : contours) {
        std::vector<cv::Point> approx;
        cv::approxPolyDP(cnt, approx, 0.02 * cv::arcLength(cnt, true), true);
        double area = cv::contourArea(approx);
        if (approx.size() == 4 && area > 500) {
            candidate = approx;
            break;
        }
    }

    if (candidate.empty()) {
        return false;
    }

    std::vector<cv::Point2f> corners = order_points(candidate);

    cv::Mat temp = input.clone();
    for (const auto& pt : corners)
        cv::circle(temp, pt, 5, cv::Scalar(0, 255, 0), -1);
    save(temp, prefix + "05_detected_corners.png");

    cv::Point2f tl = corners[0], tr = corners[1], br = corners[2], bl = corners[3];
    int width = static_cast<int>(std::max(cv::norm(br - bl), cv::norm(tr - tl)));
    int height = static_cast<int>(std::max(cv::norm(tr - br), cv::norm(tl - bl)));

    float fx = static_cast<float>(width - 1);
    float fy = static_cast<float>(height - 1);

    std::vector<cv::Point2f> dst_pts = {
        {0.0f, 0.0f},
        {fx, 0.0f},
        {fx, fy},
        {0.0f, fy}
    };

    cv::Mat M = cv::getPerspectiveTransform(corners, dst_pts);
    cv::warpPerspective(input, output_plate, M, cv::Size(width, height));
    save(output_plate, prefix + "06_warped_plate.png");

    return true;
}

cv::Mat PlatePrep::preprocess_plate(const cv::Mat& input_img, int index) {
    cv::Mat plate_img_origin = input_img.clone();

    cv::Mat plate_img;
    if (!extract_plate_region(input_img, plate_img, "img_" + std::to_string(index) + "_")) {
        // std::cout << "❌ [" << index << "] 번호판 사각형 추출 실패" << std::endl;
        return plate_img_origin;
    }

    return plate_img; // Return the processed plate image
    
    // 번호판 상하단 영역 분리
    // cv::Mat upper, lower;
    // int h = plate_img.rows;
    // upper = plate_img(cv::Range(0, static_cast<int>(h * 0.4)), cv::Range::all()).clone();
    // lower = plate_img(cv::Range(static_cast<int>(h * 0.3), h), cv::Range::all()).clone();

    // // 전처리
    // cv::Mat bin_upper = preprocess(upper);
    // cv::Mat bin_lower = preprocess(lower);

    // remove_side_dots(bin_upper);
    // save(bin_upper, "img_" + std::to_string(index) + "_" + "07_upper_image.png");
    // save(bin_lower, "img_" + std::to_string(index) + "_" + "07_lower_image.png");

    // OCR (Tesseract 호출)
    // std::string upper_text = ocr_eng(bin_upper, "0123456789");
    // std::string lower_text = ocr_kor(bin_lower, "0123456789가나다라마바사아자하허호");

    // // 텍스트 정리
    // std::string cleaned_upper, cleaned_lower;
    // for (char c : upper_text) if (std::isdigit(c)) cleaned_upper += c;
    // for (char c : lower_text) if (std::isdigit(c) || (c & 0x80)) cleaned_lower += c;

    // // 판단
    // result.reliability = (!cleaned_upper.empty() && !cleaned_lower.empty()) ? 1 :
    //                      (!cleaned_lower.empty()) ? 0 : -1;
    // result.lpNum = cleaned_upper + " " + cleaned_lower;
    // result.success = (result.reliability != -1);
}
