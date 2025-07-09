
#ifndef YOLO_HPP
#define YOLO_HPP

#include <opencv2/core/core.hpp>
#include <ncnn/net.h>

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::string ocr_result = "AA99A9999";
};

class Yolo
{
public:
    Yolo();
    ~Yolo();
    void load(const std::string& param_path, const std::string& model_path);
    int detect(cv::Mat bgr, std::vector<Object>& objects, int target_size, float prob_threshold, float nms_threshold);
    cv::Mat draw_result(const cv::Mat& bgr, const std::vector<Object>& objects);
    void calc_distance(std::vector<Object>& objects, const cv::Point2f& point, cv::Mat& image);
    std::vector<cv::Mat> crop_objects(const cv::Mat& bgr, const std::vector<Object>& objects);
private:
    ncnn::Net yolov5;
};

#endif // YOLO_HPP