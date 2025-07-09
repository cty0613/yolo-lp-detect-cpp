#ifndef TF_OCR_HPP
#define TF_OCR_HPP

#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <map>
#include <vector>
#include <string>

class TFOCR {
    public:
        void load_ocr(const std::string& model_path, const std::string& labels_path);
        std::string run_ocr(const cv::Mat& input_img);
    private:
        std::map<int, std::string> loadLabelMap(const std::string& path);
        std::vector<int> ctcGreedyDecoder(const float* logits, int time, int classes);

        std::map<int, std::string> label_map;
        std::unique_ptr<tflite::FlatBufferModel> model;
        std::unique_ptr<tflite::Interpreter> interpreter; // Added interpreter as a member
};

#endif // TF_OCR_HPP