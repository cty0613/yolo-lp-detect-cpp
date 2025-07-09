// main.cpp
#include "tf_ocr.hpp"

std::map<int, std::string> TFOCR::loadLabelMap(const std::string& path) {
    std::map<int, std::string> label_map;
    std::ifstream file(path);
    std::string line;
    int idx = 0;
    while (std::getline(file, line)) {
        label_map[idx++] = line;
    }
    return label_map;
}

std::vector<int> TFOCR::ctcGreedyDecoder(const float* logits, int time, int classes) {
    std::vector<int> result;
    int prev = -1;
    for (int t = 0; t < time; ++t) {
        int max_index = 0;
        float max_val = logits[t * classes];
        for (int c = 1; c < classes; ++c) {
            float val = logits[t * classes + c];
            if (val > max_val) {
                max_val = val;
                max_index = c;
            }
        }
        if (max_index != prev && max_index != 0) {
            result.push_back(max_index);
        }
        prev = max_index;
    }
    return result;
}

void TFOCR::load_ocr(const std::string& model_path, const std::string& labels_path) {
    // Load label map
    label_map = loadLabelMap(labels_path);

    // Load TFLite model
    model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    tflite::ops::builtin::BuiltinOpResolver resolver;
    // Removed local interpreter declaration, now using member variable
    tflite::InterpreterBuilder(*model, resolver)(&interpreter);
    interpreter->AllocateTensors();

    // Set input and output details
    // auto input_details = interpreter->inputs();
    // float* input = interpreter->typed_input_tensor<float>(0);
    // auto output_details = interpreter->outputs();
    // auto output_details = interpreter->tensor(interpreter->outputs()[0]);
}

std::string TFOCR::run_ocr(const cv::Mat& input_img) {

    cv::Mat gray;
    cv::cvtColor(input_img, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray, cv::Size(192, 96));
    gray.convertTo(gray, CV_32FC1, 1.0 / 255.0);

    // Set input
    float* input = interpreter->typed_input_tensor<float>(0);
    memcpy(input, gray.data, 96 * 192 * sizeof(float));

    // Run inference
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "ocr inference failed..." << std::endl;
        return ""; // Changed from -1 to ""
    }

    // Decode output
    auto output_details = interpreter->tensor(interpreter->outputs()[0]);
    const float* output_data = output_details->data.f;
    int time = output_details->dims->data[1];
    int classes = output_details->dims->data[2];

    std::vector<int> indices = ctcGreedyDecoder(output_data, time, classes);

    // Convert indices to characters
    std::string result;
    for (int idx : indices) {
        if (label_map.count(idx)) {
            result += label_map[idx];
        }
    }

    return result;
}
