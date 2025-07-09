// main.cpp
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <ncnn/net.h>                 // ncnn 네트워크 처리
#include <sys/mman.h>                 // POSIX shared memory
#include <sys/stat.h>                 // 파일 상태 상수
#include <fcntl.h>                    // shm_open 플래그
#include <unistd.h>                   // close(), usleep
#include <thread>                     // std::thread
#include <mutex>                      // std::mutex
#include <condition_variable>         // std::condition_variable
#include <queue>                      // std::queue
#include <vector>                     // std::vector
#include <iostream>                   // std::cout
#include <chrono>                     // std::chrono

#include "yolo.hpp"                   // Yolo 클래스 정의
#include "plate.hpp"
#include "tf_ocr.hpp"                 // TFOCR 클래스 정의
#include "json.hpp"                   // JSON 라이브러리

using json = nlohmann::json;

std::queue<cv::Mat> img_queue;        // 프레임 임시 저장 큐
std::mutex mtx;                       // 큐 보호용 뮤텍스
std::condition_variable cvn;          // 데이터 유무 통지용

const char* SHM_FRAME_NAME = "/busbom_frame";
constexpr int WIDTH = 1280;          
constexpr int HEIGHT = 720;          
constexpr int CH = 3;                

const char* SHM_COFIG_NAME = "/camera_config";
const size_t SHM_CONFIG_SIZE = 4096; // 4KB

const char* SHM_SEQUENCE_NAME = "/busbom_sequence";
const size_t SHM_SEQUENCE_SIZE = 4096; // 4KB

// For OCR processing
struct OcrInput {
    cv::Mat image;
    int index;
};

// 공유 메모리에서 프레임을 읽어 큐에 삽입
void reader_thread()
{
    int fd = shm_open("/busbom_frame", O_RDONLY, 0666);                            //  SHM open
    void* ptr = mmap(nullptr, WIDTH * HEIGHT * CH, PROT_READ, MAP_SHARED, fd, 0);  //  mmap

    while (true)
    {
        // 원본 메모리 버퍼를 OpenCV Mat으로 래핑
        cv::Mat frame(HEIGHT, WIDTH, CV_8UC3, ptr);          
        cv::Mat copy = frame.clone();                      //  safe clone

        {   
            std::unique_lock<std::mutex> lock(mtx);
            if (img_queue.size() >= 2) img_queue.pop();   //  delete old queue frame if size exceeds 2
            img_queue.push(copy);                         //  insert new frame into queue
        }
        cvn.notify_one();                                 // wake up inference thread
        std::this_thread::sleep_for(std::chrono::milliseconds(5)); //  5ms sleep
    }

    munmap(ptr, WIDTH * HEIGHT * CH);    // unmapoing shared memory
    close(fd);                           // close fd
}


void inference_thread()
{
    Yolo yolo;
    yolo.load("lp_detect_v5n.ncnn.param", "lp_detect_v5n.ncnn.bin");  
    std::cout << "Model loaded successfully." << std::endl;
    
    std::cout << "Inference thread started." << std::endl;

    TFOCR ocr;  
    ocr.load_ocr("model.tflite", "labels.names");  // Load OCR model and labels
    std::cout << "OCR model loaded successfully." << std::endl;

    PlatePrep plate_prep;  // Create PlateOCR instance
    std::cout << "PlateOCR instance created." << std::endl;

    while (true)
    {

        // objects : {cv::Rect_<float> rect; int label; float prob;}
        // frame : cv::Mat
        cv::Mat frame;
        std::vector<Object> objects;  
        {
            std::unique_lock<std::mutex> lock(mtx);
            cvn.wait(lock, []{ return !img_queue.empty(); });   // 데이터 올 때까지 대기
            frame = img_queue.front();                        
            img_queue.pop();                                    // 큐에서 제거
        }
        int w_center = frame.cols / 2; // 프레임 중앙 x 좌표
        int h_center = frame.rows / 2; // 프레임 중앙 y 좌표
        cv::Point2f center = cv::Point2f(w_center, h_center);  
        
        auto t1 = std::chrono::high_resolution_clock::now();
        // frame -> yolo.detect() -> objects[]
        yolo.detect(frame, objects, 640, 0.25f, 0.45f);                     // 추론 수행
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "[MAIN] yolo.detect() latency: " << ms << " ms" << std::endl;

        // frame, objects -> yolo.crop_objects() -> cropped[]
        // cropped license plate images from objects & frame
        std::vector<cv::Mat> cropped = yolo.crop_objects(frame, objects);   // 결과 객체 크롭

        // frame, objects, point -> yolo.calc_distance() -> objects[]
        // calculate distance from center point to each detected object, sort by distance
        // and draw lines from center to each object (-> drawed on frame)
        // objects[] will be updated with distance in prob field
        yolo.calc_distance(objects, center, frame); 

        for (size_t i = 0; i < cropped.size(); ++i)
        {
            cv::Mat preprocessed_plate = plate_prep.preprocess_plate(cropped[i], i); // 전처리
            if (preprocessed_plate.empty()) continue;  // 전처리 실패 시 건너뛰기
            
            OcrInput ocr_input;
            ocr_input.image = preprocessed_plate;
            ocr_input.index = i;

            std::string each_result = ocr.run_ocr(preprocessed_plate); // OCR 실행
            std::cout << "OCR Result for index " << i << ": " << each_result << std::endl;
            objects[i].ocr_result = each_result; // OCR 결과 저장

        }
        
        // Create JSON object from 'objects' and write to shared memory
        json json_objects = json::array();
        for (size_t i = 0; i < objects.size(); ++i) {
            json_objects.push_back({
                {"platform", static_cast<int>(i) + 1},
                {"status", "approaching"},
                {"busNumber", objects[i].ocr_result}
            });
        }

        // Write JSON to shared memory
        int shm_fd = shm_open(SHM_SEQUENCE_NAME, O_CREAT | O_RDWR, 0666);
        if (shm_fd == -1) {
            std::cerr << "Failed to open shared memory for sequence: " << strerror(errno) << std::endl;
        } else {
            if (ftruncate(shm_fd, SHM_SEQUENCE_SIZE) == -1) {
                std::cerr << "Failed to set shared memory size for sequence: " << strerror(errno) << std::endl;
            } else {
                char* shm_ptr = (char*)mmap(0, SHM_SEQUENCE_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
                if (shm_ptr == MAP_FAILED) {
                    std::cerr << "Failed to map shared memory for sequence: " << strerror(errno) << std::endl;
                } else {
                    std::string json_str = json_objects.dump();
                    strncpy(shm_ptr, json_str.c_str(), SHM_SEQUENCE_SIZE - 1);
                    shm_ptr[SHM_SEQUENCE_SIZE - 1] = '\0'; // Ensure null termination

                    munmap(shm_ptr, SHM_SEQUENCE_SIZE);
                }
            }
            close(shm_fd);
        }
        
        // frame, objects -> yolo.draw_result() -> one_shot
        cv::Mat one_shot = yolo.draw_result(frame, objects);                // 결과 이미지에 그리기
        cv::imwrite("result.jpg", one_shot); // 결과 이미지 저장
        
    }
}


int main()
{
    std::cout << "Starting YOLO License Plate Detection..." << std::endl;
    std::thread t1(reader_thread);    // 프레임 읽기 스레드 시작
    std::thread t2(inference_thread); // 추론 스레드 시작
    t1.join();                        // 메인 스레드에서 대기
    t2.join();
    return 0;
}
