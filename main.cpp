// main.cpp
#include "json.hpp"                   // JSON 라이브러리
#include "yolo.hpp"                   // Yolo 클래스 정의
#include "plate.hpp"

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

// Removed Unix Domain Sockets includes

using json = nlohmann::json;

constexpr int WIDTH = 1280;          
constexpr int HEIGHT = 720;          
constexpr int CH = 3;                

// Shared memory constants
constexpr size_t MAX_IMAGE_BUFFER_SIZE = 2 * 1024 * 1024; // 1MB for compressed image data

// Structure for shared memory image data
struct ShmImage {
    uint32_t size; // Size of the image data
    uint32_t frame_id; // To indicate new data
    uchar data[MAX_IMAGE_BUFFER_SIZE]; // Image data
};

std::queue<cv::Mat> img_queue;        // 프레임 임시 저장 큐
std::mutex mtx;                       // 큐 보호용 뮤텍스
std::condition_variable cvn;          // 데이터 유무 통지용

// For OCR processing
struct OcrInput {
    cv::Mat image;
    int index;
};
std::queue<OcrInput> ocr_queue;
std::mutex ocr_mtx;
std::condition_variable ocr_cvn;

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

void ocr_thread()
{
    PlateOCR ocr("preprocess");
    std::cout << "OCR thread started." << std::endl;

    while (true)
    {
        OcrInput ocr_input;
        {
            std::unique_lock<std::mutex> lock(ocr_mtx);
            ocr_cvn.wait(lock, []{ return !ocr_queue.empty(); });
            ocr_input = ocr_queue.front();
            ocr_queue.pop();
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        OcrResult ocr_result = ocr.process_plate(ocr_input.image, ocr_input.index);
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        if (ocr_result.success) {
            std::cout << "[OCR] :: Plate " << ocr_input.index << " OCR: " << ocr_result.lpNum << " (took " << ms << " ms)" << std::endl;
        }
    }
}

void inference_thread()
{
    Yolo yolo;
    yolo.load("lp_detect_v5n.ncnn.param", "lp_detect_v5n.ncnn.bin");  
    std::cout << "Model loaded successfully." << std::endl;
    
    // Shared memory for one_shot image
    // const char* one_shot_shm_name = "/yolo_result_shm";
    // int one_shot_shm_fd = shm_open(one_shot_shm_name, O_CREAT | O_RDWR, 0666);
    // if (one_shot_shm_fd == -1) {
    //     perror("shm_open for one_shot");
    //     return;
    // }
    // if (ftruncate(one_shot_shm_fd, sizeof(ShmImage)) == -1) {
    //     perror("ftruncate for one_shot");
    //     close(one_shot_shm_fd);
    //     return;
    // }
    // ShmImage* one_shot_shm_ptr = (ShmImage*)mmap(nullptr, sizeof(ShmImage), PROT_READ | PROT_WRITE, MAP_SHARED, one_shot_shm_fd, 0);
    // if (one_shot_shm_ptr == MAP_FAILED) {
    //     perror("mmap for one_shot");
    //     close(one_shot_shm_fd);
    //     return;
    // }
    // one_shot_shm_ptr->frame_id = 0; // Initialize frame ID
    // std::cout << "Shared memory for one_shot results created: " << one_shot_shm_name << std::endl;

    // Shared memory for cropped images
    // const int NUM_CROPPED_SHM = 5;
    // int cropped_shm_fds[NUM_CROPPED_SHM];
    // ShmImage* cropped_shm_ptrs[NUM_CROPPED_SHM];
    // char cropped_shm_names[NUM_CROPPED_SHM][64];

    // for (int j = 0; j < NUM_CROPPED_SHM; ++j) {
    //     sprintf(cropped_shm_names[j], "/yolo_cropped_shm_%d", j);
    //     cropped_shm_fds[j] = shm_open(cropped_shm_names[j], O_CREAT | O_RDWR, 0666);
    //     if (cropped_shm_fds[j] == -1) {
    //         perror("shm_open for cropped");
    //         return;
    //     }
    //     if (ftruncate(cropped_shm_fds[j], sizeof(ShmImage)) == -1) {
    //         perror("ftruncate for cropped");
    //         close(cropped_shm_fds[j]);
    //         return;
    //     }
    //     cropped_shm_ptrs[j] = (ShmImage*)mmap(nullptr, sizeof(ShmImage), PROT_READ | PROT_WRITE, MAP_SHARED, cropped_shm_fds[j], 0);
    //     if (cropped_shm_ptrs[j] == MAP_FAILED) {
    //         perror("mmap for cropped");
    //         close(cropped_shm_fds[j]);
    //         return;
    //     }
    //     cropped_shm_ptrs[j]->frame_id = 0; // Initialize frame ID
    //     std::cout << "Shared memory for cropped results " << j << " created: " << cropped_shm_names[j] << std::endl;
    // }

    std::cout << "Inference thread started." << std::endl;

    // uint32_t current_frame_id = 0;

    while (true)
    {
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
        yolo.detect(frame, objects, 640, 0.25f, 0.45f);                     // 추론 수행
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "[MAIN] yolo.detect() latency: " << ms << " ms" << std::endl;

        std::vector<cv::Mat> results = yolo.crop_objects(frame, objects);   // 결과 객체 크롭
        
        // Push cropped images to OCR queue
        if (!results.empty()) {
            std::unique_lock<std::mutex> lock(ocr_mtx);
            for (size_t i = 0; i < results.size(); ++i) {
                if (ocr_queue.size() >= 5) { // Limit queue size
                    ocr_queue.pop();
                }
                ocr_queue.push({results[i], (int)i});
            }
            ocr_cvn.notify_one();
        }

        yolo.calc_distance(objects, center, frame); 
        cv::Mat one_shot = yolo.draw_result(frame, objects);                // 결과 이미지에 그리기
        
        // current_frame_id++;
        
        // Write one_shot image to shared memory
        // std::vector<uchar> one_shot_buf;
        // cv::imencode(".jpg", one_shot, one_shot_buf);
        // if (one_shot_buf.size() <= MAX_IMAGE_BUFFER_SIZE) {
        //     one_shot_shm_ptr->size = one_shot_buf.size();
        //     memcpy(one_shot_shm_ptr->data, one_shot_buf.data(), one_shot_buf.size());
        //     one_shot_shm_ptr->frame_id = current_frame_id;
        // } else {
        //     std::cerr << "One-shot image too large for shared memory." << std::endl;
        // }

        // // Write cropped results to shared memory
        // for (size_t i = 0; i < results.size(); ++i) {
        //     int shm_index = i % NUM_CROPPED_SHM;
        //     std::vector<uchar> cropped_buf;
        //     cv::imencode(".jpg", results[i], cropped_buf);
        //     if (cropped_buf.size() <= MAX_IMAGE_BUFFER_SIZE) {
        //         cropped_shm_ptrs[shm_index]->size = cropped_buf.size();
        //         memcpy(cropped_shm_ptrs[shm_index]->data, cropped_buf.data(), cropped_buf.size());
        //         cropped_shm_ptrs[shm_index]->frame_id = current_frame_id;
        //     } else {
        //         std::cerr << "Cropped image " << i << " too large for shared memory." << std::endl;
        //     }
        // }
    }

    // Cleanup shared memory (this part will not be reached in the infinite loop)
    // munmap(one_shot_shm_ptr, sizeof(ShmImage));
    // close(one_shot_shm_fd);
    // shm_unlink(one_shot_shm_name);

    // for (int j = 0; j < NUM_CROPPED_SHM; ++j) {
    //     munmap(cropped_shm_ptrs[j], sizeof(ShmImage));
    //     close(cropped_shm_fds[j]);
    //     shm_unlink(cropped_shm_names[j]);
    // }
}


int main()
{
    std::cout << "Starting YOLO License Plate Detection..." << std::endl;
    std::thread t1(reader_thread);    // 프레임 읽기 스레드 시작
    std::thread t2(inference_thread); // 추론 스레드 시작
    std::thread t3(ocr_thread);       // OCR 스레드 시작
    t1.join();                        // 메인 스레드에서 대기
    t2.join();
    t3.join();
    return 0;
}
