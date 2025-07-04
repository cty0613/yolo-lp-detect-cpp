// main.cpp
#include "json.hpp"                   // JSON 라이브러리
#include "yolo.hpp"                   // Yolo 클래스 정의

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

using json = nlohmann::json;

constexpr int WIDTH = 1280;          
constexpr int HEIGHT = 720;          
constexpr int CH = 3;                

std::queue<cv::Mat> img_queue;        // 프레임 임시 저장 큐
std::mutex mtx;                       // 큐 보호용 뮤텍스
std::condition_variable cvn;          // 데이터 유무 통지용

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
    
    while (true)
    {
        cv::Mat frame;
        std::vector<Object> objects;  
        {
            std::unique_lock<std::mutex> lock(mtx);
            cvn.wait(lock, []{ return !img_queue.empty(); });  // 데이터 올 때까지 대기
            frame = img_queue.front();                        
            img_queue.pop();                                  // 큐에서 제거
        }
        int w_center = frame.cols / 2; // 프레임 중앙 x 좌표
        int h_center = frame.rows / 2; // 프레임 중앙 y 좌표
        cv::Point2f center = cv::Point2f(w_center, h_center); // 거리 계산용 포인트 (예시로 (0,0) 사용)
        auto t1 = std::chrono::high_resolution_clock::now();
        yolo.detect(frame, objects, 640, 0.25f, 0.45f); // 추론 수행
        auto t2 = std::chrono::high_resolution_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "[MAIN] yolo.detect() inference time: " << ms << " ms" << std::endl;
        yolo.calc_distance(objects, center); // 거리 계산 (예시로 (0,0) 사용)
        cv::circle(frame, center, 15, cv::Scalar(0, 255, 0), -1); // 중앙 포인트 표시
        cv::Mat one_shot = yolo.draw_result(frame, objects); // 결과 이미지에 그리기
        std::vector<cv::Mat> results = yolo.crop_objects(frame, objects); // 결과 이미지에 그리기
        
        cv::imwrite("result.jpg", one_shot); // 결과 이미지 저장

        // 결과를 순회하면서 이미지 저장
        for (size_t i = 0; i < results.size(); ++i) {
            std::string filename = "lp_" + std::to_string(i) + ".jpg";
            cv::imwrite(filename, results[i]);
        }
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
