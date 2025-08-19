#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

#include "face_manager.h"
#include "recognition_engine.h"
#include "face_recognition.h"  // 添加这个来获取load_model函数
#include "utils.h"  // 添加utils头文件

using namespace cv;
using namespace std;

int main()
{
    cout << "=== 人脸识别系统 ===" << endl;
    
    // 1) 初始化人脸识别模型
    if (!load_model()) {
        cerr << "错误：无法加载人脸识别模型" << endl;
        return -1;
    }
    cout << "✓ 模型加载成功！" << endl;

    // 2) 初始化人脸管理器 - 使用相对路径
    FaceManager faceManager;
    std::string picturesDir = ::utils::getPicturesDirectory();
    if (!faceManager.initialize(picturesDir)) {
        cerr << "错误：无法初始化人脸管理器" << endl;
        return -1;
    }
    
    // 3) 自动扫描并注册pictures目录中的人脸
    if (!faceManager.autoScanAndRegister()) {
        cerr << "错误：自动扫描pictures目录失败！" << endl;
        return -1;
    }
    
    cout << "[Main] 成功注册了 " << faceManager.getRegisteredCount() << " 个人脸" << endl;
    cout << "[Main] 开始实时人脸识别..." << endl;
    cout << "[Main] 按ESC键退出程序" << endl;

    // 4) 初始化识别引擎
    RecognitionEngine recognitionEngine;
    if (!recognitionEngine.initialize()) {
        cerr << "错误：无法初始化识别引擎" << endl;
        return -1;
    }

    // 5) 打开摄像头
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "错误：无法打开摄像头！" << endl;
        return -1;
    }
    namedWindow("Face Recognition", WINDOW_NORMAL);

    // 6) 主循环：检测 -> 识别 -> 显示
    while (true) {
        Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        // 处理当前帧
        auto results = recognitionEngine.processFrame(
            frame, 
            faceManager.getKnownFeatures(), 
            faceManager.getKnownLabels()
        );
        
        // 绘制识别结果
        recognitionEngine.drawResults(frame, results);

        // 显示结果
        imshow("Face Recognition", frame);
        
        // ESC 退出
        if (waitKey(10) == 27) break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
