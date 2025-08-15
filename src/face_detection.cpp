#include "face_detection.h"
#include "utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>

using namespace cv;
using namespace std;

std::vector<cv::Rect> detectFaces(const Mat& frame) {
    CascadeClassifier face_cascade;
    
    // 使用工具函数获取模型路径
    std::string haar_path = ::utils::getModelPath("haarcascade_frontalface_default.xml");
    
    // 加载人脸 Haar 级联分类器
    if (!face_cascade.load(haar_path)) {
        cerr << "[FaceDet] 错误：无法加载人脸 Haar 级联分类器: " << haar_path << endl;
        return {};
    }
    
    // 转换为灰度图像
    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    
    // 直方图均衡化，提高检测效果
    equalizeHist(gray, gray);
    
    // 检测人脸
    vector<Rect> faces;
    face_cascade.detectMultiScale(
        gray,           // 输入图像
        faces,          // 输出人脸区域
        1.1,            // 缩放因子
        3,              // 最小邻居数
        0,              // 标志
        Size(30, 30)    // 最小人脸尺寸
    );
    
    // 输出检测结果
    if (!faces.empty()) {
        cout << "[FaceDet] 检测到 " << faces.size() << " 个人脸" << endl;
    }
    
    return faces;
}