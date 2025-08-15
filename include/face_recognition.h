#ifndef FACE_RECOGNITION_H
#define FACE_RECOGNITION_H

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class FaceRecognition {
public:
    FaceRecognition();
    ~FaceRecognition();
    
    // 初始化系统
    bool initialize();
    
    // 检查系统是否已初始化
    bool isInitialized() const;
    
    // 特征提取
    cv::Mat extractFaceFeatures(const cv::Mat& faceImage);
    std::vector<cv::Mat> extractFaceFeatures(const cv::Mat& frame, const std::vector<cv::Rect>& faces);
    
    // 人脸比较
    bool compareFaces(const cv::Mat& face1, const cv::Mat& face2, double threshold = 0.9);

private:
    // 系统状态
    bool initialized;
    
    // 配置参数
    cv::Size inputSize;
    int featureDimension;
    
    // 私有方法
    cv::Mat preprocessFace(const cv::Mat& face);
    cv::Mat extractSimpleFeatures(const cv::Mat& processed);
};

// 便捷函数声明
bool load_model();
std::vector<cv::Mat> extract_face_features(const cv::Mat& frame, const std::vector<cv::Rect>& faces);
double compare_faces(const cv::Mat& face1, const cv::Mat& face2, double threshold = 0.9);

#endif
