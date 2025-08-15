#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class RecognitionEngine {
public:
    RecognitionEngine();
    ~RecognitionEngine();

    // 初始化识别引擎
    bool initialize();
    
    // 处理单帧图像
    std::vector<std::pair<cv::Rect, std::string>> processFrame(
        const cv::Mat& frame,
        const std::vector<cv::Mat>& known_features,
        const std::vector<std::string>& known_labels);
    
    // 绘制识别结果
    void drawResults(cv::Mat& frame, 
                    const std::vector<std::pair<cv::Rect, std::string>>& results);

private:
    // 人脸检测
    std::vector<cv::Rect> detectFaces(const cv::Mat& frame);
    
    // 特征提取
    std::vector<cv::Mat> extractFeatures(const cv::Mat& frame, 
                                        const std::vector<cv::Rect>& faces);
    
    // 人脸匹配
    std::string matchFace(const cv::Mat& features,
                          const std::vector<cv::Mat>& known_features,
                          const std::vector<std::string>& known_labels);
    
    // 绘制标签
    void drawLabel(cv::Mat& frame, const cv::Rect& rect, const std::string& text);

private:
    bool initialized_;
};
