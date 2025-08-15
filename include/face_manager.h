#pragma once

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

class FaceManager {
public:
    FaceManager();
    ~FaceManager();

    // 初始化人脸管理器
    bool initialize(const std::string& pictures_dir);
    
    // 自动扫描并注册pictures目录中的人脸
    bool autoScanAndRegister();
    
    // 获取已注册的人脸特征
    const std::vector<cv::Mat>& getKnownFeatures() const;
    
    // 获取已注册的人脸标签
    const std::vector<std::string>& getKnownLabels() const;
    
    // 获取注册的人脸数量
    size_t getRegisteredCount() const;
    
    // 检查是否已初始化
    bool isInitialized() const;

private:
    // 从单张图片注册人脸
    bool enrollFromImage(const std::string& img_path, 
                        const std::string& label,
                        cv::Mat& outFeatures);
    
    // 自动扫描pictures目录
    bool autoScanPicturesDirectory();

private:
    std::string pictures_directory_;
    std::vector<cv::Mat> known_features_;
    std::vector<std::string> known_labels_;
    bool initialized_;
};
