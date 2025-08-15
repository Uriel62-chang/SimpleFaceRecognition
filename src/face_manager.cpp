#include "face_manager.h"
#include "face_detection.h"
#include "face_recognition.h"
#include <filesystem>
#include <iostream>
#include <algorithm>

using namespace std::filesystem;

FaceManager::FaceManager() : initialized_(false) {
}

FaceManager::~FaceManager() {
}

bool FaceManager::initialize(const std::string& pictures_dir) {
    pictures_directory_ = pictures_dir;
    
    // 检查目录是否存在
    if (!exists(pictures_directory_)) {
        std::cerr << "[FaceManager] 错误：目录不存在: " << pictures_directory_ << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "[FaceManager] 初始化成功，目录: " << pictures_directory_ << std::endl;
    return true;
}

bool FaceManager::autoScanAndRegister() {
    if (!initialized_) {
        std::cerr << "[FaceManager] 错误：未初始化" << std::endl;
        return false;
    }
    
    std::cout << "[FaceManager] 开始自动扫描并注册人脸..." << std::endl;
    
    // 清空之前的数据
    known_features_.clear();
    known_labels_.clear();
    
    if (!autoScanPicturesDirectory()) {
        std::cerr << "[FaceManager] 自动扫描失败" << std::endl;
        return false;
    }
    
    std::cout << "[FaceManager] 自动扫描完成，成功注册 " << known_features_.size() << " 个人脸" << std::endl;
    return true;
}

const std::vector<cv::Mat>& FaceManager::getKnownFeatures() const {
    return known_features_;
}

const std::vector<std::string>& FaceManager::getKnownLabels() const {
    return known_labels_;
}

size_t FaceManager::getRegisteredCount() const {
    return known_features_.size();
}

bool FaceManager::isInitialized() const {
    return initialized_;
}

bool FaceManager::enrollFromImage(const std::string& img_path, 
                                 const std::string& label,
                                 cv::Mat& outFeatures) {
    cv::Mat img = cv::imread(img_path);
    if (img.empty()) {
        std::cerr << "[FaceManager] 无法读取图片: " << img_path << std::endl;
        return false;
    }
    
    auto faces = detectFaces(img);
    if (faces.empty()) {
        std::cerr << "[FaceManager] 图片中未检测到人脸: " << img_path << std::endl;
        return false;
    }
    
    auto descs = extract_face_features(img, {faces[0]});
    if (descs.size() != 1) {
        std::cerr << "[FaceManager] 特征提取失败: " << img_path << std::endl;
        return false;
    }
    
    outFeatures = descs[0];
    std::cout << "[FaceManager] 成功注册: " << label << " <- " << img_path << std::endl;
    return true;
}

bool FaceManager::autoScanPicturesDirectory() {
    // 支持的图片格式
    std::vector<std::string> supported_extensions = {".jpg", ".jpeg", ".png", ".bmp"};
    
    int total_files = 0;
    int success_count = 0;
    
    // 遍历目录中的所有文件
    for (const auto& entry : directory_iterator(pictures_directory_)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            std::string filepath = entry.path().string();
            
            // 检查文件扩展名
            std::string extension = entry.path().extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (std::find(supported_extensions.begin(), supported_extensions.end(), extension) != supported_extensions.end()) {
                total_files++;
                std::cout << "[FaceManager] 处理文件: " << filename << std::endl;
                
                // 从文件名提取标签（去掉扩展名）
                std::string label = entry.path().stem().string();
                
                // 尝试注册人脸
                cv::Mat features;
                if (enrollFromImage(filepath, label, features)) {
                    known_features_.push_back(features);
                    known_labels_.push_back(label);
                    success_count++;
                    std::cout << "[FaceManager] ✓ " << label << " 注册成功" << std::endl;
                } else {
                    std::cout << "[FaceManager] ✗ " << label << " 注册失败" << std::endl;
                }
            }
        }
    }
    
    std::cout << "\n[FaceManager] 扫描完成！" << std::endl;
    std::cout << "[FaceManager] 总文件数: " << total_files << std::endl;
    std::cout << "[FaceManager] 成功注册: " << success_count << std::endl;
    std::cout << "[FaceManager] 失败数量: " << (total_files - success_count) << std::endl;
    
    if (success_count > 0) {
        std::cout << "[FaceManager] 已注册的人脸:" << std::endl;
        for (size_t i = 0; i < known_labels_.size(); ++i) {
            std::cout << "  " << (i+1) << ". " << known_labels_[i] << std::endl;
        }
        return true;
    } else {
        std::cout << "[FaceManager] 警告：没有成功注册任何人脸！" << std::endl;
        return false;
    }
}
