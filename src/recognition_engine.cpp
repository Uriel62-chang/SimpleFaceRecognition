#include "recognition_engine.h"
#include "face_detection.h"
#include "face_recognition.h"
#include <iostream>
#include <iomanip> // Required for std::fixed and std::setprecision

RecognitionEngine::RecognitionEngine() : initialized_(false) {
}

RecognitionEngine::~RecognitionEngine() {
}

bool RecognitionEngine::initialize() {
    // 这里可以添加额外的初始化逻辑
    initialized_ = true;
    std::cout << "[RecognitionEngine] 初始化成功" << std::endl;
    return true;
}

std::vector<std::pair<cv::Rect, std::string>> RecognitionEngine::processFrame(
    const cv::Mat& frame,
    const std::vector<cv::Mat>& known_features,
    const std::vector<std::string>& known_labels) {
    
    std::vector<std::pair<cv::Rect, std::string>> results;
    
    if (!initialized_) {
        std::cerr << "[RecognitionEngine] 错误：未初始化" << std::endl;
        return results;
    }
    
    // 1. 人脸检测
    auto faces = detectFaces(frame);
    if (faces.empty()) {
        return results;
    }
    
    // 2. 特征提取
    auto features = extractFeatures(frame, faces);
    if (features.size() != faces.size()) {
        std::cerr << "[RecognitionEngine] 特征提取数量不匹配" << std::endl;
        return results;
    }
    
    // 3. 人脸匹配
    for (size_t i = 0; i < faces.size(); ++i) {
        std::string name = matchFace(features[i], known_features, known_labels);
        results.push_back({faces[i], name});
    }
    
    return results;
}

void RecognitionEngine::drawResults(cv::Mat& frame, 
                                   const std::vector<std::pair<cv::Rect, std::string>>& results) {
    for (const auto& result : results) {
        const cv::Rect& rect = result.first;
        const std::string& name = result.second;
        
        // 绘制人脸框
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        
        // 绘制标签
        drawLabel(frame, rect, name);
    }
}

std::vector<cv::Rect> RecognitionEngine::detectFaces(const cv::Mat& frame) {
    return ::detectFaces(frame);
}

std::vector<cv::Mat> RecognitionEngine::extractFeatures(const cv::Mat& frame, 
                                                       const std::vector<cv::Rect>& faces) {
    return ::extract_face_features(frame, faces);
}

std::string RecognitionEngine::matchFace(const cv::Mat& features,
                                         const std::vector<cv::Mat>& known_features,
                                         const std::vector<std::string>& known_labels) {
    if (known_features.empty() || known_labels.empty()) {
        return "Unknown";
    }
    
    std::string best_match = "Unknown";
    double best_similarity = -1.0;
    int best_index = -1;
    
    // 计算所有相似度并找到最佳匹配
    for (size_t i = 0; i < known_features.size(); ++i) {
        double similarity = ::compare_faces(features, known_features[i]);
        
        if (similarity > best_similarity) {
            best_similarity = similarity;
            best_match = known_labels[i];
            best_index = i;
        }
    }
    
    // 多阈值动态匹配策略
    double final_threshold = 0.6; // 默认阈值
    
    if (best_similarity >= 0.95) {
        // 极高相似度：使用严格阈值
        final_threshold = 0.90;
        std::cout << "[RecognitionEngine] 极高相似度: " << best_match 
                  << " (" << std::fixed << std::setprecision(3) << best_similarity 
                  << "), 阈值: " << final_threshold << std::endl;
    }
    else if (best_similarity >= 0.85) {
        // 高相似度：使用较高阈值
        final_threshold = 0.75;
        std::cout << "[RecognitionEngine] 高相似度: " << best_match 
                  << " (" << std::fixed << std::setprecision(3) << best_similarity 
                  << "), 阈值: " << final_threshold << std::endl;
    }
    else if (best_similarity >= 0.75) {
        // 中等相似度：使用中等阈值
        final_threshold = 0.65;
        std::cout << "[RecognitionEngine] 中等相似度: " << best_match 
                  << " (" << std::fixed << std::setprecision(3) << best_similarity 
                  << "), 阈值: " << final_threshold << std::endl;
    }
    else {
        // 低相似度：使用宽松阈值但标记为低置信度
        final_threshold = 0.60;
        std::cout << "[RecognitionEngine] 低相似度: " << best_match 
                  << " (" << std::fixed << std::setprecision(3) << best_similarity 
                  << "), 阈值: " << final_threshold << std::endl;
    }
    
    // 应用最终阈值判断
    if (best_similarity >= final_threshold) {
        // 额外检查：如果相似度接近阈值，进行二次验证
        if (best_similarity < final_threshold + 0.05) {
            std::cout << "[RecognitionEngine] 低置信度匹配，建议二次验证" << std::endl;
        }
        return best_match;
    } else {
        std::cout << "[RecognitionEngine] 相似度低于阈值，标记为Unknown" << std::endl;
        return "Unknown";
    }
}

void RecognitionEngine::drawLabel(cv::Mat& frame, const cv::Rect& rect, const std::string& text) {
    int base = 0;
    cv::Size sz = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, &base);
    
    cv::Rect bg(rect.x, std::max(0, rect.y - sz.height - 8), 
                sz.width + 8, sz.height + 8);
    
    cv::rectangle(frame, bg, cv::Scalar(0, 0, 0), cv::FILLED);
    cv::putText(frame, text, cv::Point(bg.x + 4, bg.y + bg.height - 6),
                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
}
