#include "utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace utils {

std::string getProjectRoot() {
    // 获取当前可执行文件所在目录
    std::string exePath = fs::current_path().string();
    
    // 如果在 build 目录下，向上查找项目根目录
    if (exePath.find("build") != std::string::npos) {
        size_t buildPos = exePath.find("build");
        return exePath.substr(0, buildPos);
    }
    
    return exePath;
}

std::string getModelPath(const std::string& modelName) {
    std::string projectRoot = getProjectRoot();
    std::string modelPath = projectRoot + "models\\" + modelName;
    
    // 检查文件是否存在
    if (fileExists(modelPath)) {
        return modelPath;
    }
    
    // 如果不存在，尝试其他可能的路径
    std::vector<std::string> possiblePaths = {
        projectRoot + "models\\" + modelName,
        projectRoot + "..\\models\\" + modelName,
        projectRoot + "..\\..\\models\\" + modelName,
        "models\\" + modelName,
        "..\\models\\" + modelName
    };
    
    for (const auto& path : possiblePaths) {
        if (fileExists(path)) {
            return path;
        }
    }
    
    // 如果都找不到，返回原始路径
    std::cerr << "[Utils] Warning: Model file not found: " << modelName << std::endl;
    return modelPath;
}

bool fileExists(const std::string& filePath) {
    return fs::exists(filePath);
}

cv::Mat preprocessImage(const cv::Mat& input, const cv::Size& targetSize) {
    cv::Mat processed;
    
    // 调整图像大小
    cv::resize(input, processed, targetSize);
    
    // 转换为浮点数并归一化到 [0, 1]
    processed.convertTo(processed, CV_32F);
    processed /= 255.0;
    
    // 如果输入是 BGR，转换为 RGB
    if (processed.channels() == 3) {
        cv::cvtColor(processed, processed, cv::COLOR_BGR2RGB);
    }
    
    return processed;
}

double cosineSimilarity(const cv::Mat& vec1, const cv::Mat& vec2) {
    if (vec1.empty() || vec2.empty() || vec1.size() != vec2.size()) {
        return 0.0;
    }
    
    // 确保向量是连续的
    cv::Mat v1 = vec1.reshape(1, 1);
    cv::Mat v2 = vec2.reshape(1, 1);
    
    // 计算点积
    double dotProduct = cv::sum(v1.mul(v2))[0];
    
    // 计算向量的模
    double norm1 = cv::norm(v1);
    double norm2 = cv::norm(v2);
    
    // 避免除零
    if (norm1 < 1e-10 || norm2 < 1e-10) {
        return 0.0;
    }
    
    return dotProduct / (norm1 * norm2);
}

double euclideanDistance(const cv::Mat& vec1, const cv::Mat& vec2) {
    if (vec1.empty() || vec2.empty() || vec1.size() != vec2.size()) {
        return std::numeric_limits<double>::max();
    }
    
    // 确保向量是连续的
    cv::Mat v1 = vec1.reshape(1, 1);
    cv::Mat v2 = vec2.reshape(1, 1);
    
    // 计算欧几里得距离
    cv::Mat diff = v1 - v2;
    return cv::norm(diff);
}

} // namespace utils
