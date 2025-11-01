#include "utils.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace utils {

std::string getProjectRoot() {
    // 获取当前工作目录
    fs::path currentPath = fs::current_path();
    
    // 如果在 build 目录下，向上查找项目根目录
    fs::path searchPath = currentPath;
    std::string searchPathStr = searchPath.string();
    
    if (searchPathStr.find("build") != std::string::npos) {
        // 找到包含 "build" 的路径，向上查找项目根目录
        while (searchPath.has_parent_path() && searchPath.string().find("build") != std::string::npos) {
            searchPath = searchPath.parent_path();
        }
        // 如果仍在build目录中，再向上查找一级
        if (searchPath.filename().string() == "build" || searchPath.string().find("build") != std::string::npos) {
            searchPath = searchPath.parent_path();
        }
    }
    
    // 验证是否为项目根目录：检查是否存在 models 和 pictures 目录
    fs::path modelsDir = searchPath / "models";
    fs::path picturesDir = searchPath / "pictures";
    
    if (fs::exists(modelsDir) && fs::is_directory(modelsDir) &&
        fs::exists(picturesDir) && fs::is_directory(picturesDir)) {
        return searchPath.string();
    }
    
    // 如果当前路径不是项目根目录，向上查找
    fs::path parentPath = searchPath;
    for (int i = 0; i < 5 && parentPath.has_parent_path(); ++i) {
        parentPath = parentPath.parent_path();
        modelsDir = parentPath / "models";
        picturesDir = parentPath / "pictures";
        
        if (fs::exists(modelsDir) && fs::is_directory(modelsDir) &&
            fs::exists(picturesDir) && fs::is_directory(picturesDir)) {
            return parentPath.string();
        }
    }
    
    // 如果都找不到，返回当前路径（向后兼容）
    return currentPath.string();
}

std::string getModelPath(const std::string& modelName) {
    std::string projectRoot = getProjectRoot();
    
    // 使用filesystem::path进行跨平台路径拼接
    fs::path modelPath = fs::path(projectRoot) / "models" / modelName;
    
    // 检查文件是否存在
    if (fileExists(modelPath.string())) {
        return modelPath.string();
    }
    
    // 如果不存在，尝试其他可能的路径
    std::vector<fs::path> possiblePaths = {
        fs::path(projectRoot) / "models" / modelName,
        fs::path(projectRoot) / ".." / "models" / modelName,
        fs::path(projectRoot) / ".." / ".." / "models" / modelName,
        fs::path("models") / modelName,
        fs::path("..") / "models" / modelName
    };
    
    for (const auto& path : possiblePaths) {
        if (fileExists(path.string())) {
            return path.string();
        }
    }
    
    // 如果都找不到，返回原始路径
    std::cerr << "[Utils] Warning: Model file not found: " << modelName << std::endl;
    return modelPath.string();
}

std::string getPicturesDirectory() {
    std::string projectRoot = getProjectRoot();
    
    // 使用filesystem::path进行跨平台路径拼接
    fs::path picturesPath = fs::path(projectRoot) / "pictures";
    
    // 检查目录是否存在
    if (fs::exists(picturesPath) && fs::is_directory(picturesPath)) {
        return picturesPath.string();
    }
    
    // 如果不存在，尝试其他可能的路径
    std::vector<fs::path> possiblePaths = {
        fs::path(projectRoot) / "pictures",
        fs::path(projectRoot) / ".." / "pictures",
        fs::path(projectRoot) / ".." / ".." / "pictures",
        fs::path("pictures"),
        fs::path("..") / "pictures"
    };
    
    for (const auto& path : possiblePaths) {
        if (fs::exists(path) && fs::is_directory(path)) {
            return path.string();
        }
    }
    
    // 如果都找不到，返回默认路径
    std::cerr << "[Utils] Warning: Pictures directory not found, using default path" << std::endl;
    return picturesPath.string();
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
