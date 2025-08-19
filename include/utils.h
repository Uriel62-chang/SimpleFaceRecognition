#pragma once

#include <opencv2/core.hpp>
#include <string>

namespace utils {

// 获取项目根目录
std::string getProjectRoot();

// 获取模型文件路径
std::string getModelPath(const std::string& modelName);

// 获取pictures目录路径
std::string getPicturesDirectory();

// 检查文件是否存在
bool fileExists(const std::string& filePath);

// 图像预处理
cv::Mat preprocessImage(const cv::Mat& input, const cv::Size& targetSize);

// 计算余弦相似度
double cosineSimilarity(const cv::Mat& vec1, const cv::Mat& vec2);

// 计算欧几里得距离
double euclideanDistance(const cv::Mat& vec1, const cv::Mat& vec2);

} // namespace utils
