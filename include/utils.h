#ifndef UTILS_H
#define UTILS_H

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace utils {

/**
 * @brief 获取模型文件的绝对路径
 * @param modelName 模型文件名
 * @return 模型文件的完整路径
 */
std::string getModelPath(const std::string& modelName);

/**
 * @brief 检查文件是否存在
 * @param filePath 文件路径
 * @return 文件是否存在
 */
bool fileExists(const std::string& filePath);

/**
 * @brief 获取项目根目录
 * @return 项目根目录的绝对路径
 */
std::string getProjectRoot();

/**
 * @brief 图像预处理：调整大小、归一化等
 * @param input 输入图像
 * @param targetSize 目标尺寸
 * @return 预处理后的图像
 */
cv::Mat preprocessImage(const cv::Mat& input, const cv::Size& targetSize = cv::Size(160, 160));

/**
 * @brief 计算两个特征向量的余弦相似度
 * @param vec1 特征向量1
 * @param vec2 特征向量2
 * @return 相似度分数 (0-1)
 */
double cosineSimilarity(const cv::Mat& vec1, const cv::Mat& vec2);

/**
 * @brief 计算两个特征向量的欧几里得距离
 * @param vec1 特征向量1
 * @param vec2 特征向量2
 * @return 距离值
 */
double euclideanDistance(const cv::Mat& vec1, const cv::Mat& vec2);

} // namespace utils

#endif // UTILS_H
