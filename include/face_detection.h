#ifndef FACE_DETECTION_H
#define FACE_DETECTION_H

#include <opencv2/opencv.hpp>

// 人脸检测函数声明
std::vector<cv::Rect> detectFaces(const cv::Mat& frame);

#endif
