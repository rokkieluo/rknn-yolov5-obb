

#ifndef RK3588_DEMO_CV_DRAW_H
#define RK3588_DEMO_CV_DRAW_H

#include <opencv2/opencv.hpp>

#include "types/yolo_datatype.h"

// draw detections on img
cv::Scalar get_color(int class_id);

void DrawObbDetections(cv::Mat& image, const std::vector<ObbDetection>& detections);

#endif //RK3588_DEMO_CV_DRAW_H
