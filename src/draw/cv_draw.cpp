#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <algorithm>
#include "types/yolo_datatype.h"

// 生成颜色
cv::Scalar get_color(int class_id)
{
    cv::RNG rng(class_id);  // 使用 class_id 作为种子，和 Python seed 效果相同
    return cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)); // BGR
}

// 绘制旋转框和标签
void DrawObbDetections(cv::Mat& image, const std::vector<ObbDetection>& detections)
{
    
    const int thickness = 2;          // 线条粗细
    const double font_scale = 0.6;    // 字体大小
    const int font_thickness = 2;     // 文字粗细
    const int font = cv::FONT_HERSHEY_SIMPLEX;

    for (size_t i = 0; i < detections.size(); ++i)
    {
        const auto& det = detections[i];
        
        // 颜色（优先使用 det.color，否则按 class_id 生成）
        cv::Scalar color = (det.color == cv::Scalar(0,0,0)) ? get_color(det.class_id) : det.color;

        // poly 8 点 -> 4 个 cv::Point
        cv::Point pts[4];
        for (int j = 0; j < 4; j++)
        {
            pts[j] = cv::Point(static_cast<int>(det.poly[j*2]), static_cast<int>(det.poly[j*2 + 1]));
        }

        // 绘制旋转框（闭合，抗锯齿）
        const cv::Point* ppt[1] = { pts };
        int npt[] = { 4 };
        cv::polylines(image, ppt, npt, 1, true, color, thickness, cv::LINE_AA);

        // 计算多边形的最小外接矩形左上角（用于标签位置）
        int x_min = pts[0].x, y_min = pts[0].y;
        for (int j = 1; j < 4; j++) {
            x_min = std::min(x_min, pts[j].x);
            y_min = std::min(y_min, pts[j].y);
        }

        // 标签文本
        std::string label;
        if (!det.className.empty() && det.class_id < 10) {  // 假设类别不多
            label = cv::format("%s: %.2f", det.className.c_str(), det.confidence);
        } else {
            label = cv::format("cls%d: %.2f", det.class_id, det.confidence);
        }

        // 获取文本大小
        int baseLine = 0;
        cv::Size labelSize = cv::getTextSize(label, font, font_scale, font_thickness, &baseLine);

        // 标签位置：放在左上角上方（模仿 Python label_y = max(y_min - 10, ...)）
        int label_x = x_min;
        int label_y = std::max(y_min - 10, labelSize.height + 5);

        // 背景矩形（半透明效果用不到，但保留纯色背景）
        cv::rectangle(image,
                      cv::Point(label_x, label_y - labelSize.height - baseLine - 5),
                      cv::Point(label_x + labelSize.width + 5, label_y + baseLine),
                      color, cv::FILLED);  // 实心背景，使用检测框颜色

        // 绘制白色文本
        cv::putText(image, label,
                    cv::Point(label_x + 2, label_y - 5),
                    font, font_scale,
                    cv::Scalar(255, 255, 255),  // 白色文字
                    font_thickness, cv::LINE_AA);

    }
    
}