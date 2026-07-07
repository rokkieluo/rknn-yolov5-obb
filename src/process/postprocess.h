#ifndef RK3588_DEMO_POSTPROCESS_YOLOV5_OBB_H
#define RK3588_DEMO_POSTPROCESS_YOLOV5_OBB_H

#include <vector>
#include <cstdint>
#include "types/yolo_datatype.h"
#include <opencv2/imgproc.hpp>
/*
 * YOLOv5-OBB 后处理接口
 *
 * 模型特性：
 *  - 4 个输出头
 *  - 输出维度：1 x 187 x H x W
 *  - na = 1
 *  - theta_bins = 180
 *  - OBB 使用 poly[8] 表示
 */

/**
 * @brief YOLOv5-OBB 后处理主入口
 *
 * @param outputs       4 个输出头指针（int8）
 * @param model_h       网络输入高度（如 640）
 * @param model_w       网络输入宽度（如 640）
 * @param conf_thresh   置信度阈值（如 0.25）
 * @param nms_thresh    NMS 阈值（如 0.45）
 * @param zps           每个输出头的 zero point（长度=4）
 * @param scales        每个输出头的 scale（长度=4）
 * @param final_dets    输出的 OBB 检测结果
 */
void postprocess_yolov5_obb(
    int8_t* outputs[4],
    int model_h,
    int model_w,
    float conf_thresh,
    float nms_thresh,
    const int* zps,
    const float* scales,
    std::vector<ObbDetection>& final_dets
);
void rbox_to_poly(float cx, float cy, float w, float h, float theta, float poly[8]);

#endif // RK3588_DEMO_POSTPROCESS_YOLOV5_OBB_H
