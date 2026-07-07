#include <vector>
#include <cmath>
#include <cstring>
#include <algorithm>
#include "types/yolo_datatype.h"

/* ================== 模型参数 ================== */

#define NUM_CLASSES 2
#define NUM_HEADS   4
#define THETA_BINS  180

static const int STRIDES[NUM_HEADS] = {2, 4, 8, 16};

static const float ANCHORS[NUM_HEADS][2] = {
    {2.49609375f, 2.498046875f},
    {3.404296875f, 1.4990234375f},
    {2.974609375f, 1.0205078125f},
    {1.0908203125f, 1.09375f}
};

static const char* CLASS_NAMES[NUM_CLASSES] = {
    "class0", "class1"
};

/* ================== 工具函数 ================== */

static inline float sigmoid(float x)
{
    return 1.f / (1.f + expf(-x));
}

static inline float dequant(int8_t v, int zp, float scale)
{
    return ((float)v - zp) * scale;
}

/* cx,cy,w,h,theta → poly[8] */
void rbox_to_poly(float cx, float cy, float w, float h, float theta, float poly[8]) {
    // 标准化（匹配 Python）
    if (w < h) {
        std::swap(w, h);
        theta += M_PI / 2.0f;
    }
    theta = fmod(theta + 2 * M_PI, M_PI);

    float cos_t = cosf(theta);
    float sin_t = sinf(theta);
    float vec1_x = w / 2.0f * cos_t;
    float vec1_y = w / 2.0f * sin_t;
    float vec2_x = -h / 2.0f * sin_t;  // 负sin匹配水平
    float vec2_y = h / 2.0f * cos_t;

    poly[0] = cx + vec1_x + vec2_x;  // pt1
    poly[1] = cy + vec1_y + vec2_y;
    poly[2] = cx + vec1_x - vec2_x;  // pt2
    poly[3] = cy + vec1_y - vec2_y;
    poly[4] = cx - vec1_x - vec2_x;  // pt3
    poly[5] = cy - vec1_y - vec2_y;
    poly[6] = cx - vec1_x + vec2_x;  // pt4
    poly[7] = cy - vec1_y + vec2_y;


}

/* ================== 单 Head 解码 ================== */

static void decode_head(
    int8_t* input,
    int grid_h,
    int grid_w,
    int stride,
    float anchor_w,
    float anchor_h,
    int zp,
    float scale,
    float conf_thresh,
    std::vector<ObbDetection>& results)
{
    int grid_size = grid_h * grid_w;
    int no = 5 + NUM_CLASSES + THETA_BINS;  // 通道总数：5 (xywho) + classes + theta bins

    for (int i = 0; i < grid_h; ++i)
    {
        for (int j = 0; j < grid_w; ++j)
        {
            int idx = (i * grid_w + j);  // 当前 grid cell 的起始偏移

            // 对象置信度
            float obj = sigmoid(dequant(input[idx + 4 * grid_size], zp, scale));
            if (obj < conf_thresh) continue;

            // 中心点解码
            float cx = (sigmoid(dequant(input[idx + 0 * grid_size], zp, scale)) * 2.0f - 0.5f + j) * stride;
            float cy = (sigmoid(dequant(input[idx + 1 * grid_size], zp, scale)) * 2.0f - 0.5f + i) * stride;

            float raw_w = dequant(input[idx + 2 * grid_size], zp, scale);
            float w = powf(sigmoid(raw_w) * 2.0f, 2.0f) * anchor_w * (float)stride;

            float raw_h = dequant(input[idx + 3 * grid_size], zp, scale);
            float h = powf(sigmoid(raw_h) * 2.0f, 2.0f) * anchor_h * (float)stride;

            // 类别分数（取最大）
            int best_cls = 0;
            float best_cls_score = 0.f;
            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                float p = sigmoid(dequant(input[idx + (5 + c) * grid_size], zp, scale));
                if (p > best_cls_score)
                {
                    best_cls_score = p;
                    best_cls = c;
                }
            }

            // 初步分数
            float score = obj * best_cls_score;
            if (score < conf_thresh) continue;

            int theta_offset = idx + (5 + NUM_CLASSES) * grid_size;
            float max_theta = -FLT_MAX;
            int best_bin = 0;
            for (int t = 0; t < THETA_BINS; ++t) {
                float val = sigmoid(dequant(input[theta_offset + t * grid_size], zp, scale));
                if (val > max_theta) {
                    max_theta = val;
                    best_bin = t;
                }
            }
            float theta = ((float)best_bin - 90.0f) * M_PI / 180.0f;
            theta = -theta;

            float final_score = score;
            if (final_score < conf_thresh) continue;

            // 构造检测结果
            ObbDetection det;
            det.class_id    = best_cls;
            det.className   = CLASS_NAMES[best_cls];
            det.confidence  = final_score;
            det.cx          = cx;
            det.cy          = cy;
            det.w           = w;
            det.h           = h;
            det.theta       = theta;

            // 生成多边形
            rbox_to_poly(cx, cy, w, h, theta, det.poly);

            results.push_back(det);
        }
    }
}

/* ================== 多边形 IoU（用于 NMS） ================== */

static float poly_area(const float p[8])
{
    float area = 0.f;
    for (int i = 0; i < 4; ++i)
    {
        int j = (i + 1) % 4;
        area += p[2*i] * p[2*j + 1] - p[2*j] * p[2*i + 1];
    }
    return fabs(area) * 0.5f;
}

/* 这里用简化 NMS（按 bbox 包围盒） */
static bool iou_poly_simple(const float a[8], const float b[8], float thresh)
{
    cv::Point2f pts_a[4] = {
        cv::Point2f(a[0], a[1]), cv::Point2f(a[2], a[3]),
        cv::Point2f(a[4], a[5]), cv::Point2f(a[6], a[7])
    };
    cv::Point2f pts_b[4] = {
        cv::Point2f(b[0], b[1]), cv::Point2f(b[2], b[3]),
        cv::Point2f(b[4], b[5]), cv::Point2f(b[6], b[7])
    };

    cv::RotatedRect rect_a = cv::minAreaRect(std::vector<cv::Point2f>(pts_a, pts_a + 4));
    cv::RotatedRect rect_b = cv::minAreaRect(std::vector<cv::Point2f>(pts_b, pts_b + 4));

    std::vector<cv::Point2f> inter_pts;
    cv::rotatedRectangleIntersection(rect_a, rect_b, inter_pts);

    if (inter_pts.empty()) return false;

    float inter_area = cv::contourArea(inter_pts);
    float area_a = rect_a.size.width * rect_a.size.height;
    float area_b = rect_b.size.width * rect_b.size.height;

    float iou = inter_area / (area_a + area_b - inter_area + 1e-6f);
    return iou > thresh;
}

/* ================== 主入口 ================== */

void postprocess_yolov5_obb(
    int8_t* outputs[NUM_HEADS],
    int model_h,
    int model_w,
    float conf_thresh,
    float nms_thresh,
    const int* zps,
    const float* scales,
    std::vector<ObbDetection>& final_dets)
{
    std::vector<ObbDetection> dets;

    for (int i = 0; i < NUM_HEADS; ++i)
    {
        decode_head(
            outputs[i],
            model_h / STRIDES[i],
            model_w / STRIDES[i],
            STRIDES[i],
            ANCHORS[i][0],
            ANCHORS[i][1],
            zps[i],
            scales[i],
            conf_thresh,
            dets);
    }

    std::sort(dets.begin(), dets.end(),
        [](const ObbDetection& a, const ObbDetection& b) {
            return a.confidence > b.confidence;
        });

    std::vector<bool> removed(dets.size(), false);

    for (size_t i = 0; i < dets.size(); ++i) {
        if (removed[i]) continue;
        final_dets.push_back(dets[i]);
        for (size_t j = i + 1; j < dets.size(); ++j) {
            if (removed[j]) continue;
            if (dets[i].class_id == dets[j].class_id && iou_poly_simple(dets[i].poly, dets[j].poly, nms_thresh))
                removed[j] = true;
        }
    }
}
