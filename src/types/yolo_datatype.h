

#ifndef RK3588_DEMO_NN_DATATYPE_H
#define RK3588_DEMO_NN_DATATYPE_H

#include <opencv2/opencv.hpp>

typedef struct _nn_object_s {
    float x;
    float y;
    float w;
    float h;
    float score;
    int class_id;
} nn_object_s;

struct ObbDetection
{
    int class_id{0};
    std::string className{};
    float confidence{0.0f};
    cv::Scalar color{};

    float poly[8]{};   // x1 y1 x2 y2 x3 y3 x4 y4

    float cx, cy, w, h, theta;
};

#endif //RK3588_DEMO_NN_DATATYPE_H
