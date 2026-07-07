#include <opencv2/opencv.hpp>
#include <chrono>              // 用于计时
#include <vector>
#include <cstdio>
#include <cmath>
#include "task/yolov5.h"
#include "utils/logging.h"
#include "draw/cv_draw.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        printf("Usage: %s <model_file> <img_file> [loop_count]\n", argv[0]);
        return -1;
    }

    const char *model_file = argv[1];
    const char *img_file = argv[2];
    
    // 可选参数：循环次数，默认 100 次
    int loop_count = 1;
    if (argc >= 4) {
        loop_count = std::atoi(argv[3]);
        if (loop_count <= 0) loop_count = 1;
    }

    // 读取图片（只读一次）
    cv::Mat img = cv::imread(img_file);
    if (img.empty()) {
        printf("Error: cannot read image %s\n", img_file);
        return -1;
    }
    NN_LOG_INFO("img size: %d x %d", img.cols, img.rows);

    // 初始化和加载模型（只做一次）
    Yolov5 yolo;
    if (yolo.LoadModel(model_file) != NN_SUCCESS) {
        NN_LOG_ERROR("Failed to load model");
        return -1;
    }

    // 用于保存结果（可选，只保留最后一次或不保存）
    std::vector<ObbDetection> objects;

    // 计时开始
    auto start_time = std::chrono::high_resolution_clock::now();

    // 循环推理多次
    for (int i = 0; i < loop_count; ++i) {
        objects.clear();  // 每次清空，避免内存累积
        yolo.Run(img, objects);

        // 可选：打印进度（大量循环时有用）
        if ((i + 1) % 10 == 0) {
            printf("Completed %d / %d inferences\n", i + 1, loop_count);
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();

    // 计算耗时
    auto total_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    double total_ms = total_us / 1000.0;
    double avg_ms = total_ms / loop_count;

    printf("\n");
    printf("=== 推理性能统计 ===\n");
    printf("循环次数: %d\n", loop_count);
    printf("总耗时: %.2f ms\n", total_ms);
    printf("平均单次耗时: %.2f ms\n", avg_ms);
    printf("FPS (帧率): %.2f fps\n", 1000.0 / avg_ms);

    // 最后一次的结果用于可视化（可选）
    printf("最后一次检测到目标数量: %zu\n", objects.size());
    for (size_t i = 0; i < objects.size(); ++i) {
        float theta_rad = objects[i].theta;

        // 1. 转成度数
        float theta_deg = theta_rad * 180.0f / static_cast<float>(M_PI);

        // 2. 规范化到 [-180, 180]
        while (theta_deg > 180.0f)  theta_deg -= 360.0f;
        while (theta_deg <= -180.0f) theta_deg += 360.0f;
        
        printf("目标%d: 类别=%d, 置信度=%.2f, 角度=%.2f\n", 
               (int)i, objects[i].class_id, objects[i].confidence, theta_deg);
    }

    // 显示和保存最后一次结果
    DrawObbDetections(img, objects);
    cv::imwrite("result.jpg", img);

    return 0;
}