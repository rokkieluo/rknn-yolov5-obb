#include "yolov5.h"

#include <memory>

#include "utils/logging.h"
#include "process/preprocess.h"
#include "types/yolo_datatype.h"

#include <ctime>


// 构造函数
Yolov5::Yolov5()
{
    engine_ = CreateRKNNEngine();
    input_tensor_.data = nullptr;
    original_img_width_ = 0;
    original_img_height_ = 0;
}
// 析构函数
Yolov5::~Yolov5()
{
    if (input_tensor_.data != nullptr)
    {
        free(input_tensor_.data);
        input_tensor_.data = nullptr;
    }
    for (auto &tensor : output_tensors_)
    {
        if (tensor.data != nullptr) {
            free(tensor.data);
            tensor.data = nullptr;
        }
    }
}

// 加载模型，获取输入输出属性
nn_error_e Yolov5::LoadModel(const char *model_path)
{
    auto ret = engine_->LoadModelFile(model_path);
    if (ret != NN_SUCCESS)
    {
        NN_LOG_ERROR("yolo load model file failed");
        return ret;
    }
    // get input tensor
    auto input_shapes = engine_->GetInputShapes();

    // check number of input and n_dims
    if (input_shapes.size() != 1)
    {
        NN_LOG_ERROR("yolo input tensor number is not 1, but %ld", input_shapes.size());
        return NN_RKNN_INPUT_ATTR_ERROR;
    }
    nn_tensor_attr_to_cvimg_input_data(input_shapes[0], input_tensor_);
    input_tensor_.data = malloc(input_tensor_.attr.size);

    auto output_shapes = engine_->GetOutputShapes();

    // 检查输出数量是否符合分割模型
    if (output_shapes.size() < 4) {
        NN_LOG_ERROR("yolo output tensor number is less than 4 for obb model, but %ld", output_shapes.size());
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }

    for (int i = 0; i < output_shapes.size(); i++)
    {
        tensor_data_s tensor;
        tensor.attr.n_elems = output_shapes[i].n_elems;
        tensor.attr.n_dims = output_shapes[i].n_dims;
        for (int j = 0; j < output_shapes[i].n_dims; j++)
        {
            tensor.attr.dims[j] = output_shapes[i].dims[j];
        }
        // output tensor needs to be int8 for obb heads
        if (i < 4 && output_shapes[i].type != NN_TENSOR_INT8)  // 4个输出为INT8
        {
            NN_LOG_ERROR("yolo output tensor %d type is not int8, but %d", i, output_shapes[i].type);
            return NN_RKNN_OUTPUT_ATTR_ERROR;
        }
        tensor.attr.type = output_shapes[i].type;
        tensor.attr.index = i;
        tensor.attr.size = output_shapes[i].n_elems * nn_tensor_type_to_size(tensor.attr.type);
        tensor.data = malloc(tensor.attr.size);
        output_tensors_.push_back(tensor);
        out_zps_.push_back(output_shapes[i].zp);
        out_scales_.push_back(output_shapes[i].scale);
    }
    return NN_SUCCESS;
}

// 图像预处理
nn_error_e Yolov5::Preprocess(const cv::Mat &img, const std::string process_type, cv::Mat &image_letterbox)
{

    // 预处理包含：letterbox、归一化、BGR2RGB、NCWH
    // 其中RKNN会做：归一化、NCWH转换（详见课程文档），所以这里只需要做letterbox、BGR2RGB

    // 比例
    float wh_ratio = (float)input_tensor_.attr.dims[2] / (float)input_tensor_.attr.dims[1];

    // lettorbox

    if (process_type == "opencv")
    {
        // BGR2RGB，resize，再放入input_tensor_中
        letterbox_info_ = letterbox(img, image_letterbox, wh_ratio);
        cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
    }

    return NN_SUCCESS;
}

// 推理
nn_error_e Yolov5::Inference()
{
    std::vector<tensor_data_s> inputs;
    // 将input_tensor_放入inputs中
    inputs.push_back(input_tensor_);
    // 运行模型
    engine_->Run(inputs, output_tensors_, false);
    return NN_SUCCESS;
}

// 运行模型
nn_error_e Yolov5::Run(const cv::Mat &img, std::vector<ObbDetection> &objects)
{
    // 保存原始图像尺寸
    original_img_width_ = img.cols;
    original_img_height_ = img.rows;
    
    // letterbox后的图像
    cv::Mat image_letterbox;
    // 预处理
    auto start_time_pre = std::chrono::high_resolution_clock::now();
    Preprocess(img, "opencv", image_letterbox);
    auto end_time_pre = std::chrono::high_resolution_clock::now();
    // 推理
    auto start_time_inf = std::chrono::high_resolution_clock::now();
    Inference();
    auto end_time_inf = std::chrono::high_resolution_clock::now();
    // 后处理
    auto start_time_pos = std::chrono::high_resolution_clock::now();
    Postprocess(image_letterbox, objects);
    auto end_time_pos = std::chrono::high_resolution_clock::now();
    NN_LOG_INFO("Preprocess time: %f ms", std::chrono::duration_cast<std::chrono::microseconds>(end_time_pre - start_time_pre).count() / 1000.0f);
    NN_LOG_INFO("Inference time: %f ms", std::chrono::duration_cast<std::chrono::microseconds>(end_time_inf - start_time_inf).count() / 1000.0f);
    NN_LOG_INFO("Postprocess time: %f ms", std::chrono::duration_cast<std::chrono::microseconds>(end_time_pos - start_time_pos).count() / 1000.0f);
    return NN_SUCCESS;
}

// 后处理
nn_error_e Yolov5::Postprocess(const cv::Mat &image_letterbox, std::vector<ObbDetection> &objects)
{
    int height = input_tensor_.attr.dims[1];
    int width = input_tensor_.attr.dims[2];

    objects.clear();
    std::vector<ObbDetection> obb_results;

    // 检查输出张量数量
    if (output_tensors_.size() < 4) {
        NN_LOG_ERROR("Not enough output tensors for obb model");
        return NN_RKNN_OUTPUT_ATTR_ERROR;
    }

    int8_t* outputs[4] = {
    (int8_t*)output_tensors_[0].data,
    (int8_t*)output_tensors_[1].data,
    (int8_t*)output_tensors_[2].data,
    (int8_t*)output_tensors_[3].data
    };

    // 调用支持分割的后处理函数
    postprocess_yolov5_obb(outputs, height, width, 0.25f, 0.2f, out_zps_.data(), out_scales_.data(), obb_results);

    float gain = std::min((float)image_letterbox.rows / (float)original_img_height_,
                          (float)image_letterbox.cols / (float)original_img_width_);

    float pad_x = letterbox_info_.hor ? (float)letterbox_info_.pad : 0.0f;
    float pad_y = letterbox_info_.hor ? 0.0f : (float)letterbox_info_.pad;

    // 统一缩放和偏移
    for (auto &obj : obb_results) {
        // 1. 减 pad（恢复到 resize 前的空间）
        obj.cx -= pad_x;
        obj.cy -= pad_y;

        // 2. 重新生成 poly（在减 pad 后的坐标上）
        rbox_to_poly(obj.cx, obj.cy, obj.w, obj.h, obj.theta, obj.poly);

        // 3. 乘 1/gain（从 letterbox 空间 → 原始图像空间）
        float scale_to_original = 1.0f / gain;
        obj.cx *= scale_to_original;
        obj.cy *= scale_to_original;
        obj.w  *= scale_to_original;
        obj.h  *= scale_to_original;

        for (int i = 0; i < 8; i++) {
            obj.poly[i] *= scale_to_original;
        }
    }

    objects = obb_results;

    return NN_SUCCESS;
}