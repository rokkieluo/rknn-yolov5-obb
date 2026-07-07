# RKNN YOLOv5-OBB 使用教程

本工程是在 RK3588 平台上运行的 YOLOv5-OBB 旋转框目标检测示例。程序通过 RKNN Runtime 加载 `.rknn` 模型，使用 OpenCV 完成图片读取和结果绘制，最终输出带旋转检测框的 `result.jpg`。

当前默认启用的是单张图片推理入口 `yolov5_img`。工程中也保留了线程池视频推理相关代码，但 CMake 中默认未开启，而且线程池代码仍引用旧的 `Detection/DrawDetections` 接口，直接启用前需要改成当前 OBB 使用的 `ObbDetection/DrawObbDetections`。

## 目录结构

```text
.
├── 3rdparty/                 # RGA、allocator 等第三方依赖
├── librknn_api/              # RKNN Runtime 头文件和 librknnrt.so
├── media/                    # 建议放测试图片或视频
├── src/
│   ├── draw/                 # 检测结果绘制
│   ├── engine/               # RKNN 推理封装
│   ├── process/              # 前处理和后处理
│   ├── task/                 # YOLOv5 推理任务封装
│   ├── types/                # 数据结构定义
│   ├── yolov5_img.cpp        # 单图推理 main
│   └── yolov5_thread_pool.cpp# 视频/线程池示例入口，默认未编译
├── weights/                  # 建议放 RKNN 模型
└── CMakeLists.txt
```

## 环境依赖

建议在 RK3588 板端 Linux 环境编译和运行。

需要准备：

- CMake 3.11 或更高版本
- 支持 C++17 的编译器
- OpenCV 开发库
- RKNN Runtime，工程已包含 `librknn_api/aarch64/librknnrt.so`
- RGA 库，工程已包含 `3rdparty/rga/RK3588/lib/Linux/aarch64/librga.so`

如果使用交叉编译，需要额外准备目标平台的 OpenCV、RKNN Runtime、RGA 以及对应 toolchain，并按实际路径修改 `CMakeLists.txt`。

## 编译

在工程根目录执行：

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

编译成功后会生成：

```text
build/yolov5_img
```

如果运行时提示找不到 `librknnrt.so` 或 `librga.so`，可以在 `build` 目录下设置动态库路径：

```bash
export LD_LIBRARY_PATH=$PWD/../librknn_api/aarch64:$PWD/../3rdparty/rga/RK3588/lib/Linux/aarch64:$LD_LIBRARY_PATH
```

也可以把相关 `.so` 安装到系统库路径中。

## 单张图片推理

命令格式：

```bash
./yolov5_img <model_file> <img_file> [loop_count]
```

参数说明：

- `model_file`：RKNN 模型路径，例如 `../weights/yolov5_obb.rknn`
- `img_file`：输入图片路径，例如 `../media/test.jpg`
- `loop_count`：可选，循环推理次数，默认 `1`，用于统计平均耗时和 FPS

示例：

```bash
cd build
./yolov5_img ../weights/yolov5_obb.rknn ../media/test.jpg
```

循环推理 100 次并统计平均耗时：

```bash
./yolov5_img ../weights/yolov5_obb.rknn ../media/test.jpg 100
```

程序会在当前运行目录生成：

```text
result.jpg
```

终端会打印模型输入输出信息、前处理耗时、推理耗时、后处理耗时、目标数量、类别、置信度和角度等信息。

## 当前模型假设

当前后处理代码按下面的 YOLOv5-OBB 输出格式编写：

- 输入图像按 letterbox 方式缩放，默认目标高度为 `640`
- 模型只有 1 个输入
- 模型至少有 4 个输出头
- 前 4 个输出头类型为 `int8`
- 每个输出头使用 1 个 anchor
- 当前类别数为 `2`
- 当前角度分类 bin 数为 `180`
- 每个网格点通道数为 `5 + NUM_CLASSES + THETA_BINS`
- 当前通道数为 `5 + 2 + 180 = 187`
- 当前 stride 为 `{2, 4, 8, 16}`
- 当前 anchor 为：

```cpp
static const float ANCHORS[NUM_HEADS][2] = {
    {2.49609375f, 2.498046875f},
    {3.404296875f, 1.4990234375f},
    {2.974609375f, 1.0205078125f},
    {1.0908203125f, 1.09375f}
};
```

如果你的模型结构和这些假设不一致，需要同步修改代码，否则会出现类别错误、框位置错误、角度错误、输出越界或没有检测结果。

## 使用自己的模型时需要修改哪里

只替换模型文件路径不需要改代码，运行命令传入新的 `.rknn` 即可：

```bash
./yolov5_img ../weights/your_model.rknn ../media/test.jpg
```

但前提是你的模型输出格式必须和当前工程完全一致。如果类别数、输出头数量、stride、anchor、输入尺寸或角度编码不同，需要修改以下位置。

### 1. 修改类别数和类别名

文件：`src/process/postprocess.cpp`

```cpp
#define NUM_CLASSES 2

static const char* CLASS_NAMES[NUM_CLASSES] = {
    "class0", "class1"
};
```

将 `NUM_CLASSES` 改成自己的类别数量，并把 `CLASS_NAMES` 改成自己的类别名。

如果类别数超过 10，建议同时修改 `src/draw/cv_draw.cpp` 中的标签判断：

```cpp
if (!det.className.empty() && det.class_id < 10)
```

可以改为：

```cpp
if (!det.className.empty())
```

### 2. 修改输出头数量

文件：`src/process/postprocess.cpp`

```cpp
#define NUM_HEADS 4
```

如果你的模型不是 4 个输出头，需要同步修改：

- `src/process/postprocess.cpp` 中的 `NUM_HEADS`
- `STRIDES`
- `ANCHORS`
- `postprocess_yolov5_obb` 的输出数组使用方式
- `src/process/postprocess.h` 中的输出数组声明
- `src/task/yolov5.cpp` 中对输出数量的检查和 `outputs` 数组

当前 `src/task/yolov5.cpp` 写死使用前 4 个输出：

```cpp
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
```

输出头数量变化时，这里必须一起改。

### 3. 修改 stride 和 anchor

文件：`src/process/postprocess.cpp`

```cpp
static const int STRIDES[NUM_HEADS] = {2, 4, 8, 16};

static const float ANCHORS[NUM_HEADS][2] = {
    {2.49609375f, 2.498046875f},
    {3.404296875f, 1.4990234375f},
    {2.974609375f, 1.0205078125f},
    {1.0908203125f, 1.09375f}
};
```

这里必须和训练、导出、转换 RKNN 时的模型输出头一致。  
如果你的 YOLOv5-OBB 是常见的 3 个检测头，通常需要改为 3 组 stride 和 anchor，并同步修改输出头数量。

### 4. 修改角度编码方式

文件：`src/process/postprocess.cpp`

当前角度使用 180 个 bin：

```cpp
#define THETA_BINS 180
```

当前解码逻辑：

```cpp
float theta = ((float)best_bin - 90.0f) * M_PI / 180.0f;
theta = -theta;
```

如果你的模型角度不是 180 分类，例如直接回归角度、90 分类、周期角度、DOTA le90/le135 表示方式等，需要修改：

- `THETA_BINS`
- `theta_offset`
- `best_bin` 或角度回归读取方式
- `theta` 的弧度换算和正负方向
- 必要时修改 `rbox_to_poly`

### 5. 修改输入尺寸或 letterbox

文件：`src/process/preprocess.cpp`

当前 letterbox 中写死了目标高度：

```cpp
float target_h = 640.0f;
float target_w = target_h * wh_ratio;
```

如果你的模型不是 640 输入，需要改这里。更推荐的做法是把模型真实输入高宽从 `Yolov5::Preprocess` 传入 `letterbox`，避免写死。

`src/task/yolov5.cpp` 中会从 RKNN 模型读取输入尺寸，并在 `cvimg2tensor` 时使用：

```cpp
cvimg2tensor(image_letterbox, input_tensor_.attr.dims[2], input_tensor_.attr.dims[1], input_tensor_);
```

但 letterbox 阶段仍然用了固定 `640`，所以非 640 模型一定要检查。

### 6. 修改置信度和 NMS 阈值

文件：`src/task/yolov5.cpp`

当前阈值在 `Postprocess` 中写死：

```cpp
postprocess_yolov5_obb(outputs, height, width, 0.25f, 0.2f,
                       out_zps_.data(), out_scales_.data(), obb_results);
```

其中：

- `0.25f` 是置信度阈值
- `0.2f` 是旋转框 NMS 阈值

如果检测结果太少，可以适当降低置信度阈值；如果重复框太多，可以调整 NMS 阈值。

### 7. 检查输出类型

文件：`src/task/yolov5.cpp`

当前代码要求前 4 个输出是 `int8`：

```cpp
if (i < 4 && output_shapes[i].type != NN_TENSOR_INT8)
```

如果你的 RKNN 模型输出是 `float32` 或其他类型，需要修改：

- 输出 tensor 类型检查
- `postprocess_yolov5_obb` 的输入类型
- `decode_head` 中的反量化逻辑
- `engine_->Run(inputs, output_tensors_, false)` 的 `want_float` 参数

当前后处理默认使用 RKNN 的量化输出，并通过 `zp` 和 `scale` 反量化。

### 8. 如果不是 OBB 模型

如果你的模型是普通水平框 YOLO，而不是旋转框 OBB，不能只改类别名和 anchor。需要替换：

- `src/process/postprocess.cpp`
- `src/process/postprocess.h`
- `src/types/yolo_datatype.h` 中的检测结果结构
- `src/draw/cv_draw.cpp` 中的绘制逻辑
- `src/task/yolov5.cpp` 中调用后处理和绘制结果的相关类型

## 启用线程池视频示例

当前 `CMakeLists.txt` 中线程池目标被注释：

```cmake
# add_executable(yolov5_thread_pool
#     src/yolov5_thread_pool.cpp
#     src/task/yolov5_thread_pool.cpp
# )
```

如果需要启用视频推理，需要先做两件事：

1. 放开 CMake 中 `yolov5_thread_pool` 的 `add_executable` 和 `target_link_libraries`
2. 将 `src/task/yolov5_thread_pool.h/.cpp` 中旧的 `Detection`、`DrawDetections` 改成当前 OBB 使用的 `ObbDetection`、`DrawObbDetections`

然后重新编译。

视频入口参数来自 `src/yolov5_thread_pool.cpp`：

```bash
./yolov5_thread_pool <model_file> <video_file> [num_threads] [record]
```

其中 `record=1` 时会尝试保存 `result_pool.mp4`。

## 常见问题

### 1. `cmake ..` 报 CMake 语法错误

检查 `CMakeLists.txt` 是否因为编码或换行问题导致命令被中文注释吞掉。需要确保这些命令是独立的 CMake 命令，而不是注释的一部分：

```cmake
set(CMAKE_CXX_STANDARD 17)
set(LIB_ARCH "aarch64")
set(RKNN_API_PATH ${CMAKE_CURRENT_SOURCE_DIR}/librknn_api)
target_link_libraries(nn_process ...)
add_library(rknn_engine SHARED src/engine/rknn_engine.cpp)
target_link_libraries(rknn_engine ...)
target_link_libraries(yolov5_lib ...)
target_link_libraries(draw_lib ...)
```

### 2. 运行时报 `load model file fail`

检查模型路径是否正确，建议使用绝对路径或确认当前运行目录。例如在 `build` 目录运行时，模型通常写成：

```bash
../weights/your_model.rknn
```

### 3. 运行时报 `rknn_init fail`

常见原因：

- `.rknn` 不是为 RK3588 平台转换的
- RKNN Runtime 和板端驱动版本不匹配
- 模型文件损坏或路径错误

程序启动时会打印 RKNN API version 和 Driver version，可以据此检查版本。

### 4. 没有检测结果

优先检查：

- 类别数 `NUM_CLASSES` 是否正确
- 输出头数量、stride、anchor 是否匹配
- 输入尺寸和 letterbox 是否匹配
- 角度编码是否匹配
- 置信度阈值是否过高
- 图片通道是否为 3 通道

### 5. 框位置正确但类别名错误

修改 `src/process/postprocess.cpp` 中的 `CLASS_NAMES`，并确保顺序和训练数据集的类别顺序一致。

### 6. 框位置或角度明显错误

优先检查：

- `STRIDES`
- `ANCHORS`
- `THETA_BINS`
- `theta` 解码公式
- `rbox_to_poly`
- `preprocess.cpp` 中的输入尺寸和 letterbox 逻辑

## 开发入口速查

常用文件：

- `src/yolov5_img.cpp`：命令行参数、循环推理、结果保存
- `src/task/yolov5.cpp`：模型加载、前处理、推理、后处理串联
- `src/process/preprocess.cpp`：letterbox、BGR 转 RGB、图像写入输入 tensor
- `src/process/postprocess.cpp`：YOLOv5-OBB 解码、类别、anchor、NMS
- `src/draw/cv_draw.cpp`：旋转框和标签绘制
- `src/engine/rknn_engine.cpp`：RKNN Runtime 加载、输入输出查询、推理执行

