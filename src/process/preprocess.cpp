// 预处理

#include "preprocess.h"

#include "utils/logging.h"
#include "im2d.h"
#include "dma_alloc.h"
#include "im2d.hpp"
#include "rga.h"
#include "RgaUtils.h"


// opencv 版本的 letterbox
// opencv 版本的 letterbox
LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    float img_width = img.cols;
    float img_height = img.rows;
    
    // 参考Python版本的letterbox实现
    float target_h = 640.0f;  // 假设模型输入高度为640
    float target_w = target_h * wh_ratio;  // 根据比例计算目标宽度
    
    // 计算缩放比例
    float r = std::min(target_w / img_width, target_h / img_height);
    
    // 计算缩放后的尺寸
    int new_unpad_w = (int)round(img_width * r);
    int new_unpad_h = (int)round(img_height * r);
    
    LetterBoxInfo info;
    int pad_left = 0, pad_right = 0, pad_top = 0, pad_bottom = 0;
    
    if (img_width / (float)img_height > wh_ratio) {
        // 图像更宽（w/h > wh_ratio），需要垂直填充（pad高度）
        info.hor = false;
        float dh = target_h - new_unpad_h;
        pad_top = (int)(dh / 2 - 0.1);
        pad_bottom = (int)(dh / 2 + 0.1);
    } else {
        // 图像更高（w/h <= wh_ratio），需要水平填充（pad宽度）
        info.hor = true;
        float dw = target_w - new_unpad_w;
        pad_left = (int)(dw / 2 - 0.1);
        pad_right = (int)(dw / 2 + 0.1);
    }
    info.pad = info.hor ? pad_left : pad_top;
    
    // 先resize
    cv::Mat resized_img;
    cv::resize(img, resized_img, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);
    
    // 再填充
    cv::copyMakeBorder(resized_img, img_letterbox, 
                       pad_top, pad_bottom, pad_left, pad_right, 
                       cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
    
    return info;
}

// opencv resize
void cvimg2tensor(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    // BGR to RGB
    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);
    // resize img
    cv::Mat img_resized;
    // resize img
    cv::resize(img_rgb, img_resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
    // BGR to RGB
    memcpy(tensor.data, img_resized.data, tensor.attr.size);
}

// rga 版本的 resize
void cvimg2tensor_rga(const cv::Mat &img, uint32_t width, uint32_t height, tensor_data_s &tensor)
{
    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }

    cv::Mat img_rgb;
    cv::cvtColor(img, img_rgb, cv::COLOR_BGR2RGB);

    im_rect src_rect;
    im_rect dst_rect;
    memset(&src_rect, 0, sizeof(src_rect));
    memset(&dst_rect, 0, sizeof(dst_rect));
    rga_buffer_t src = wrapbuffer_virtualaddr((void *)img_rgb.data, img.cols, img.rows, RK_FORMAT_RGB_888);
    rga_buffer_t dst = wrapbuffer_virtualaddr((void *)tensor.data, width, height, RK_FORMAT_RGB_888);
    int ret = imcheck(src, dst, src_rect, dst_rect);
    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }
    imresize(src, dst);
}

// rga 版本的 letterbox
LetterBoxInfo letterbox_rga(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio)
{
    // img has to be 3 channels
    if (img.channels() != 3)
    {
        NN_LOG_ERROR("img has to be 3 channels");
        exit(-1);
    }
    float img_width = img.cols;
    float img_height = img.rows;

    int letterbox_width = 0;
    int letterbox_height = 0;

    LetterBoxInfo info;
    int padding_hor = 0;
    int padding_ver = 0;

    if (img_width / img_height > wh_ratio)
    {
        info.hor = false;
        letterbox_width = img_width;
        letterbox_height = img_width / wh_ratio;
        info.pad = (letterbox_height - img_height) / 2.f;
        padding_hor = 0;
        padding_ver = info.pad;
    }
    else
    {
        info.hor = true;
        letterbox_width = img_height * wh_ratio;
        letterbox_height = img_height;
        info.pad = (letterbox_width - img_width) / 2.f;
        padding_hor = info.pad;
        padding_ver = 0;
    }
    // rga add border
    img_letterbox = cv::Mat::zeros(letterbox_height, letterbox_width, CV_8UC3);

    int ret = 0;
    int src_dma_fd, dst_dma_fd;
    char *src_buf, *dst_buf;
    int src_buf_size, dst_buf_size;
    rga_buffer_t src_img, dst_img;
    rga_buffer_handle_t src_handle, dst_handle;

    memset(&src_img, 0, sizeof(src_img));
    memset(&dst_img, 0, sizeof(dst_img));

    src_buf_size = img_width * img_height * get_bpp_from_format(RK_FORMAT_RGB_888);
    dst_buf_size = letterbox_width * letterbox_height * get_bpp_from_format(RK_FORMAT_RGB_888);

    ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHED_PATH, src_buf_size, &src_dma_fd, (void **)&src_buf);
    if (ret < 0) 
    {
        NN_LOG_ERROR("alloc src dma_heap buffer failed!\n");
        exit(-1);
    }

    ret = dma_buf_alloc(DMA_HEAP_DMA32_UNCACHED_PATH, dst_buf_size, &dst_dma_fd, (void **)&dst_buf);
    if (ret < 0) 
    {
        NN_LOG_ERROR("alloc dst dma_heap buffer failed!\n");
        dma_buf_free(src_buf_size, &src_dma_fd, src_buf);
        exit(-1);
    }

    memcpy(src_buf, img.data, src_buf_size);
    memset(dst_buf, 0, dst_buf_size);

    src_handle = importbuffer_fd(src_dma_fd, src_buf_size);
    dst_handle = importbuffer_fd(dst_dma_fd, dst_buf_size);

    if (src_handle == 0 || dst_handle == 0) {
        NN_LOG_ERROR("importbuffer failed!\n");
        releasebuffer_handle(src_handle);
        releasebuffer_handle(dst_handle);
    }

    src_img = wrapbuffer_handle(src_handle, img_width, img_height, RK_FORMAT_RGB_888);
    dst_img = wrapbuffer_handle(dst_handle, letterbox_width, letterbox_height, RK_FORMAT_RGB_888);

    ret = imcheck(src_img, dst_img, {}, {});

    if (IM_STATUS_NOERROR != ret)
    {
        NN_LOG_ERROR("%d, check error! %s", __LINE__, imStrError((IM_STATUS)ret));
        exit(-1);
    }

    ret = immakeBorder(src_img, dst_img, padding_ver, padding_ver, padding_hor, padding_hor, IM_BORDER_CONSTANT);
    if (ret == IM_STATUS_SUCCESS) 
    {
        NN_LOG_INFO("running success!\n");
        memcpy(img_letterbox.data, dst_buf, dst_buf_size);
    } 
    else 
    {
        NN_LOG_ERROR("running failed, %s\n", imStrError((IM_STATUS)ret));
        dma_buf_free(src_buf_size, &src_dma_fd, src_buf);
        dma_buf_free(dst_buf_size, &dst_dma_fd, dst_buf);
    }

    // 清理资源
    releasebuffer_handle(src_handle);
    releasebuffer_handle(dst_handle);
    dma_buf_free(src_buf_size, &src_dma_fd, src_buf);
    dma_buf_free(dst_buf_size, &dst_dma_fd, dst_buf);

    return info;
}