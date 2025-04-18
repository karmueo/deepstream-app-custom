#ifndef VIDEORECOGNITION_TRT_H
#define VIDEORECOGNITION_TRT_H

#include <iostream>
#include "NvInfer.h" // Include TensorRT header for IRuntime
#include "cuda_runtime_api.h"
// #include "cuda_fp16.h"
#include "logging.h"

using namespace nvinfer1;

// 宏为 CUDA 调用提供了简洁一致的错误检查机制，在出现问题时能够及时报告并中断运行
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)

struct RECOGNITION
{
    std::string class_name; //
    int class_id;           //
    float score;            // 置信度
};

class VideoRecognitionTRT
{
public:
    VideoRecognitionTRT(const std::string &engine_name, const u_int32_t input_imt_shape);
    virtual ~VideoRecognitionTRT();

    // 执行推理
    bool do_inference();

private:
    void deserialize_engine(const std::string &engine_name);

protected:
    size_t size_{0};
    char *trt_model_stream_ = nullptr;     // 保存 TensorRT 序列化后原始二进制数据的成员指针
    IRuntime *runtime_ = nullptr;          // TensorRT 运行时对象
    ICudaEngine *engine_ = nullptr;        // TensorRT 引擎对象
    IExecutionContext *context_ = nullptr; // TensorRT 执行上下文对象
    Logger gLogger_;                       // 日志记录器对象
    u_int32_t input_imt_shape_;            // 输入图像的形状，默认长款相等
    cudaStream_t stream_;                  // CUDA 流对象
    uint32_t input_size_;                  // 输入数据的大小
};

#endif