#ifndef X3D_TRT_H
#define X3D_TRT_H

#include "videorecognitionTrt.h"
#include <vector>

class X3dTrt : public VideoRecognitionTRT
{
public:
    // 构造函数
    X3dTrt(const std::string &engine_name, const u_int32_t input_imt_shape);
    // 析构
    ~X3dTrt() override;

    // 初始化模型输出
    bool prepare_output(const std::string &output_name);

    // 准备模型输入
    // X3D: 输入形状 1 x 3 x 32 x 64 x 64 (NCTHW)
    bool prepare_input(const std::string &input_name, const int &num_frames, 
                       const int &height, const int &width, const float *input_data);

    // 获取输出
    void get_output(float *output_data);

    // 解析输出结果
    RECOGNITION parse_output(const float *output_data);

    int GetOutputSize() const
    {
        return this->output_size_;
    }

private:
    int output_size_;                   // 模型输出有多少个float
    float *output_ = nullptr;           // 模型输出数据
    void *input_dev_buffer_ = nullptr;  // 输入数据在GPU上的缓冲区
    void *output_dev_buffer_ = nullptr; // 输出数据在GPU上的缓冲区
};

#endif
