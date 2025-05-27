#ifndef TSN_TRT_H
#define TSN_TRT_H

#include "videorecognitionTrt.h"
#include <vector>

class tsnTrt : public VideoRecognitionTRT
{
public:
    // 构造函数
    tsnTrt(const std::string &engine_name, const u_int32_t input_imt_shape);
    // 析构
    ~tsnTrt() override;

    // 初始化模型输出
    bool prepare_output(const std::string &output_name);

    // 准备模型输入
    bool prepare_input(const std::string &input_name, const int &clip_num, const int &clip_length, const float *input_data);

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
    void *intput_dev_buffer_ = nullptr; // 输入数据在GPU上的缓冲区
    void *output_dev_buffer_ = nullptr; // 输出数据在GPU上的缓冲区
};

#endif