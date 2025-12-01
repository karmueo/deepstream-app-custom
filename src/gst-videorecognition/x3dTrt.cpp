#include "x3dTrt.h"
#include <cassert>
#include <cmath>
#include <iostream>

X3dTrt::X3dTrt(const std::string &engine_name, const u_int32_t input_imt_shape)
    : VideoRecognitionTRT(engine_name, input_imt_shape)
{
    this->output_size_ = 0;
    this->output_ = nullptr;
    this->input_dev_buffer_ = nullptr;
    this->output_dev_buffer_ = nullptr;
}

X3dTrt::~X3dTrt()
{
    if (this->output_ != nullptr)
    {
        delete[] this->output_;
        this->output_ = nullptr;
    }
    if (this->input_dev_buffer_ != nullptr)
    {
        CHECK(cudaFree(this->input_dev_buffer_));
        this->input_dev_buffer_ = nullptr;
    }
    if (this->output_dev_buffer_ != nullptr)
    {
        CHECK(cudaFree(this->output_dev_buffer_));
        this->output_dev_buffer_ = nullptr;
    }
}

bool X3dTrt::prepare_output(const std::string &output_name)
{
    assert(engine_->getTensorDataType(output_name.c_str()) == nvinfer1::DataType::kFLOAT);

    // 如果已经准备过（设备缓冲区存在）直接返回成功
    if (output_dev_buffer_ != nullptr && this->output_size_ > 0)
    {
        // 再次绑定一次以防上下文重置
        return context_->setTensorAddress(output_name.c_str(), output_dev_buffer_);
    }

    // 获取输出维度
    auto out_dims = this->engine_->getTensorShape(output_name.c_str());
    
    // X3D 输出应该是 [1, num_classes]，例如 [1, 2]
    int output_num = 1;
    for (int j = 0; j < out_dims.nbDims; j++)
    {
        output_num *= out_dims.d[j];
    }

    std::cout << "Output size: " << output_num << std::endl;

    // 分配输出内存
    if (!this->output_)
        this->output_ = new float[output_num];
    this->output_size_ = output_num * sizeof(float);
    CHECK(cudaMalloc(&this->output_dev_buffer_, this->output_size_));

    // 设置输出绑定
    return this->context_->setTensorAddress(output_name.c_str(), this->output_dev_buffer_);
}

bool X3dTrt::prepare_input(const std::string &input_name, const int &num_frames,
                           const int &height, const int &width, const float *input_data)
{
    // 确保输入名称与模型中的输入名称匹配
    assert(engine_->getTensorDataType(input_name.c_str()) == nvinfer1::DataType::kFLOAT);

    // X3D 输入: 1 x 3 x num_frames x height x width (NCTHW)
    size_t input_size = 1 * 3 * num_frames * height * width * sizeof(float);
    
    auto int_dims = this->engine_->getTensorShape(input_name.c_str());
    int input_num = 1;
    for (int j = 0; j < int_dims.nbDims; j++)
    {
        input_num *= int_dims.d[j];
    }
    assert(input_num * sizeof(float) == input_size);

    // 若上一次输入缓冲尚未释放，先释放
    if (input_dev_buffer_)
    {
        CHECK(cudaFree(input_dev_buffer_));
        input_dev_buffer_ = nullptr;
    }

    // 分配输入缓冲区
    CHECK(cudaMalloc(&this->input_dev_buffer_, input_size));

    // 将输入数据从主机复制到设备
    CHECK(cudaMemcpyAsync(this->input_dev_buffer_, input_data, input_size, 
                          cudaMemcpyHostToDevice, stream_));

    // 设置输入绑定
    bool res = this->context_->setTensorAddress(input_name.c_str(), this->input_dev_buffer_);

    return res;
}

void X3dTrt::get_output(float *output_data)
{
    if (!output_dev_buffer_)
    {
        std::cerr << "[X3dTrt] get_output called with null output_dev_buffer_" << std::endl;
        return;
    }
    // 将输出从设备复制到主机
    CHECK(cudaMemcpyAsync(output_data, this->output_dev_buffer_,
                          this->output_size_, cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_); // 等待复制完成

    // 仅释放一次性输入缓冲，输出缓冲保留供下次推理复用
    if (input_dev_buffer_)
    {
        CHECK(cudaFree(input_dev_buffer_));
        input_dev_buffer_ = nullptr;
    }
}

RECOGNITION X3dTrt::parse_output(const float *output_data)
{
    RECOGNITION result;
    result.class_id = -1;
    result.score = -1.0f;

    // 计算实际输出数量（字节转为float数量）
    int num_classes = this->output_size_ / sizeof(float);

    // 找到最大值的索引
    for (int i = 0; i < num_classes; ++i)
    {
        if (output_data[i] > result.score)
        {
            result.score = output_data[i];
            result.class_id = i;
        }
    }

    // 设置类别名称（根据实际需求修改）
    if (result.class_id == 0)
    {
        result.class_name = "class_0";
    }
    else if (result.class_id == 1)
    {
        result.class_name = "class_1";
    }
    else
    {
        result.class_name = "unknown";
    }

    return result;
}
