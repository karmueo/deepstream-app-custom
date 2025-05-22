#include "tsnTrt.h"
#include <algorithm>

tsnTrt::tsnTrt(const std::string &engine_name, const u_int32_t input_imt_shape) : VideoRecognitionTRT(engine_name, input_imt_shape)
{
}

tsnTrt::~tsnTrt()
{
    if (this->output_ != nullptr)
    {
        delete[] this->output_;
        this->output_ = nullptr;
    }
}

bool tsnTrt::prepare_output(const std::string &output_name)
{
    assert(engine_->getTensorDataType(output_name.c_str()) == nvinfer1::DataType::kFLOAT);

    // 应该等于类别数量
    auto out_dims = this->engine_->getTensorShape("output");

    int output_num = 1;
    for (int j = 0; j < out_dims.nbDims; j++)
    {
        output_num *= out_dims.d[j];
    }
    this->output_ = new float[output_num];
    this->output_size_ = output_num * sizeof(float);
    CHECK(cudaMalloc(&output_dev_buffer_, this->output_size_));

    bool res = context_->setTensorAddress(output_name.c_str(), output_dev_buffer_);

    return res;
}

bool tsnTrt::prepare_input(const std::string &input_name,
                           const int &clip_length,
                           const float *input_data)
{
    assert(engine_->getNbIOTensors() == 2);

    // 从张量名称中确定缓冲区所需的数据类型
    // 确保输入名称与模型中的输入名称匹配
    assert(engine_->getTensorDataType(input_name.c_str()) == nvinfer1::DataType::kFLOAT);

    // 在设备上创建GPU缓冲区
    // INPUT
    auto input_size = clip_length * 3 * input_imt_shape_ * input_imt_shape_ * sizeof(float);
    CHECK(cudaMalloc(&intput_dev_buffer_, input_size));
    // DMA（直接内存访问）输入批处理数据到设备，在异步上推断批处理，然后DMA输出回到主机
    CHECK(cudaMemcpyAsync(intput_dev_buffer_, input_data, input_size, cudaMemcpyHostToDevice, stream_));
    // 给定输入或输出张量的简短设置内存地址。
    bool res = context_->setTensorAddress(input_name.c_str(), intput_dev_buffer_);

    return res;
}

void tsnTrt::get_output(float *output_data)
{
    CHECK(cudaMemcpyAsync(output_data, output_dev_buffer_, this->output_size_, cudaMemcpyDeviceToHost, stream_));

    // 等待流完成
    cudaStreamSynchronize(stream_);

    // release buffers
    CHECK(cudaFree(intput_dev_buffer_));
    CHECK(cudaFree(output_dev_buffer_));
    intput_dev_buffer_ = nullptr;
    output_dev_buffer_ = nullptr;
}

RECOGNITION tsnTrt::parse_output(const float *output_data)
{
    // 解析输出结果
    std::vector<RECOGNITION> result;
    for (int i = 0; i < this->output_size_ / sizeof(float); i++)
    {
        RECOGNITION rec;
        rec.class_id = i;
        rec.score = output_data[i];
        result.push_back(rec);
    }

    // Sort the results based on score
    std::sort(result.begin(), result.end(), [](const RECOGNITION &a, const RECOGNITION &b)
              { return a.score > b.score; });

    return result[0]; // Return the top result
}
