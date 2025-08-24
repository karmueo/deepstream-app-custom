#include "tsnTrt.h"
#include <algorithm>
#include "process.h"

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
    if (output_dev_buffer_)
    {
        cudaFree(output_dev_buffer_);
        output_dev_buffer_ = nullptr;
    }
}

bool tsnTrt::prepare_output(const std::string &output_name)
{
    assert(engine_->getTensorDataType(output_name.c_str()) == nvinfer1::DataType::kFLOAT);

    // 如果已经准备过（设备缓冲区存在）直接返回成功
    if (output_dev_buffer_ != nullptr && this->output_size_ > 0)
    {
        // 再次绑定一次以防上下文重置
        return context_->setTensorAddress(output_name.c_str(), output_dev_buffer_);
    }

    auto out_dims = this->engine_->getTensorShape("/Softmax_output_0");
    int output_num = 1;
    for (int j = 0; j < out_dims.nbDims; j++)
        output_num *= out_dims.d[j];

    // 只分配一次主机缓存
    if (!this->output_)
        this->output_ = new float[output_num];
    this->output_size_ = output_num * sizeof(float);
    CHECK(cudaMalloc(&output_dev_buffer_, this->output_size_));
    return context_->setTensorAddress(output_name.c_str(), output_dev_buffer_);
}

bool tsnTrt::prepare_input(const std::string &input_name,
                           const int &clip_num,
                           const int &clip_length,
                           const float *input_data)
{
    assert(engine_->getNbIOTensors() == 2);

    // 从张量名称中确定缓冲区所需的数据类型
    // 确保输入名称与模型中的输入名称匹配
    assert(engine_->getTensorDataType(input_name.c_str()) == nvinfer1::DataType::kFLOAT);

    // 在设备上创建GPU缓冲区
    // INPUT
    auto input_size = clip_num * clip_length * 3 * input_imt_shape_ * input_imt_shape_ * sizeof(float);
    auto int_dims = this->engine_->getTensorShape("input");
    int input_num = 1;
    for (int j = 0; j < int_dims.nbDims; j++)
    {
        input_num *= int_dims.d[j];
    }
    assert(input_num * sizeof(float) == input_size);

    // 若上一次输入缓冲尚未释放，先释放（理论上 get_output 会释放）
    if (intput_dev_buffer_)
    {
        CHECK(cudaFree(intput_dev_buffer_));
        intput_dev_buffer_ = nullptr;
    }
    CHECK(cudaMalloc(&intput_dev_buffer_, input_size));
    // DMA（直接内存访问）输入批处理数据到设备，在异步上推断批处理，然后DMA输出回到主机
    CHECK(cudaMemcpyAsync(intput_dev_buffer_, input_data, input_size, cudaMemcpyHostToDevice, stream_));
    // 给定输入或输出张量的简短设置内存地址。
    bool res = context_->setTensorAddress(input_name.c_str(), intput_dev_buffer_);

    return res;
}

void tsnTrt::get_output(float *output_data)
{
    if (!output_dev_buffer_)
    {
        std::cerr << "[tsnTrt] get_output called with null output_dev_buffer_" << std::endl;
        return;
    }
    CHECK(cudaMemcpyAsync(output_data, output_dev_buffer_, this->output_size_, cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_); // 等待复制完成

    // 仅释放一次性输入缓冲，输出缓冲保留供下次推理复用
    if (intput_dev_buffer_)
    {
        CHECK(cudaFree(intput_dev_buffer_));
        intput_dev_buffer_ = nullptr;
    }
}

RECOGNITION tsnTrt::parse_output(const float *output_data)
{
    const std::vector<int> shape = {1, 4, 2};
    int class_num = shape[2];
    // 解析输出结果
    std::vector<RECOGNITION> result;
    result.resize(class_num);

    auto data = reshape_to_3d(output_data, shape);
    for (auto i = 0; i < shape[1]; i++)
    {
        for (auto j = 0; j < class_num; j++)
        {
            RECOGNITION rec = result[j];
            rec.class_id = j;
            rec.score += data[0][i][j];
            result[j] = rec;
        }
    }

    for (auto i = 0; i < class_num; i++)
    {
        result[i].score = result[i].score / shape[1];
    }

    /* for (int i = 0; i < this->output_size_ / sizeof(float); i++)
    {
        RECOGNITION rec;
        rec.class_id = i;
        rec.score = output_data[i];
        result.push_back(rec);
    } */

    // Sort the results based on score
    std::sort(result.begin(), result.end(), [](const RECOGNITION &a, const RECOGNITION &b)
              { return a.score > b.score; });

    return result[0]; // Return the top result
}
