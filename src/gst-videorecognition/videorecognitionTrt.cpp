#include "videorecognitionTrt.h"
#include <fstream>
#include <cassert>

VideoRecognitionTRT::VideoRecognitionTRT(const std::string &engine_name,
                                         const u_int32_t input_imt_shape)
{
    // deserialize engine
    this->deserialize_engine(engine_name);
    this->input_imt_shape_ = input_imt_shape;
    CHECK(cudaStreamCreate(&this->stream_));
}

VideoRecognitionTRT::~VideoRecognitionTRT()
{
    CHECK(cudaStreamDestroy(stream_));
    delete context_;
    delete engine_;
    delete runtime_;
}

bool VideoRecognitionTRT::do_inference()
{
    // inference
    bool res = context_->enqueueV3(stream_);
    return res;
}

void VideoRecognitionTRT::deserialize_engine(const std::string &engine_name)
{
    // create a model using the API directly and serialize it to a stream
    std::ifstream file(engine_name, std::ios::binary);
    if (file.good())
    {
        file.seekg(0, file.end);
        size_ = file.tellg();
        file.seekg(0, file.beg);
        this->trt_model_stream_ = new char[this->size_];
        assert(this->trt_model_stream_);
        file.read(trt_model_stream_, this->size_);
        file.close();
    }

    this->runtime_ = createInferRuntime(this->gLogger_);
    assert(this->runtime_ != nullptr);

    this->engine_ = this->runtime_->deserializeCudaEngine(trt_model_stream_,
                                                          this->size_);
    assert(this->engine_ != nullptr);

    this->context_ = this->engine_->createExecutionContext();
    assert(context_ != nullptr);
}