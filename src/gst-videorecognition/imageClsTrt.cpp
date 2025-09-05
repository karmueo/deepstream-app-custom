#include "imageClsTrt.h"
#include <cassert>
#include <cstring>

ImageClsTrt::ImageClsTrt(const std::string &engine_path) : VideoRecognitionTRT(engine_path, 32) {
    discoverIO();
}

ImageClsTrt::~ImageClsTrt() {
    if (device_input_) cudaFree(device_input_);
    if (device_output_) cudaFree(device_output_);
}

void ImageClsTrt::discoverIO() {
    int nb = engine_->getNbIOTensors();
    for (int i = 0; i < nb; ++i) {
        const char *name = engine_->getIOTensorName(i);
        if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
        } else {
            output_name_ = name;
        }
    }
    if (input_name_.empty()) input_name_ = engine_->getIOTensorName(0);
    if (output_name_.empty()) output_name_ = engine_->getIOTensorName(nb - 1);
}

bool ImageClsTrt::prepare() {
    if (ready_) return true;
    input_bytes_ = 1 * 3 * 32 * 32 * sizeof(float);
    output_bytes_ = 1 * numClasses() * sizeof(float);
    cudaMalloc(&device_input_, input_bytes_);
    cudaMalloc(&device_output_, output_bytes_);
    bool ok1 = context_->setTensorAddress(input_name_.c_str(), device_input_);
    bool ok2 = context_->setTensorAddress(output_name_.c_str(), device_output_);
    ready_ = ok1 && ok2;
    return ready_;
}

bool ImageClsTrt::infer(const float *host_input, float *host_output) {
    if (!ready_) return false;
    cudaMemcpyAsync(device_input_, host_input, input_bytes_, cudaMemcpyHostToDevice, stream_);
    if (!context_->enqueueV3(stream_)) return false;
    cudaMemcpyAsync(host_output, device_output_, output_bytes_, cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    return true;
}
