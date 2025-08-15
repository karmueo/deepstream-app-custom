#ifndef IMAGE_CLS_TRT_H
#define IMAGE_CLS_TRT_H

#include "videorecognitionTrt.h"
#include <vector>

/* 简单单帧分类模型封装：输入 [1,3,224,224]，输出 [1,3] softmax */
class ImageClsTrt : public VideoRecognitionTRT {
public:
    explicit ImageClsTrt(const std::string &engine_path);
    ~ImageClsTrt() override;

    bool prepare();               // 绑定输入输出缓冲区（一次）
    bool infer(const float *host_input, float *host_output); // 同步推理
    inline int inputH() const { return 224; }
    inline int inputW() const { return 224; }
    inline int numClasses() const { return 3; }
    inline const std::string &inputName() const { return input_name_; }
    inline const std::string &outputName() const { return output_name_; }

private:
    void discoverIO();
    void *device_input_ = nullptr;
    void *device_output_ = nullptr;
    size_t input_bytes_ = 0;
    size_t output_bytes_ = 0;
    std::string input_name_;
    std::string output_name_;
    bool ready_ = false;
};

#endif // IMAGE_CLS_TRT_H
