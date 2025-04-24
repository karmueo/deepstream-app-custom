#ifndef __VIDEORECOGNITION_PROCESS_H__
#define __VIDEORECOGNITION_PROCESS_H__

#include <opencv2/opencv.hpp>

// 定义一个类，该类主要实现对视频帧的各种自定义预处理
class Process
{
public:
    Process(int max_history_frames = 300);
    ~Process();

    // 添加视频帧
    void addFrame(const cv::Mat &frame);

    // 视频帧采样
    std::vector<cv::Mat> sampleFrames(const int &num_samples);

    // 测试函数
    void testFunc(const std::string &videoPath);

    void convertCvInputToTensorRT(const std::vector<cv::Mat> &frames,
                                  float *input_data,
                                  int clip_len,
                                  int height,
                                  int width);

    void convertCvInputToTensorRT(float *input_data,
                                  int clip_len,
                                  int height,
                                  int width);

    // 获取当前视频帧长度
    int getCurrentFrameLength() const
    {
        return m_vTargetFrames.size();
    }

    // 清空释放视频帧
    void clearFrames()
    {
        m_vTargetFrames.clear();
    }

private:
    // 输入一张图片，和缩放尺寸，对其进行保持长宽比的缩放，其他区域用黑色填充
    cv::Mat resizeWithAspectRatio(const cv::Mat &src, const cv::Size &size);

    cv::Mat half_norm(const cv::Mat &img);

private:
    // 目标视频帧vector
    std::vector<cv::Mat> m_vTargetFrames;

    const float m_mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float m_norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};

    // 模型输入尺寸
    cv::Size m_input_size;

    int m_max_history_frames; // 最大保存历史帧数
};

#endif /* __VIDEORECOGNITION_PROCESS_H__ */