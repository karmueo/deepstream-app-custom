#ifndef __VIDEORECOGNITION_PROCESS_H__
#define __VIDEORECOGNITION_PROCESS_H__

#include <opencv2/opencv.hpp>

std::vector<std::vector<std::vector<float>>> reshape_to_3d(const float *output_data, const std::vector<int> &shape);

// 定义一个类，该类主要实现对视频帧的各种自定义预处理
class Process
{
public:
    Process(int max_history_frames = 300);
    ~Process();

    // 添加视频帧
    void addFrame(const cv::Mat &frame);

    // 测试函数
    void testFunc(const std::string &videoPath);

    void convertCvInputToTensorRT(std::vector<float> &input_data,
                                  const int &clip_len,
                                  const int &height,
                                  const int &width,
                                  const int &frame_interval);

    void convertCvInputToNtchwTensorRT(std::vector<float> &input_data,
                                       const int &num_clips,
                                       const int &clip_len,
                                       const int &height,
                                       const int &width,
                                       const int &frame_interval);

    void loadImagesFromDirectory(const std::string &directoryPath,
                                 std::vector<float> &input_data,
                                 int clip_len,
                                 int height,
                                 int width,
                                 int frame_interval);

    void loadImagesFromDirectory2(const std::string &directoryPath,
                                  std::vector<float> &input_data,
                                  int num_clips,
                                  int clip_len,
                                  int height,
                                  int width);

    // X3D 预处理：32帧，64x64，中心裁剪，归一化
    void convertCvInputToX3dTensorRT(std::vector<float> &input_data,
                                    const int &num_frames,
                                    const int &height,
                                    const int &width,
                                    const int &sampling_rate);

    // 获取当前视频帧长度
    int getCurrentFrameLength() const
    {
        return m_vTargetFrames.size();
    }

    // 清空释放视频帧
    void clearFrames();

    static float IOU(const cv::Rect &srcRect, const cv::Rect &dstRect);

    void SaveVectorToTxt(const std::vector<float> &data, const std::string &filename);

    std::vector<float> build_input_tensor(const std::vector<cv::Mat> &images,
                                          int clip_len,
                                          int height,
                                          int width,
                                          bool bgr_to_rgb);

    void preprocess3(const cv::Mat &srcframe, float *inputTensorValues);

private:
    // 输入一张图片，和缩放尺寸，对其进行保持长宽比的缩放，其他区域用黑色填充
    cv::Mat resizeWithAspectRatio(const cv::Mat &src, const cv::Size &size);

    cv::Mat resize(const cv::Mat &src, const cv::Size &size);

    cv::Mat half_norm(const cv::Mat &img);

    std::vector<cv::Mat> sampleFrames(const std::vector<cv::Mat> &images,
                                      const int &num_samples,
                                      const int &clip_len,
                                      const int &frame_interval);
    std::vector<cv::Mat> sampleFrames2(const std::vector<cv::Mat> &images,
                                       const int &num_samples,
                                       const int &clip_len,
                                       const int &frame_interval);

    std::vector<std::vector<cv::Mat>> getSampleClips(const std::vector<cv::Mat> &src_images, int num_clips, int clip_len);

    float get_pixel_value(const std::vector<cv::Mat> &images, const int &c, const int &t, const int &h, const int &w);

    cv::Mat preprocess(const cv::Mat &srcframe);

    // HACK:从本地文件读取输入，测试用
    std::vector<float> loadDataFromFile(const std::string &txt_path);

private:
    std::vector<cv::Mat> m_vTargetFrames; // 视频帧,RGB,224x224
    std::vector<cv::Mat> m_vRawTargetFrames;

    const float m_mean_vals[3] = {0.45f, 0.45f, 0.45f}; // RGB
    const float m_norm_vals[3] = {0.225f, 0.225f, 0.225f};

    // 模型输入尺寸
    cv::Size m_input_size;

    int m_max_history_frames; // 最大保存历史帧数

    int m_current_dir_num;
    std::string m_current_dir_path;
};

#endif /* __VIDEORECOGNITION_PROCESS_H__ */