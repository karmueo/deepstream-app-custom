#include "process.h"

Process::Process()
{
    m_input_size = cv::Size(224, 224); // 模型输入尺寸
}

Process::~Process()
{
}

void Process::addFrame(const cv::Mat &frame)
{
    // 将视频帧添加到目标视频帧vector中
    m_vTargetFrames.push_back(frame);

    // 控制视频帧数量
    if (m_vTargetFrames.size() > 300)
    {
        m_vTargetFrames.erase(m_vTargetFrames.begin());
    }
}

void Process::convertCvInputToTensorRT(const std::vector<cv::Mat> &frames,
                                       float *input_data,
                                       int clip_len,
                                       int height,
                                       int width)
{
    // 参数校验
    assert(frames.size() <= clip_len);
    assert(frames[0].channels() == 3);
    assert(frames[0].type() == CV_32FC3);
    assert(frames[0].rows == height && frames[0].cols == width);

    auto actual_frames = frames.size();

    // 内存布局转换 [batch, clip_len, channel, height, width]
    for (int t = 0; t < clip_len; ++t)
    {
        // 选择帧（超出范围用最后一帧）
        const cv::Mat &frame = frames[std::min(t, static_cast<int>(actual_frames - 1))];

        // 获取当前帧在input_data中的起始位置
        float *frame_data = input_data + t * 3 * height * width;

        // 通道分离并交错存储 (OpenCV的HWC -> TensorRT的CHW)
        std::vector<cv::Mat> channels(3);
        cv::split(frame, channels);

        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    // 计算目标位置 [1][t][c][h][w]
                    int dst_idx = c * (height * width) + h * width + w;
                    frame_data[dst_idx] = channels[c].at<float>(h, w);
                }
            }
        }
    }
}

cv::Mat Process::half_norm(const cv::Mat &img)
{
    int channels = 3;
    int img_h = img.rows;
    int img_w = img.cols;

    cv::Mat img_cp;
    img_cp = img.clone();
    cv::Mat norm_img(img_h, img_w, CV_32FC3); // Mat自己分配内存
    float *ptr = norm_img.ptr<float>();
    for (size_t c = 0; c < channels; c++)
    {
        for (size_t h = 0; h < img_h; h++)
        {
            for (size_t w = 0; w < img_w; w++)
            {
                ptr[h * img_w * 3 + w * 3 + c] =
                    cv::saturate_cast<float>((((float)img_cp.at<cv::Vec3b>(h, w)[c]) - m_mean_vals[c]) * m_norm_vals[c]);
            }
        }
    }
    return norm_img;
}

cv::Mat Process::resizeWithAspectRatio(const cv::Mat &src, const cv::Size &size)
{
    // 计算缩放比例
    float scale = std::min(static_cast<float>(size.width) / src.cols,
                           static_cast<float>(size.height) / src.rows);

    // 计算新的尺寸
    cv::Size newSize(static_cast<int>(src.cols * scale), static_cast<int>(src.rows * scale));

    // 创建一个黑色背景的图像
    cv::Mat resizedImage(size, src.type(), cv::Scalar(0, 0, 0));

    // 将缩放后的图像放置在中心位置
    cv::Rect roi((size.width - newSize.width) / 2, (size.height - newSize.height) / 2, newSize.width, newSize.height);
    cv::resize(src, resizedImage(roi), newSize);

    return resizedImage;
}

void Process::testFunc(const std::string &videoPath)
{
    // 1. 打开视频文件
    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open video file " << videoPath << std::endl;
        return;
    }

    // 2. 获取视频信息
    int frameCount = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));

    // 3. 逐帧处理
    int processedFrames = 0;
    cv::Mat frame;
    while (cap.read(frame))
    {
        // 调用处理函数
        addFrame(frame);

        processedFrames++;

        // 打印进度（每10帧或关键帧）
        if (processedFrames % 10 == 0)
        {
            std::cout << "Processed " << processedFrames << " / " << frameCount
                      << " frames (" << (100.0 * processedFrames / frameCount) << "%)"
                      << std::endl;
        }
    }

    // 4. 释放资源
    cap.release();

    std::cout << "Finished processing. Total frames processed: "
              << processedFrames << std::endl;
}

std::vector<cv::Mat> Process::sampleFrames(const int &num_samples)
{
    // 目前总的视频帧数
    int total_frames = m_vTargetFrames.size();

    // 生成采样索引（避免浮点累计误差）
    std::vector<int> indices;

    // 如果总的视频帧数大于采样数，则进行采样
    if (total_frames > num_samples)
    {
        indices.reserve(num_samples);
        const float step = static_cast<float>(total_frames - 1) / (num_samples - 1);

        for (int i = 0; i < num_samples; ++i)
        {
            int idx = static_cast<int>(std::round(i * step));
            idx = std::clamp(idx, 0, total_frames - 1);
            indices.push_back(idx);
        }
    }
    else
    {
        // 如果总的视频帧数小于等于采样数，则直接添加所有帧
        for (int i = 0; i < total_frames; ++i)
        {
            indices.push_back(i);
        }
    }

    std::vector<cv::Mat> sampled_frames;
    sampled_frames.reserve(indices.size());

    // 采样视频帧
    for (const auto &idx : indices)
    {
        cv::Mat sampled_frame = m_vTargetFrames[idx].clone();
        // 转换为RGB格式
        cv::cvtColor(sampled_frame, sampled_frame, cv::COLOR_BGR2RGB);
        cv::Mat resized_frame = resizeWithAspectRatio(sampled_frame, m_input_size);
        // 归一化
        cv::Mat norm_img = half_norm(resized_frame);
        sampled_frames.push_back(norm_img);
    }

    return sampled_frames;
}
