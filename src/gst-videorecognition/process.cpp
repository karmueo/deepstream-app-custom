#include "process.h"
#include <dirent.h>
#include <filesystem>
#include <fstream>

Process::Process(int max_history_frames)
{
    m_input_size = cv::Size(224, 224);             // 模型输入尺寸
    m_max_history_frames = max_history_frames;     // 最大保存历史帧数
    m_vTargetFrames.reserve(m_max_history_frames); // 预留空间
    m_current_dir_num = 0;                         // 当前目录编号

#ifdef SAVE_IMAGES
    // 创建文件夹 ./vidoe_recognition_data/{m_current_dir_num}
    m_current_dir_path = "./video_recognition_data/" + std::to_string(m_current_dir_num);
    if (!std::filesystem::exists(m_current_dir_path))
    {
        std::filesystem::create_directories(m_current_dir_path);
    }
#endif
}

Process::~Process()
{
}

void Process::addFrame(const cv::Mat &frame)
{
    // 控制视频帧数量
    if (m_vTargetFrames.size() >= m_max_history_frames)
    {
        m_vTargetFrames.erase(m_vTargetFrames.begin());
    }

#ifdef SAVE_IMAGES
    if (m_vRawTargetFrames.size() >= m_max_history_frames)
    {
        m_vRawTargetFrames.erase(m_vRawTargetFrames.begin());
    }
    m_vRawTargetFrames.push_back(frame);
#endif

    cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    cv::Mat norm_img;
    // 判断尺寸是否等于输入尺寸
    if (frame.rows != m_input_size.height || frame.cols != m_input_size.width)
    {
        cv::Mat resizeMat = resizeWithAspectRatio(frame, m_input_size);
        // norm_img = half_norm(resizeMat);
        m_vTargetFrames.push_back(resizeMat);
    }
    else
    {
        // norm_img = half_norm(frame);
        m_vTargetFrames.push_back(frame);
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
    assert(frames[0].rows == height && frames[0].cols == width);

    auto actual_frames = frames.size();

    // 内存布局转换 [batch, clip_len, channel, height, width]
    for (int t = 0; t < clip_len; ++t)
    {
        // 选择帧（超出范围用最后一帧）
        const cv::Mat &frame = frames[std::min(t, static_cast<int>(actual_frames - 1))];

        // 转换为RGB格式
        cv::cvtColor(frame, frame, cv::COLOR_BGR2RGB);
        // 判断尺寸是否等于输入尺寸
        if (frame.rows != m_input_size.height || frame.cols != m_input_size.width)
        {
            cv::Mat resizeMat = resizeWithAspectRatio(frame, m_input_size);
        }

        // 归一化
        cv::Mat norm_img = half_norm(frame);

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

void Process::convertCvInputToTensorRT(std::vector<float> &input_data,
                                       const int &clip_len,
                                       const int &height,
                                       const int &width,
                                       const int &frame_interval)
{
    // 参数校验
    assert(m_vTargetFrames[0].channels() == 3);
    assert(m_vTargetFrames[0].rows == height && m_vTargetFrames[0].cols == width);

    auto ori_clip_len = (clip_len - 1) * frame_interval + 1;
    auto clip_offset = std::max(static_cast<int>(m_vTargetFrames.size() - ori_clip_len), 0);
    clip_offset = std::floor(clip_offset / 2.0f);
    // 计算实际需要提取的帧范围
    int start_frame = static_cast<int>(clip_offset);
    int end_frame = start_frame + ori_clip_len;
    // 确保不越界
    end_frame = std::min(end_frame, static_cast<int>(m_vTargetFrames.size()));
    // 提取帧到新 vector（高效连续存储）
    std::vector<cv::Mat> extracted_frames;
    extracted_frames.reserve(end_frame - start_frame); // 预分配内存
    for (int i = start_frame; i < end_frame; i += frame_interval)
    {
        extracted_frames.push_back(m_vTargetFrames[i].clone()); // 深拷贝避免原数据被修改
    }

    auto actual_frames = extracted_frames.size();
    assert(actual_frames == clip_len);

    // vec_input_data为指针数组，存储每一帧的图像数据
    std::vector<float *> vec_input_data(clip_len);
    vec_input_data.resize(clip_len);
    const int image_len = width * height * 3;
    for (int i = 0; i < clip_len; ++i)
    {
        vec_input_data[i] = new float[image_len];
    }

    for (int i = 0; i < clip_len; ++i)
    {
        // 把每一帧的图像数据进行归一化处理后存储到vec_input_data中
        preprocess3(extracted_frames[i], vec_input_data[i]);
    }

    const size_t total_elements = 1 * clip_len * 3 * width * height;
    std::vector<float> output(total_elements);
    for (int i = 0; i < clip_len; ++i)
    {
        for (int j = 0; j < image_len; ++j)
        {
            // vec_input_data[i]表示第i张图，j表示该图的第j个像素点
            // 计算在一维数组中的位置
            output[i * image_len + j] = vec_input_data[i][j];
        }
        delete[] vec_input_data[i];
    }
    input_data = output;

#ifdef SAVE_IMAGES
    // 遍历保存m_vRawTargetFrames
    for (int i = 0; i < m_vRawTargetFrames.size(); ++i)
    {
        cv::imwrite(m_current_dir_path + "/" + std::to_string(i) + ".jpg", m_vRawTargetFrames[i]);
    }
#endif
}

void Process::loadImagesFromDirectory(
    const std::string &directoryPath,
    std::vector<float> &input_data,
    int clip_len,
    int height,
    int width,
    int frame_interval)
{
    // 1. 检查文件夹是否存在
    DIR *dir = opendir(directoryPath.c_str());
    if (!dir)
    {
        return;
    }

    // 2. 遍历文件夹，收集所有 .jpg 文件
    std::vector<std::string> filenames;
    struct dirent *entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string filename = entry->d_name;
        if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".jpg")
        {
            filenames.push_back(filename);
        }
    }
    closedir(dir);

    // 3. 按数字顺序排序文件名（0.jpg, 1.jpg, ..., 255.jpg）
    std::sort(filenames.begin(), filenames.end(), [](const std::string &a, const std::string &b)
              {
        int num_a = std::stoi(a.substr(0, a.find('.')));
        int num_b = std::stoi(b.substr(0, b.find('.')));
        return num_a < num_b; });

    // 4. 按顺序读取图片到 vector<cv::Mat>
    std::vector<cv::Mat> images;
    for (const auto &filename : filenames)
    {
        std::string full_path = directoryPath + filename;
        cv::Mat img = cv::imread(full_path, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cerr << "Warning: Failed to load image " << full_path << std::endl;
            continue;
        }
        // cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // cv::Mat norm_img;
        // norm_img = half_norm(img);
        images.push_back(img);
    }

    auto ori_clip_len = (clip_len - 1) * frame_interval + 1;
    auto clip_offset = std::max(static_cast<int>(images.size() - ori_clip_len), 0);
    clip_offset = std::floor(clip_offset / 2.0f);
    // 计算实际需要提取的帧范围
    int start_frame = static_cast<int>(clip_offset);
    int end_frame = start_frame + ori_clip_len;
    // 确保不越界
    end_frame = std::min(end_frame, static_cast<int>(images.size()));

    // 提取帧到新 vector（高效连续存储）
    std::vector<cv::Mat> extracted_frames;
    extracted_frames.reserve(end_frame - start_frame); // 预分配内存
    for (int i = start_frame; i < end_frame; i += frame_interval)
    {
        extracted_frames.push_back(images[i].clone()); // 深拷贝避免原数据被修改
    }

    auto actual_frames = extracted_frames.size();
    assert(actual_frames == clip_len);
    // 将帧数据拷贝到 GPU（NCTHW 格式）
    // std::vector<float> cpu_data(1 * 3 * clip_len * height * width);
    // input_data = build_input_tensor(extracted_frames, clip_len, height, width, true);

    std::vector<float *> vec_input_data(clip_len);
    vec_input_data.resize(clip_len);
    const int image_len = width * height * 3;
    for (int i = 0; i < clip_len; ++i)
    {
        vec_input_data[i] = new float[image_len];
    }

    for (int i = 0; i < clip_len; ++i)
    {
        preprocess3(extracted_frames[i], vec_input_data[i]);
    }

    const size_t total_elements = 1 * clip_len * 3 * width * height;
    std::vector<float> output(total_elements);
    for (int i = 0; i < clip_len; ++i)
    {
        for (int j = 0; j < image_len; ++j)
        {
            output[i * image_len + j] = vec_input_data[i][j];
        }
        delete[] vec_input_data[i];
    }
    input_data = output;

    // SaveVectorToTxt(input_data, "input_data.txt");
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
                auto v = img_cp.at<cv::Vec3b>(h, w)[c];
                auto res = (((float)img_cp.at<cv::Vec3b>(h, w)[c]) - m_mean_vals[c]) * m_norm_vals[c];
                ptr[h * img_w * 3 + w * 3 + c] =
                    cv::saturate_cast<float>(res);
            }
        }
    }
    return norm_img;
}

float Process::get_pixel_value(const std::vector<cv::Mat> &images, const int &c, const int &t, const int &h, const int &w)
{
    if (t >= images.size())
    {
        std::cerr << "Error: Frame index out of range!" << std::endl;
        return 0.0f;
    }
    cv::Mat img = images[t];
    if (img.empty())
    {
        std::cerr << "Error: Image is empty!" << std::endl;
        return 0.0f;
    }
    float pixel_value = 0.0f;
    pixel_value = img.at<cv::Vec3f>(h, w)[c];
    return pixel_value;
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
        sampled_frames.push_back(sampled_frame);
    }

    return sampled_frames;
}

void Process::clearFrames()
{
    m_vTargetFrames.clear();
    m_vRawTargetFrames.clear();

#ifdef SAVE_IMAGES
    m_current_dir_num++;
    m_current_dir_path = "./video_recognition_data/" + std::to_string(m_current_dir_num);
    if (!std::filesystem::exists(m_current_dir_path))
    {
        std::filesystem::create_directories(m_current_dir_path);
    }
#endif
}

float Process::IOU(const cv::Rect &srcRect, const cv::Rect &dstRect)
{
    cv::Rect intersection;
    intersection = srcRect & dstRect;

    auto area_src = static_cast<float>(srcRect.area());
    auto area_dst = static_cast<float>(dstRect.area());
    auto area_intersection = static_cast<float>(intersection.area());
    float iou = area_intersection / (area_src + area_dst - area_intersection);
    return iou;
}

void Process::SaveVectorToTxt(const std::vector<float> &data, const std::string &filename)
{
    std::ofstream outfile(filename);
    if (!outfile.is_open())
    {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    // 每行写入一个数值
    for (const float value : data)
    {
        outfile << value << std::endl;
    }

    outfile.close();
    std::cout << "Data saved to " << filename << std::endl;
}

std::vector<float> Process::build_input_tensor(
    const std::vector<cv::Mat> &images,
    int clip_len,
    int height,
    int width,
    bool bgr_to_rgb)
{
    // 参数检查
    {
        if (images.size() != clip_len)
            throw std::runtime_error("Input must contain exactly 16 images!");
    }

    // 计算总元素数量
    const size_t total_elements = 1 * clip_len * 3 * width * height;
    std::vector<float> output(total_elements);

    // 指针指向输出数据的起始位置
    float *data_ptr = output.data();

    // 遍历每张图像
    for (const cv::Mat &img : images)
    {
        // 检查图像尺寸和通道
        if (img.cols != width || img.rows != height || img.channels() != 3)
        {
            throw std::runtime_error("Images must be 224x224 with 3 channels!");
        }

        cvtColor(img, img, cv::COLOR_BGR2RGB); // 转RGB
        cv::Mat img_float;
        img.convertTo(img_float, CV_32F, 1.0 / 255); // divided by 255转float

        // 遍历每个像素（H=224, W=224）
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                auto pixel = img_float.at<cv::Vec3f>(h, w);

                data_ptr[0] = ((float)pixel[0] - m_mean_vals[0]) / m_norm_vals[0];
                data_ptr[1] = ((float)pixel[1] - m_mean_vals[1]) / m_norm_vals[1];
                data_ptr[2] = ((float)pixel[2] - m_mean_vals[2]) / m_norm_vals[2];

                // 移动指针到下一个空间位置
                data_ptr += 3;
            }
        }
    }

    return output;
}

void Process::preprocess3(const cv::Mat &srcframe, float *inputTensorValues)
{
    //=====================================================================
    /*std::vector<float> mean_ = { 0.485f, 0.456f, 0.406f };
    std::vector<float> std_vals = { 0.229f, 0.224f, 0.225f };
    cv::Size input_WH(750, 500);*/

    cv::Mat img = srcframe.clone();
    cv::Mat img_float;

    // cvtColor(srcframe, imgRGBresize, cv::COLOR_BGR2RGB);  // 转RGB
    srcframe.convertTo(img_float, CV_32F, 1.0 / 255); // divided by 255转float
    std::vector<cv::Mat> channels(3);                 // cv::Mat channels[3]; //分离通道进行HWC->CHW
    cv::Mat dst;
    cv::split(img_float, channels);

    for (int i = 0; i < img_float.channels(); i++) // 标准化ImageNet
    {
        channels[i] -= m_mean_vals[i]; // mean均值
        channels[i] /= m_norm_vals[i]; // std方差
    }
    cv::merge(channels, dst);
    int img_float_len = img_float.cols * img_float.rows;

    for (int i = 0; i < img_float.rows; i++)
    {
        float *pixel = dst.ptr<float>(i);
        for (int j = 0; j < img_float.cols; j++)
        {
            inputTensorValues[i * img_float.cols + j] = pixel[0];
            inputTensorValues[1 * img_float_len + i * img_float.cols + j] = pixel[1];
            inputTensorValues[2 * img_float_len + i * img_float.cols + j] = pixel[2];
            pixel += 3;
        }
    }
}