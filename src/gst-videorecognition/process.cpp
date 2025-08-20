#include "process.h"
#include <dirent.h>
#include <filesystem>
#include <fstream>
#include <chrono>

Process::Process(int max_history_frames)
{
    m_input_size = cv::Size(32, 32);             // 模型输入尺寸
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
    if (m_vRawTargetFrames.size() >= 60)
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
        // TODO:
        // cv::Mat resizeMat = resizeWithAspectRatio(frame, m_input_size);
        cv::Mat resizeMat = resize(frame, m_input_size);
        // norm_img = half_norm(resizeMat);
        m_vTargetFrames.push_back(resizeMat);
    }
    else
    {
        // norm_img = half_norm(frame);
        m_vTargetFrames.push_back(frame);
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

    // auto ori_clip_len = (clip_len - 1) * frame_interval + 1;
    // auto clip_offset = std::max(static_cast<int>(m_vTargetFrames.size() - ori_clip_len), 0);
    // clip_offset = std::floor(clip_offset / 2.0f);
    // // 计算实际需要提取的帧范围
    // int start_frame = static_cast<int>(clip_offset);
    // int end_frame = start_frame + ori_clip_len;
    // // 确保不越界
    // end_frame = std::min(end_frame, static_cast<int>(m_vTargetFrames.size()));
    // // 提取帧到新 vector（高效连续存储）
    // std::vector<cv::Mat> extracted_frames;
    // extracted_frames.reserve(end_frame - start_frame); // 预分配内存
    // for (int i = start_frame; i < end_frame; i += frame_interval)
    // {
    //     extracted_frames.push_back(m_vTargetFrames[i].clone()); // 深拷贝避免原数据被修改
    // }

    std::vector<cv::Mat> extracted_frames = sampleFrames2(m_vTargetFrames, 1, clip_len, frame_interval);
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

void Process::convertCvInputToNtchwTensorRT(std::vector<float> &input_data,
                                            const int &num_clips,
                                            const int &clip_len,
                                            const int &height,
                                            const int &width,
                                            const int &frame_interval)
{
    // 参数校验
    assert(m_vTargetFrames[0].channels() == 3);
    assert(m_vTargetFrames[0].rows == height && m_vTargetFrames[0].cols == width);

    std::vector<std::vector<cv::Mat>> extracted_num_frames = getSampleClips(m_vTargetFrames, num_clips, clip_len);

    auto actual_num_clips = extracted_num_frames.size();
    assert(actual_num_clips == num_clips);
    auto actual_frames = extracted_num_frames[actual_num_clips - 1].size();
    assert(actual_frames == clip_len);

    std::vector<float> output(1 * num_clips * 3 * clip_len * height * width);
    for (int i = 0; i < num_clips; ++i)
    {
        for (int c = 0; c < 3; ++c)
        {
            for (int t = 0; t < clip_len; ++t)
            {
                for (int h = 0; h < height; ++h)
                {
                    for (int w = 0; w < width; ++w)
                    {
                        // 计算目标位置 [1][t][c][h][w]
                        int dst_idx =
                            i * (3 * clip_len * height * width) + c * (clip_len * height * width) + t * (height * width) + h * width + w;
                        output[dst_idx] = extracted_num_frames[i][t].at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
        }
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

    // 4. 按顺序读取图片到 vector<cv::Mat> 作为m_vTargetFrames的仿真
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
        cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
        // cv::Mat norm_img;
        // norm_img = half_norm(img);
        images.push_back(img);
    }

    // 总共需要多少帧
    auto ori_clip_len = (clip_len - 1) * frame_interval + 1;
    // 计算偏移量，队列中的帧数大于ori_clip_len时，取中间的ori_clip_len帧
    auto clip_offset = std::max(static_cast<int>(images.size() - ori_clip_len), 0);
    clip_offset = std::floor(clip_offset / 2.0f);
    // 计算实际需要提取的帧范围
    int start_frame = static_cast<int>(clip_offset);
    int end_frame = start_frame + ori_clip_len;
    // 确保不越界
    end_frame = std::min(end_frame, static_cast<int>(images.size()));

    // NOTE: 方法1
    // 将帧数据拷贝到 GPU（NCTHW 格式）
    // std::vector<float> cpu_data(1 * 3 * clip_len * height * width);
    // input_data = build_input_tensor(extracted_frames, clip_len, height, width, true);
    // 提取帧到新 vector（高效连续存储）
    auto start_time = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> extracted_frames = sampleFrames2(images, 1, clip_len, frame_interval);
    auto actual_frames = extracted_frames.size();
    assert(actual_frames == clip_len);
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
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "1. Time taken to process frames: " << duration.count() << " ms" << std::endl;
    // input_data = output;

    // NOTE: 方法2
    start_time = std::chrono::high_resolution_clock::now();
    std::vector<cv::Mat> extracted_frames2 = sampleFrames2(images, 1, clip_len, frame_interval);

    auto actual_frames2 = extracted_frames2.size();
    assert(actual_frames2 == clip_len);

    std::vector<float> output2(1 * 3 * clip_len * height * width);
    for (int t = 0; t < clip_len; ++t)
    {
        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    // 计算目标位置 [1][t][c][h][w]
                    int dst_idx = t * (3 * height * width) + c * (height * width) + h * width + w;
                    output2[dst_idx] = extracted_frames2[t].at<cv::Vec3f>(h, w)[c];
                }
            }
        }
    }
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    std::cout << "2. Time taken to process frames: " << duration.count() << " ms" << std::endl;

    assert(output2.size() == output.size());
    // 计算 ouput 和 output2 的差异
    for (size_t i = 0; i < output.size(); ++i)
    {
        if (std::abs(output[i] - output2[i]) > 1e-5)
        {
            std::cout << "Difference at index " << i << ": " << output[i] << " vs " << output2[i] << std::endl;
        }
    }

    // 将数据拷贝到 input_data
    input_data = output2;

    // SaveVectorToTxt(input_data, "input_data.txt");
}

void Process::loadImagesFromDirectory2(
    const std::string &directoryPath,
    std::vector<float> &input_data,
    int num_clips,
    int clip_len,
    int height,
    int width)
{
    // 1. 检查文件夹是否存在
    // DIR *dir = opendir(directoryPath.c_str());
    // if (!dir)
    // {
    //     return;
    // }

    // // 2. 遍历文件夹，收集所有 .jpg 文件
    // std::vector<std::string> filenames;
    // struct dirent *entry;
    // while ((entry = readdir(dir)) != nullptr)
    // {
    //     std::string filename = entry->d_name;
    //     if (filename.size() >= 4 && filename.substr(filename.size() - 4) == ".jpg")
    //     {
    //         filenames.push_back(filename);
    //     }
    // }
    // closedir(dir);

    // // 3. 按数字顺序排序文件名（0.jpg, 1.jpg, ..., 255.jpg）
    // std::sort(filenames.begin(), filenames.end(), [](const std::string &a, const std::string &b)
    //           {
    //     int num_a = std::stoi(a.substr(0, a.find('.')));
    //     int num_b = std::stoi(b.substr(0, b.find('.')));
    //     return num_a < num_b; });

    // // 4. 按顺序读取图片到 vector<cv::Mat> 作为m_vTargetFrames的仿真
    // std::vector<cv::Mat> images;
    // for (const auto &filename : filenames)
    // {
    //     std::string full_path = directoryPath + filename;
    //     cv::Mat img = cv::imread(full_path, cv::IMREAD_COLOR);
    //     if (img.empty())
    //     {
    //         std::cerr << "Warning: Failed to load image " << full_path << std::endl;
    //         continue;
    //     }
    //     cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    //     // cv::Mat norm_img;
    //     // norm_img = half_norm(img);
    //     images.push_back(img);
    // }

    // std::vector<std::vector<cv::Mat>> extracted_num_frames = getSampleClips(images, num_clips, clip_len);

    // auto actual_num_clips = extracted_num_frames.size();
    // assert(actual_num_clips == num_clips);
    // auto actual_frames = extracted_num_frames[actual_num_clips - 1].size();
    // assert(actual_frames == clip_len);

    // std::vector<float> output(1 * num_clips * 3 * clip_len * height * width);
    // for (int i = 0; i < num_clips; ++i)
    // {
    //     for (int c = 0; c < 3; ++c)
    //     {
    //         for (int t = 0; t < clip_len; ++t)
    //         {
    //             for (int h = 0; h < height; ++h)
    //             {
    //                 for (int w = 0; w < width; ++w)
    //                 {
    //                     // 计算目标位置 [1][t][c][h][w]
    //                     int dst_idx =
    //                         i * (3 * clip_len * height * width) + c * (clip_len * height * width) + t * (height * width) + h * width + w;
    //                     output[dst_idx] = extracted_num_frames[i][t].at<cv::Vec3f>(h, w)[c];
    //                 }
    //             }
    //         }
    //     }
    // }
    // input_data = output;

    input_data = loadDataFromFile("/workspace/deepstream-app-custom/src/gst-videorecognition/input_data.txt");

    // SaveVectorToTxt(input_data, "/workspace/deepstream-app-custom/src/gst-videorecognition/input_data.txt");
}

std::vector<cv::Mat> Process::sampleFrames(const std::vector<cv::Mat> &images,
                                           const int &num_samples, const int &clip_len, const int &frame_interval)
{
    // 总共需要多少帧
    auto ori_clip_len = (clip_len - 1) * frame_interval + 1;
    // 计算偏移量，队列中的帧数大于ori_clip_len时，取中间的ori_clip_len帧
    auto clip_offset = std::max(static_cast<int>(images.size() - ori_clip_len), 0);
    clip_offset = std::floor(clip_offset / 2.0f);
    // 计算实际需要提取的帧范围
    int start_frame = static_cast<int>(clip_offset);
    int end_frame = start_frame + ori_clip_len;
    // 确保不越界
    end_frame = std::min(end_frame, static_cast<int>(images.size()));
    std::vector<cv::Mat> extracted_frames;
    extracted_frames.reserve(end_frame - start_frame); // 预分配内存
    for (int i = start_frame; i < end_frame; i += frame_interval)
    {
        extracted_frames.push_back(images[i].clone()); // 深拷贝避免原数据被修改
    }

    return extracted_frames;
}

std::vector<cv::Mat> Process::sampleFrames2(const std::vector<cv::Mat> &images, const int &num_samples, const int &clip_len, const int &frame_interval)
{
    // 总共需要多少帧
    auto ori_clip_len = (clip_len - 1) * frame_interval + 1;
    // 计算偏移量，队列中的帧数大于ori_clip_len时，取中间的ori_clip_len帧
    auto clip_offset = std::max(static_cast<int>(images.size() - ori_clip_len), 0);
    clip_offset = std::floor(clip_offset / 2.0f);
    // 计算实际需要提取的帧范围
    int start_frame = static_cast<int>(clip_offset);
    int end_frame = start_frame + ori_clip_len;
    // 确保不越界
    end_frame = std::min(end_frame, static_cast<int>(images.size()));
    std::vector<cv::Mat> extracted_frames;
    extracted_frames.reserve(end_frame - start_frame); // 预分配内存
    for (int i = start_frame; i < end_frame; i += frame_interval)
    {
        cv::Mat img_float = preprocess(images[i]);
        extracted_frames.push_back(img_float);
    }

    return extracted_frames;
}

std::vector<std::vector<cv::Mat>> Process::getSampleClips(
    const std::vector<cv::Mat> &src_images,
    int num_clips,
    int clip_len)
{
    std::vector<std::vector<cv::Mat>> clips;  // 存储所有片段
    const int num_frames = src_images.size(); // 视频总帧数

    // 检查输入有效性
    if (num_frames < clip_len || num_clips <= 0 || clip_len <= 0)
    {
        return clips; // 返回空结果
    }

    // 计算每个片段的基准长度（浮点数精度）
    const float seg_size = float(num_frames - 1) / clip_len;
    // 计算片段内采样点间隔（均匀分布）
    const float duration = seg_size / (num_clips + 1);

    // 遍历每个片段
    for (int k = 0; k < num_clips; ++k)
    {
        std::vector<cv::Mat> current_clip; // 当前片段的帧集合

        // 采样当前片段的每一帧
        for (int i = 0; i < clip_len; ++i)
        {
            // 计算当前帧在原始视频中的起始位置
            const int start = static_cast<int>(std::round(seg_size * i));
            // 计算精确索引（考虑片段偏移）
            const int frame_index = start + static_cast<int>(duration * (k + 1));
            // 确保索引不越界
            const int safe_index = std::min(frame_index, num_frames - 1);

            // 将帧加入当前片段
            // 进行预处理
            cv::Mat img_float = preprocess(src_images[safe_index]);
            current_clip.push_back(img_float);
        }

        // 将当前片段加入结果集
        clips.push_back(current_clip);
    }

    return clips;
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
    cv::Mat img_float;
    img.convertTo(img_float, CV_32F, 1.0 / 255); // divided by 255转float
    std::vector<cv::Mat> channels(3);            // cv::Mat channels[3]; //分离通道进行HWC->CHW
    cv::Mat dst;
    cv::split(img_float, channels);

    for (int i = 0; i < img_float.channels(); i++) // 标准化ImageNet
    {
        channels[i] -= m_mean_vals[i]; // mean均值
        channels[i] /= m_norm_vals[i]; // std方差
    }
    cv::merge(channels, dst);

    float pixel_value = 0.0f;
    pixel_value = dst.at<cv::Vec3f>(h, w)[c];
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

cv::Mat Process::resize(const cv::Mat &src, const cv::Size &size)
{
    // 直接拉伸到目标尺寸
    cv::Mat resizedImage;
    cv::resize(src, resizedImage, size, 0, 0, cv::INTER_LINEAR); // 使用线性插值
    
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

std::vector<float> Process::loadDataFromFile(const std::string &txt_path)
{
    // 1. 读取文件
    std::ifstream file(txt_path);
    if (!file.is_open())
    {
        throw std::runtime_error("无法打开文件: " + txt_path);
    }

    // 2. 读取所有行到 vector
    std::vector<float> data;
    std::string line;
    while (std::getline(file, line))
    {
        data.push_back(std::stof(line)); // 转换为 float
    }

    return data;
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

        // 遍历每个像素（H=32, W=32）
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

cv::Mat Process::preprocess(const cv::Mat &srcframe)
{
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

    return dst;
}

void Process::preprocess3(const cv::Mat &srcframe, float *inputTensorValues)
{
    //=====================================================================
    /*std::vector<float> mean_ = { 0.485f, 0.456f, 0.406f };
    std::vector<float> std_vals = { 0.229f, 0.224f, 0.225f };
    cv::Size input_WH(750, 500);*/

    // cv::Mat img = srcframe.clone();
    // cv::Mat img_float;

    // // cvtColor(srcframe, imgRGBresize, cv::COLOR_BGR2RGB);  // 转RGB
    // srcframe.convertTo(img_float, CV_32F, 1.0 / 255); // divided by 255转float
    // std::vector<cv::Mat> channels(3);                 // cv::Mat channels[3]; //分离通道进行HWC->CHW
    // cv::Mat dst;
    // cv::split(img_float, channels);

    // for (int i = 0; i < img_float.channels(); i++) // 标准化ImageNet
    // {
    //     channels[i] -= m_mean_vals[i]; // mean均值
    //     channels[i] /= m_norm_vals[i]; // std方差
    // }
    // cv::merge(channels, dst);
    // int img_float_len = img_float.cols * img_float.rows;
    // for (int i = 0; i < img_float.rows; i++)
    // {
    //     float *pixel = dst.ptr<float>(i);
    //     for (int j = 0; j < img_float.cols; j++)
    //     {
    //         inputTensorValues[i * img_float.cols + j] = pixel[0];
    //         inputTensorValues[1 * img_float_len + i * img_float.cols + j] = pixel[1];
    //         inputTensorValues[2 * img_float_len + i * img_float.cols + j] = pixel[2];
    //         pixel += 3;
    //     }
    // }
    int img_float_len = srcframe.cols * srcframe.rows;
    for (int i = 0; i < srcframe.rows; i++)
    {
        const float *pixel = srcframe.ptr<float>(i);
        for (int j = 0; j < srcframe.cols; j++)
        {
            inputTensorValues[i * srcframe.cols + j] = pixel[0];
            inputTensorValues[1 * img_float_len + i * srcframe.cols + j] = pixel[1];
            inputTensorValues[2 * img_float_len + i * srcframe.cols + j] = pixel[2];
            pixel += 3;
        }
    }
}

std::vector<std::vector<std::vector<float>>> reshape_to_3d(const float *output_data, const std::vector<int> &shape)
{
    // 检查形状合法性
    if (shape.size() != 3)
    {
        throw std::invalid_argument("Shape must have exactly 3 dimensions");
    }
    if (shape[0] <= 0 || shape[1] <= 0 || shape[2] <= 0)
    {
        throw std::invalid_argument("All shape dimensions must be positive");
    }

    const int total_elements = shape[0] * shape[1] * shape[2];
    std::vector<std::vector<std::vector<float>>> result(shape[0], std::vector<std::vector<float>>(shape[1], std::vector<float>(shape[2])));

    // 将一维数据填充到三维结构中
    int index = 0;
    for (int i = 0; i < shape[0]; ++i)
    {
        for (int j = 0; j < shape[1]; ++j)
        {
            for (int k = 0; k < shape[2]; ++k)
            {
                if (index >= total_elements)
                {
                    throw std::out_of_range("Output_data has fewer elements than required by shape");
                }
                result[i][j][k] = output_data[index++];
            }
        }
    }

    return result;
}