#ifndef FACE_DETECT_CORE_HPP
#define FACE_DETECT_CORE_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace face {
struct Detection {
    cv::Rect2f box;
    std::array<cv::Point2f, 5> points{};
    float score = 0;
    std::vector<float> embedding;
};
struct Sample {
    int64_t person = 0;
    std::string name;
    std::vector<float> embedding;
};
struct Match {
    int64_t person = 0;
    std::string name = "陌生人";
    float similarity = 0;
    bool matched = false;
};
struct Config {
    std::string detector_engine;
    std::string recognizer_engine;
    std::string gallery_file;
    unsigned interval = 0;
    float detection_threshold = .5f;
    float nms_threshold = .4f;
    float recognition_threshold = .4f;
};
struct DetectionInput { cv::Mat blob; float scale; };
Config load_config(const std::string &path);
DetectionInput detector_input(const cv::Mat &image);
cv::Mat recognition_input(const cv::Mat &image, float mean, float stddev);
cv::Mat alignment_matrix(const std::array<cv::Point2f, 5> &points);
cv::Mat align_face(const cv::Mat &image, const Detection &face);
std::vector<Detection> nms(std::vector<Detection> faces, float threshold);
std::vector<Detection> decode_scrfd(const std::vector<std::vector<float>> &outputs,
                                    float scale, float threshold, float nms_threshold);
bool normalize(std::vector<float> &embedding);
Match match(const std::vector<float> &embedding, const std::vector<Sample> &gallery,
            float threshold);
std::string sha256_file(const std::string &path);
std::string runtime_identity(int gpu_id);

class Engine {
public:
    Engine(const std::string &path, int gpu_id);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    std::vector<std::vector<float>> run(const cv::Mat &blob);
    const std::string &source_hash() const;
    const std::string &role() const;
    float mean() const;
    float stddev() const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
class Pipeline {
public:
    Pipeline(const Config &config, int gpu_id);
    std::vector<Detection> analyze(const cv::Mat &image);
    std::string model_id() const;
    void warmup();
private:
    Engine detector_;
    Engine recognizer_;
    float mean_ = 127.5f;
    float stddev_ = 127.5f;
    float detection_threshold_;
    float nms_threshold_;
};

class Gallery {
public:
    explicit Gallery(const std::string &path, bool readonly);
    ~Gallery();
    Gallery(const Gallery &) = delete;
    Gallery &operator=(const Gallery &) = delete;
    int64_t add_person(const std::string &name);
    bool add_sample(int64_t person, const std::string &source, const std::string &hash,
                    const std::string &model, const std::vector<float> &embedding);
    std::vector<Sample> samples(const std::string &model) const;
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
} // namespace face
#endif
