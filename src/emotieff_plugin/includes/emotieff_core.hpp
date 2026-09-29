#ifndef EMOTIEFF_CORE_HPP
#define EMOTIEFF_CORE_HPP

#include <array>
#include <memory>
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace emotieff {

constexpr int kClasses = 8;

struct Config {
    std::string model = "enet_b0_8_best_vgaf";
    std::string engine;
    unsigned interval = 0;
};

struct Result {
    int index = -1;
    std::array<float, kClasses> probabilities{};
    std::vector<float> logits;
};

Config load_config(const std::string &path);
int input_size(const std::string &model);
const char *label(int index, bool chinese = true);
cv::Rect crop(const cv::Size &size, const cv::Rect2f &box);
cv::Mat make_input(const cv::Mat &bgr, const std::string &model);
Result scores(const std::vector<float> &logits, const std::string &model);
std::string sha256_file(const std::string &path);
std::string runtime_identity(int gpu_id);

class Engine {
public:
    Engine(const std::string &path, const std::string &model, int gpu_id);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    std::vector<float> run(const cv::Mat &input);
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

class Pipeline {
public:
    Pipeline(const Config &config, int gpu_id);
    Result predict(const cv::Mat &bgr);
    Result analyze(const cv::Mat &bgr, const cv::Rect2f &box);
    void warmup();
private:
    std::string model_;
    Engine engine_;
};

} // namespace emotieff
#endif
