#include "emotieff_core.hpp"
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

namespace emotieff {

int input_size(const std::string &model) {
    if (model == "enet_b0_8_best_vgaf") return 224;
    if (model == "mbf_va_mtl") return 112;
    throw std::runtime_error("unsupported emotion model: " + model);
}

const char *label(int index, bool chinese) {
    static constexpr const char *zh[] = {"愤怒", "轻蔑", "厌恶", "恐惧", "高兴", "中性", "悲伤", "惊讶"};
    static constexpr const char *en[] = {"Anger", "Contempt", "Disgust", "Fear", "Happiness",
                                         "Neutral", "Sadness", "Surprise"};
    return index >= 0 && index < kClasses ? (chinese ? zh[index] : en[index]) :
           (chinese ? "无法判断" : "");
}

cv::Rect crop(const cv::Size &size, const cv::Rect2f &box) {
    if (size.width <= 0 || size.height <= 0 || !std::isfinite(box.x) || !std::isfinite(box.y) ||
        !std::isfinite(box.width) || !std::isfinite(box.height) || box.width <= 0 || box.height <= 0)
        return {};
    int x1 = int(std::clamp(double(box.x), 0., double(size.width)));
    int y1 = int(std::clamp(double(box.y), 0., double(size.height)));
    int x2 = int(std::clamp(double(box.x) + box.width, 0., double(size.width)));
    int y2 = int(std::clamp(double(box.y) + box.height, 0., double(size.height)));
    return {x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1)};
}

cv::Mat make_input(const cv::Mat &bgr, const std::string &model) {
    const int side = input_size(model);
    if (bgr.empty() || bgr.type() != CV_8UC3)
        throw std::runtime_error("expected nonempty BGR face crop");
    cv::Mat resized;
    cv::resize(bgr, resized, {side, side}, 0, 0, cv::INTER_LINEAR);
    const bool mbf = model == "mbf_va_mtl";
    const std::array<double, 3> mean = mbf ? std::array<double, 3>{.5, .5, .5} :
                                            std::array<double, 3>{.485, .456, .406};
    const std::array<double, 3> stddev = mbf ? std::array<double, 3>{.5, .5, .5} :
                                              std::array<double, 3>{.229, .224, .225};
    int dims[] = {1, 3, side, side};
    cv::Mat blob(4, dims, CV_32F);
    for (int y = 0; y < side; ++y) {
        const auto *row = resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < side; ++x)
            for (int c = 0; c < 3; ++c)
                blob.ptr<float>()[c * side * side + y * side + x] =
                    float((row[x][2 - c] / 255. - mean[c]) / stddev[c]);
    }
    return blob;
}

Result scores(const std::vector<float> &logits, const std::string &model) {
    const bool mbf = input_size(model) == 112;
    if (logits.size() != (mbf ? 10U : 8U))
        throw std::runtime_error("emotion output shape mismatch");
    for (float v : logits)
        if (!std::isfinite(v)) throw std::runtime_error("non-finite emotion output");
    Result out;
    out.logits = logits;
    const auto best = std::max_element(logits.begin(), logits.begin() + kClasses);
    out.index = int(best - logits.begin());
    double total = 0;
    for (int i = 0; i < kClasses; ++i) total += std::exp(double(logits[i]) - *best);
    for (int i = 0; i < kClasses; ++i)
        out.probabilities[i] = float(std::exp(double(logits[i]) - *best) / total);
    return out;
}

Pipeline::Pipeline(const Config &config, int gpu_id)
    : model_(config.model), engine_(config.engine, config.model, gpu_id) {}

Result Pipeline::predict(const cv::Mat &bgr) {
    return scores(engine_.run(make_input(bgr, model_)), model_);
}

Result Pipeline::analyze(const cv::Mat &bgr, const cv::Rect2f &box) {
    const auto roi = crop(bgr.size(), box);
    return roi.empty() ? Result{} : predict(bgr(roi));
}

void Pipeline::warmup() {
    for (int i = 0; i < 3; ++i)
        predict(cv::Mat::zeros(input_size(model_), input_size(model_), CV_8UC3));
}

} // namespace emotieff
