#include "emotieff_core.hpp"
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>

int main() {
    cv::Mat bgr(2, 3, CV_8UC3, cv::Scalar(0, 128, 255));
    for (const auto &model : {"enet_b0_8_best_vgaf", "mbf_va_mtl"}) {
        auto blob = emotieff::make_input(bgr, model);
        const int side = emotieff::input_size(model);
        assert(blob.size[2] == side && blob.size[3] == side);
        const float *values = blob.ptr<float>();
        if (side == 112) {
            assert(std::abs(values[0] - 1.f) < 1e-6);
            assert(std::abs(values[side * side] - float(128. / 127.5 - 1)) < 1e-6);
            assert(std::abs(values[2 * side * side] + 1.f) < 1e-6);
        } else {
            assert(std::abs(values[0] - float((1. - .485) / .229)) < 1e-6);
            assert(std::abs(values[side * side] - float((128. / 255. - .456) / .224)) < 1e-6);
            assert(std::abs(values[2 * side * side] - float(-.406 / .225)) < 1e-6);
        }
    }
    assert(emotieff::crop({100, 80}, {-10, -20, 40, 50}) == cv::Rect(0, 0, 30, 30));
    assert(emotieff::crop({100, 80}, {90, 70, 30, 50}) == cv::Rect(90, 70, 10, 10));
    assert(emotieff::crop({100, 80}, {100, 0, 10, 10}).empty());
    assert(emotieff::crop({100, 80}, {NAN, 0, 10, 10}).empty());
    std::vector<float> logits{1000, 999, 998, 997, 996, 995, 994, 993};
    auto enet = emotieff::scores(logits, "enet_b0_8_best_vgaf");
    assert(enet.index == 0 && std::string(emotieff::label(0)) == "愤怒");
    double total = 0;
    for (float value : enet.probabilities) total += value;
    assert(std::abs(total - 1) < 1e-6);
    logits.insert(logits.end(), {10000, 20000});
    auto mbf = emotieff::scores(logits, "mbf_va_mtl");
    assert(mbf.probabilities == enet.probabilities && mbf.logits.size() == 10);
    bool failed = false;
    try { emotieff::scores(logits, "enet_b0_8_best_vgaf"); }
    catch (const std::runtime_error &) { failed = true; }
    assert(failed);
    logits[0] = std::numeric_limits<float>::infinity();
    failed = false;
    try { emotieff::scores(logits, "mbf_va_mtl"); }
    catch (const std::runtime_error &) { failed = true; }
    assert(failed);

    const auto config_path = std::filesystem::temp_directory_path() / "emotieff-config-test.yml";
    { std::ofstream file(config_path); file << "property:\n  emotion-model: mbf_va_mtl\n"
                                            "  emotion-engine: ../model/test.engine\n  interval: 2\n"; }
    auto config = emotieff::load_config(config_path.string());
    assert(config.model == "mbf_va_mtl" && config.interval == 2);
    assert(config.engine == (config_path.parent_path() / "../model/test.engine").lexically_normal());
    std::filesystem::remove(config_path);
}
