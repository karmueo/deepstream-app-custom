#include "face_core.hpp"
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>

static void require(bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
}
int main() {
    try {
        cv::Mat image(1080, 1920, CV_8UC3, cv::Scalar(10, 20, 30));
        const auto input = face::detector_input(image);
        require(input.blob.total() == 3 * 640 * 640, "detector blob size");
        require(std::abs(input.scale - 1.f / 3) < 1e-6, "detector scale");
        require(std::abs(input.blob.ptr<float>()[0] - (30 - 127.5) / 128) < 1e-6,
                "RGB normalization");
        std::vector<std::vector<float>> outputs(9);
        for (int level = 0; level < 3; ++level) {
            const int count = (640 / (8 << level)) * (640 / (8 << level)) * 2;
            outputs[level].resize(count);
            outputs[level + 3].resize(count * 4);
            outputs[level + 6].resize(count * 10);
        }
        const int index = 2 * (9 * 80 + 20);
        outputs[0][index] = .9f;
        for (int i = 0; i < 4; ++i) outputs[3][index * 4 + i] = 1;
        auto faces = face::decode_scrfd(outputs, input.scale, .5f, .4f);
        require(faces.size() == 1 && std::abs(faces[0].box.x - 456) < .001,
                "SCRFD coordinate restoration");
        auto vector = std::vector<float>{3, 4};
        require(face::normalize(vector) && std::abs(vector[0] - .6) < 1e-6, "normalize");
        require(!face::normalize(vector = {std::numeric_limits<float>::quiet_NaN()}),
                "reject NaN embedding");
        face::Sample sample{1, "测试", {.6f, .8f}};
        auto match = face::match({.6f, .8f}, {sample}, .5f);
        require(match.matched && match.person == 1 && match.name == "测试", "gallery match");
        match = face::match({.6f, .8f}, {sample}, 1.f);
        require(match.matched, "similarity boundary");
        const auto dbpath = std::filesystem::temp_directory_path() / "face-core-test.sqlite";
        std::filesystem::remove(dbpath);
        {
            face::Gallery db(dbpath.string(), false);
            const auto person = db.add_person("测试");
            require(db.add_sample(person, "a.jpg", "hash", "model-a", {.6f, .8f}), "insert");
            require(!db.add_sample(person, "a.jpg", "hash", "model-a", {.6f, .8f}), "dedup");
            require(db.samples("model-a").size() == 1, "read samples");
            require(db.samples("model-b").empty(), "model isolation");
        }
        {
            face::Gallery db(dbpath.string(), true);
            require(db.samples("model-a").size() == 1, "read-only gallery");
        }
        std::filesystem::remove(dbpath);
        std::cout << "face core tests passed\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
