#include "face_core.hpp"
#include <iomanip>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

int main(int argc, char **argv) {
    try {
        if (argc != 4 || std::string(argv[1]) != "--config")
            throw std::runtime_error("usage: face-image-check --config FILE IMAGE");
        auto config = face::load_config(argv[2]);
        face::Pipeline pipeline(config, 0);
        auto image = cv::imread(argv[3]);
        if (image.empty()) throw std::runtime_error("cannot read image");
        std::cout << std::setprecision(9) << "{\"model_id\":\"" << pipeline.model_id()
                  << "\",\"faces\":[";
        bool first = true;
        for (const auto &f : pipeline.analyze(image)) {
            if (!first) std::cout << ',';
            first = false;
            std::cout << "{\"bbox\":[" << f.box.x << ',' << f.box.y << ','
                      << f.box.width << ',' << f.box.height << "],\"score\":" << f.score
                      << ",\"landmarks\":[";
            for (int i = 0; i < 5; ++i) {
                if (i) std::cout << ',';
                std::cout << '[' << f.points[i].x << ',' << f.points[i].y << ']';
            }
            std::cout << "],\"embedding\":[";
            for (size_t i = 0; i < f.embedding.size(); ++i) {
                if (i) std::cout << ',';
                std::cout << f.embedding[i];
            }
            std::cout << "]}";
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
