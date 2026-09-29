#include "face_core.hpp"
#include <cmath>
#include <filesystem>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace face {
Config load_config(const std::string &path) {
    const auto root = YAML::LoadFile(path);
    const auto node = root["property"];
    if (!node || !node.IsMap()) throw std::runtime_error("missing property map");
    Config out;
    const auto base = std::filesystem::absolute(path).parent_path();
    auto read_path = [&](const char *key, bool required) {
        if (!node[key] || !node[key].IsScalar()) {
            if (required) throw std::runtime_error(std::string("missing ") + key);
            return std::string();
        }
        const auto value = node[key].as<std::string>();
        if (value.empty()) {
            if (required) throw std::runtime_error(std::string("empty ") + key);
            return value;
        }
        return (base / value).lexically_normal().string();
    };
    out.detector_engine = read_path("detector-engine", true);
    out.recognizer_engine = read_path("recognizer-engine", true);
    out.gallery_file = read_path("gallery-file", false);
    if (node["interval"]) out.interval = node["interval"].as<unsigned>();
    if (node["detection-threshold"]) out.detection_threshold = node["detection-threshold"].as<float>();
    if (node["nms-threshold"]) out.nms_threshold = node["nms-threshold"].as<float>();
    if (node["recognition-threshold"]) out.recognition_threshold = node["recognition-threshold"].as<float>();
    if (out.interval > 10000 || !std::isfinite(out.detection_threshold) ||
        out.detection_threshold < 0 || out.detection_threshold > 1 ||
        !std::isfinite(out.nms_threshold) || out.nms_threshold < 0 || out.nms_threshold > 1 ||
        !std::isfinite(out.recognition_threshold) || out.recognition_threshold < -1 ||
        out.recognition_threshold > 1)
        throw std::runtime_error("invalid face detection configuration");
    for (const auto &entry : node) {
        const std::string key = entry.first.as<std::string>();
        if (key != "detector-engine" && key != "recognizer-engine" && key != "gallery-file" &&
            key != "interval" && key != "detection-threshold" && key != "nms-threshold" &&
            key != "recognition-threshold")
            throw std::runtime_error("unknown face property: " + key);
    }
    return out;
}
} // namespace face
