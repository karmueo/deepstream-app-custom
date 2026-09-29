#include "emotieff_core.hpp"
#include <filesystem>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace emotieff {
Config load_config(const std::string &path) {
    const auto root = YAML::LoadFile(path);
    const auto node = root["property"];
    if (!node || !node.IsMap()) throw std::runtime_error("missing property map");
    Config config;
    for (const auto &entry : node) {
        const auto key = entry.first.as<std::string>();
        if (key == "emotion-model") config.model = entry.second.as<std::string>();
        else if (key == "emotion-engine") config.engine = entry.second.as<std::string>();
        else if (key == "interval") config.interval = entry.second.as<unsigned>();
        else throw std::runtime_error("unknown emotion property: " + key);
    }
    input_size(config.model);
    if (config.interval > 10000) throw std::runtime_error("emotion interval out of range");
    if (config.engine.empty()) throw std::runtime_error("emotion-engine is required");
    config.engine = (std::filesystem::absolute(path).parent_path() / config.engine)
                        .lexically_normal().string();
    return config;
}
} // namespace emotieff
