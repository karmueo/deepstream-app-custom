#include "deepstream_common.h"
#include "deepstream_config_yaml.h"
#include <exception>
#include <stdexcept>
#include <string>

extern "C" gboolean parse_face_detect_yaml(NvDsFaceDetectConfig *config, gchar *path) {
    try {
        const auto node = YAML::LoadFile(path)["face-detect"];
        if (!node || !node.IsMap()) return FALSE;
        config->enable = FALSE;
        config->gpu_id = 0;
        config->unique_id = 16;
        for (const auto &entry : node) {
            const std::string key = entry.first.as<std::string>();
            if (key == "enable") config->enable = entry.second.as<gboolean>();
            else if (key == "gpu-id") config->gpu_id = entry.second.as<guint>();
            else if (key == "unique-id") config->unique_id = entry.second.as<guint>();
            else if (key == "config-file") {
                const auto value = entry.second.as<std::string>();
                if (value.empty()) throw std::runtime_error("face-detect config-file is empty");
                gchar *absolute_file = g_canonicalize_filename(path, NULL);
                gchar *directory = g_path_get_dirname(absolute_file);
                config->config_file = g_canonicalize_filename(value.c_str(), directory);
                g_free(directory);
                g_free(absolute_file);
            } else throw std::runtime_error("unknown face-detect key: " + key);
        }
        if (config->enable && (!config->config_file || config->unique_id == 0)) return FALSE;
        return TRUE;
    } catch (const std::exception &e) {
        g_printerr("Invalid face-detect configuration: %s\n", e.what());
        return FALSE;
    }
}
