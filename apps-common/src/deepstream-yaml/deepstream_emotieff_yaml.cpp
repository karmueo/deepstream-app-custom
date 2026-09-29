#include "deepstream_common.h"
#include "deepstream_config_yaml.h"
#include <exception>
#include <stdexcept>
#include <string>

extern "C" gboolean parse_emotieff_yaml(NvDsEmotieffConfig *config, gchar *path) {
    try {
        const auto node = YAML::LoadFile(path)["emotieff"];
        if (!node || !node.IsMap()) return FALSE;
        config->enable = FALSE;
        config->gpu_id = 0;
        config->unique_id = 17;
        for (const auto &entry : node) {
            const std::string key = entry.first.as<std::string>();
            if (key == "enable") config->enable = entry.second.as<gboolean>();
            else if (key == "gpu-id") config->gpu_id = entry.second.as<guint>();
            else if (key == "unique-id") config->unique_id = entry.second.as<guint>();
            else if (key == "config-file") {
                const auto value = entry.second.as<std::string>();
                if (value.empty()) throw std::runtime_error("emotieff config-file is empty");
                gchar *absolute = g_canonicalize_filename(path, NULL);
                gchar *directory = g_path_get_dirname(absolute);
                g_free(config->config_file);
                config->config_file = g_canonicalize_filename(value.c_str(), directory);
                g_free(directory);
                g_free(absolute);
            } else throw std::runtime_error("unknown emotieff key: " + key);
        }
        return !config->enable || (config->config_file && config->unique_id != 0);
    } catch (const std::exception &error) {
        g_printerr("Invalid emotieff configuration: %s\n", error.what());
        return FALSE;
    }
}
