#include "emotieff_core.hpp"
#include <iostream>
#include <json-glib/json-glib.h>
#include <opencv2/imgcodecs.hpp>
#include <stdexcept>

int main(int argc, char **argv) {
    if (argc != 4 || std::string(argv[1]) != "--config") {
        std::cerr << "Usage: emotieff-image-check --config CONFIG FACE_CROP_IMAGE\n";
        return 2;
    }
    try {
        const auto config = emotieff::load_config(argv[2]);
        const auto image = cv::imread(argv[3]);
        if (image.empty()) throw std::runtime_error("cannot read face crop");
        emotieff::Pipeline pipeline(config, 0);
        const auto result = pipeline.predict(image);
        auto *object = json_object_new();
        json_object_set_string_member(object, "model", config.model.c_str());
        json_object_set_int_member(object, "class_id", result.index);
        json_object_set_string_member(object, "emotion", emotieff::label(result.index, false));
        auto *probabilities = json_array_new();
        for (float value : result.probabilities) json_array_add_double_element(probabilities, value);
        json_object_set_array_member(object, "probabilities", probabilities);
        auto *logits = json_array_new();
        for (float value : result.logits) json_array_add_double_element(logits, value);
        json_object_set_array_member(object, "logits", logits);
        auto *root = json_node_new(JSON_NODE_OBJECT);
        json_node_take_object(root, object);
        auto *generator = json_generator_new();
        json_generator_set_root(generator, root);
        gchar *json = json_generator_to_data(generator, nullptr);
        std::cout << json << '\n';
        g_free(json);
        g_object_unref(generator);
        json_node_free(root);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
