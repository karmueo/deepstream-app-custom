#include "emotieff_core.hpp"
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <filesystem>
#include <glib.h>
#include <json-glib/json-glib.h>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
    }
};

void write_file(const std::string &path, const void *data, size_t size) {
    GError *error = nullptr;
    if (!g_file_set_contents(path.c_str(), static_cast<const char *>(data), size, &error)) {
        const std::string message = error ? error->message : "file write failed";
        g_clear_error(&error);
        throw std::runtime_error(message);
    }
}

JsonArray *integers(std::initializer_list<int> values) {
    JsonArray *array = json_array_new();
    for (int value : values) json_array_add_int_element(array, value);
    return array;
}

JsonArray *doubles(std::initializer_list<double> values) {
    JsonArray *array = json_array_new();
    for (double value : values) json_array_add_double_element(array, value);
    return array;
}

void build(const std::string &onnx, const std::string &model, const std::string &dest, int gpu_id) {
    const int side = emotieff::input_size(model);
    const bool mbf = side == 112;
    Logger logger;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("cannot create TensorRT builder");
    std::unique_ptr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
    if (!parser || !parser->parseFromFile(onnx.c_str(), int(nvinfer1::ILogger::Severity::kWARNING)))
        throw std::runtime_error("emotion ONNX parse failed");
    if (network->getNbInputs() != 1 || network->getNbOutputs() != 1)
        throw std::runtime_error("expected one emotion input and output");
    auto *input = network->getInput(0);
    auto output_shape = network->getOutput(0)->getDimensions();
    auto input_shape = input->getDimensions();
    if (input_shape.nbDims != 4 || input_shape.d[1] != 3 || input_shape.d[2] != side ||
        input_shape.d[3] != side || output_shape.nbDims != 2 ||
        output_shape.d[1] != (mbf ? 10 : 8))
        throw std::runtime_error("emotion ONNX shape does not match model");
    input->setDimensions(nvinfer1::Dims4(1, 3, side, side));

    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512ULL << 20);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    config->setMaxAuxStreams(0);
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*network, *config));
    if (!plan) throw std::runtime_error("emotion engine build failed");
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine) throw std::runtime_error("built emotion engine cannot deserialize");
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(engine->createEngineInspector());
    const std::string layers = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
    if (layers.find("Half") == std::string::npos && layers.find("FP16") == std::string::npos)
        throw std::runtime_error("engine layer report has no FP16 implementation");
    write_file(dest, plan->data(), plan->size());
    write_file(dest + ".layers.json", layers.data(), layers.size());

    auto *meta = json_object_new();
    json_object_set_int_member(meta, "schema", 1);
    json_object_set_string_member(meta, "runtime", emotieff::runtime_identity(gpu_id).c_str());
    json_object_set_string_member(meta, "precision", "fp16");
    json_object_set_string_member(meta, "role", "emotion");
    json_object_set_string_member(meta, "model", model.c_str());
    const auto relative = std::filesystem::relative(std::filesystem::absolute(onnx),
                                                     std::filesystem::absolute(dest).parent_path());
    json_object_set_string_member(meta, "onnx_path", relative.string().c_str());
    json_object_set_string_member(meta, "onnx_sha256", emotieff::sha256_file(onnx).c_str());
    json_object_set_string_member(meta, "engine_sha256", emotieff::sha256_file(dest).c_str());
    json_object_set_string_member(meta, "input_name", input->getName());
    json_object_set_array_member(meta, "input_shape", integers({1, 3, side, side}));
    auto *output_names = json_array_new();
    json_array_add_string_element(output_names, network->getOutput(0)->getName());
    json_object_set_array_member(meta, "output_names", output_names);
    json_object_set_array_member(meta, "mean", mbf ? doubles({.5, .5, .5}) :
                                                       doubles({.485, .456, .406}));
    json_object_set_array_member(meta, "std", mbf ? doubles({.5, .5, .5}) :
                                                      doubles({.229, .224, .225}));
    json_object_set_string_member(meta, "layout", "RGB-NCHW");
    auto *labels = json_array_new();
    for (int i = 0; i < emotieff::kClasses; ++i)
        json_array_add_string_element(labels, emotieff::label(i, false));
    json_object_set_array_member(meta, "labels", labels);
    json_object_set_int_member(meta, "workspace_mib", 512);
    json_object_set_boolean_member(meta, "fp16_verified", TRUE);
    auto *tensors = json_array_new();
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char *name = engine->getIOTensorName(i);
        const auto dims = engine->getTensorShape(name);
        auto *shape = json_array_new();
        for (int j = 0; j < dims.nbDims; ++j) json_array_add_int_element(shape, dims.d[j]);
        auto *tensor = json_object_new();
        json_object_set_string_member(tensor, "name", name);
        json_object_set_array_member(tensor, "shape", shape);
        json_object_set_int_member(tensor, "dtype", int(engine->getTensorDataType(name)));
        json_array_add_object_element(tensors, tensor);
    }
    json_object_set_array_member(meta, "tensors", tensors);
    auto *root = json_node_new(JSON_NODE_OBJECT);
    json_node_take_object(root, meta);
    auto *generator = json_generator_new();
    json_generator_set_root(generator, root);
    json_generator_set_pretty(generator, TRUE);
    gsize length = 0;
    gchar *json = json_generator_to_data(generator, &length);
    write_file(dest + ".json", json, length);
    g_free(json);
    g_object_unref(generator);
    json_node_free(root);

    emotieff::Pipeline verify({model, dest, 0}, gpu_id);
    verify.warmup();
    std::cout << "Built and warmed " << dest << '\n';
}
} // namespace

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: emotieff-build-engine MODEL ONNX OUTPUT [GPU_ID]\n";
        return 2;
    }
    try {
        const int gpu_id = argc == 5 ? std::stoi(argv[4]) : 0;
        std::filesystem::create_directories(std::filesystem::absolute(argv[3]).parent_path());
        build(argv[2], argv[1], argv[3], gpu_id);
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
