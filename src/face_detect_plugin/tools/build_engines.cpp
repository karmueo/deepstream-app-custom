#include "face_core.hpp"
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
        std::string message = error ? error->message : "write failed";
        g_clear_error(&error); throw std::runtime_error(message);
    }
}
JsonArray *numbers(std::initializer_list<int> values) {
    JsonArray *array = json_array_new();
    for (int value : values) json_array_add_int_element(array, value);
    return array;
}
void build(const std::string &onnx, const std::string &dest, bool detection, int gpu_id) {
    Logger logger;
    std::cout << "Building " << onnx << " ..." << std::endl;
    std::unique_ptr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
    if (!builder) throw std::runtime_error("cannot create TensorRT builder");
    std::unique_ptr<nvinfer1::INetworkDefinition> net(builder->createNetworkV2(0));
    std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*net, logger));
    if (!parser || !parser->parseFromFile(onnx.c_str(), int(Logger::Severity::kWARNING)))
        throw std::runtime_error("ONNX parse failed");
    if (net->getNbInputs() != 1 || net->getNbOutputs() != (detection ? 9 : 1))
        throw std::runtime_error("expected buffalo_sc model with 9/1 outputs");
    auto *input = net->getInput(0);
    const int size = detection ? 640 : 112;
    input->setDimensions(nvinfer1::Dims4(1, 3, size, size));
    JsonArray *output_names = json_array_new();
    for (int i = 0; i < net->getNbOutputs(); ++i)
        json_array_add_string_element(output_names, net->getOutput(i)->getName());
    std::unique_ptr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 512ULL << 20);
    config->setProfilingVerbosity(nvinfer1::ProfilingVerbosity::kDETAILED);
    config->setMaxAuxStreams(0);
    std::unique_ptr<nvinfer1::IHostMemory> plan(builder->buildSerializedNetwork(*net, *config));
    if (!plan) throw std::runtime_error("FP16 engine build failed");
    std::unique_ptr<nvinfer1::IRuntime> runtime(nvinfer1::createInferRuntime(logger));
    std::unique_ptr<nvinfer1::ICudaEngine> engine(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine) throw std::runtime_error("built engine cannot deserialize");
    std::unique_ptr<nvinfer1::IEngineInspector> inspector(engine->createEngineInspector());
    const std::string layers = inspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON);
    if (layers.find("Half") == std::string::npos && layers.find("FP16") == std::string::npos)
        throw std::runtime_error("inspector did not confirm FP16 layers");
    write_file(dest, plan->data(), plan->size());
    write_file(dest + ".layers.json", layers.data(), layers.size());
    JsonObject *meta = json_object_new();
    json_object_set_int_member(meta, "schema", 1);
    json_object_set_string_member(meta, "runtime", face::runtime_identity(gpu_id).c_str());
    json_object_set_string_member(meta, "precision", "fp16");
    json_object_set_string_member(meta, "role", detection ? "scrfd" : "arcface");
    const auto relative_onnx = std::filesystem::absolute(onnx).lexically_relative(
        std::filesystem::absolute(dest).parent_path());
    json_object_set_string_member(meta, "onnx_path", relative_onnx.string().c_str());
    json_object_set_string_member(meta, "onnx_sha256", face::sha256_file(onnx).c_str());
    json_object_set_string_member(meta, "engine_sha256", face::sha256_file(dest).c_str());
    json_object_set_string_member(meta, "input_name", input->getName());
    json_object_set_array_member(meta, "input_shape", numbers({1, 3, size, size}));
    json_object_set_array_member(meta, "output_names", output_names);
    json_object_set_double_member(meta, "mean", 127.5);
    json_object_set_double_member(meta, "std", detection ? 128. : 127.5);
    json_object_set_string_member(meta, "layout", "RGB-NCHW");
    json_object_set_int_member(meta, "workspace_mib", 512);
    json_object_set_boolean_member(meta, "fp16_verified", TRUE);
    JsonArray *tensors = json_array_new();
    for (int i = 0; i < engine->getNbIOTensors(); ++i) {
        const char *name = engine->getIOTensorName(i);
        const auto dims = engine->getTensorShape(name);
        JsonArray *shape = json_array_new();
        for (int j = 0; j < dims.nbDims; ++j) json_array_add_int_element(shape, dims.d[j]);
        JsonObject *item = json_object_new();
        json_object_set_string_member(item, "name", name);
        json_object_set_array_member(item, "shape", shape);
        json_object_set_int_member(item, "dtype", int(engine->getTensorDataType(name)));
        json_array_add_object_element(tensors, item);
    }
    json_object_set_array_member(meta, "tensors", tensors);
    JsonNode *root = json_node_new(JSON_NODE_OBJECT);
    json_node_take_object(root, meta);
    JsonGenerator *generator = json_generator_new();
    json_generator_set_root(generator, root);
    gsize length = 0;
    gchar *serialized = json_generator_to_data(generator, &length);
    write_file(dest + ".json", serialized, length);
    g_free(serialized); g_object_unref(generator); json_node_free(root);
    std::cout << "Saved " << dest << std::endl;
}
}
int main(int argc, char **argv) {
    try {
        std::string models, output;
        int gpu_id = 0;
        for (int i = 1; i < argc; ++i) {
            const std::string arg = argv[i];
            if ((arg == "--models" || arg == "--output" || arg == "--gpu-id") && i + 1 < argc) {
                const std::string value = argv[++i];
                if (arg == "--models") models = value;
                else if (arg == "--output") output = value;
                else gpu_id = std::stoi(value);
            } else throw std::runtime_error("usage: face-build-engines --models DIR --output DIR [--gpu-id N]");
        }
        if (models.empty() || output.empty()) throw std::runtime_error("models and output are required");
        std::filesystem::create_directories(output);
        build((std::filesystem::path(models) / "det_500m.onnx").string(),
              (std::filesystem::path(output) / "detector.engine").string(), true, gpu_id);
        build((std::filesystem::path(models) / "w600k_mbf.onnx").string(),
              (std::filesystem::path(output) / "recognizer.engine").string(), false, gpu_id);
        face::Config cfg;
        cfg.detector_engine = (std::filesystem::path(output) / "detector.engine").string();
        cfg.recognizer_engine = (std::filesystem::path(output) / "recognizer.engine").string();
        face::Pipeline verify(cfg, gpu_id); verify.warmup();
        std::cout << "Both engines passed warmup." << std::endl;
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
