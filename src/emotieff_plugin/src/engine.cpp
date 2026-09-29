#include "emotieff_core.hpp"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <glib.h>
#include <json-glib/json-glib.h>
#include <memory>
#include <stdexcept>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

namespace emotieff {
namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}

class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *message) noexcept override {
        if (severity <= Severity::kWARNING) g_warning("TensorRT: %s", message);
    }
};

std::string required_string(JsonObject *object, const char *key) {
    if (!json_object_has_member(object, key) ||
        json_node_get_value_type(json_object_get_member(object, key)) != G_TYPE_STRING)
        throw std::runtime_error(std::string("missing engine metadata: ") + key);
    return json_object_get_string_member(object, key);
}

JsonArray *required_array(JsonObject *object, const char *key, guint length) {
    if (!json_object_has_member(object, key))
        throw std::runtime_error(std::string("missing engine metadata: ") + key);
    auto *array = json_object_get_array_member(object, key);
    if (!array || json_array_get_length(array) != length)
        throw std::runtime_error(std::string("wrong engine metadata array: ") + key);
    return array;
}

size_t elements(nvinfer1::Dims dims) {
    size_t count = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) throw std::runtime_error("unresolved engine dimension");
        count *= size_t(dims.d[i]);
    }
    return count;
}

struct Buffer {
    std::string name;
    nvinfer1::DataType dtype = nvinfer1::DataType::kFLOAT;
    size_t count = 0;
    size_t bytes = 0;
    bool input = false;
    void *device = nullptr;
    void *host = nullptr;
    ~Buffer() {
        if (device) cudaFree(device);
        if (host) cudaFreeHost(host);
    }
};
} // namespace

std::string sha256_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("cannot read model file: " + path);
    GChecksum *checksum = g_checksum_new(G_CHECKSUM_SHA256);
    char chunk[1 << 16];
    while (file.read(chunk, sizeof(chunk)) || file.gcount())
        g_checksum_update(checksum, reinterpret_cast<const guchar *>(chunk), file.gcount());
    const std::string hash = g_checksum_get_string(checksum);
    g_checksum_free(checksum);
    if (file.bad()) throw std::runtime_error("model read failed: " + path);
    return hash;
}

std::string runtime_identity(int gpu_id) {
    check(cudaSetDevice(gpu_id));
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, gpu_id));
    return "TRT" + std::to_string(NV_TENSORRT_MAJOR) + "." +
        std::to_string(NV_TENSORRT_MINOR) + "." + std::to_string(NV_TENSORRT_PATCH) +
        "." + std::to_string(NV_TENSORRT_BUILD) + "|" + properties.name + "|SM" +
        std::to_string(properties.major) + std::to_string(properties.minor);
}

struct Engine::Impl {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<std::unique_ptr<Buffer>> buffers;
    Buffer *input = nullptr;
    Buffer *output = nullptr;
    cudaStream_t stream = nullptr;
    ~Impl() {
        if (stream) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
};

Engine::Engine(const std::string &path, const std::string &model, int gpu_id)
    : p_(std::make_unique<Impl>()) {
    check(cudaSetDevice(gpu_id));
    auto parser = std::unique_ptr<JsonParser, decltype(&g_object_unref)>(json_parser_new(), &g_object_unref);
    GError *error = nullptr;
    if (!json_parser_load_from_file(parser.get(), (path + ".json").c_str(), &error)) {
        std::string message = error ? error->message : "cannot load engine metadata";
        g_clear_error(&error);
        throw std::runtime_error(message);
    }
    JsonNode *root = json_parser_get_root(parser.get());
    if (!root || !JSON_NODE_HOLDS_OBJECT(root)) throw std::runtime_error("invalid engine metadata");
    JsonObject *meta = json_node_get_object(root);
    if (!json_object_has_member(meta, "schema") || json_object_get_int_member(meta, "schema") != 1 ||
        required_string(meta, "runtime") != runtime_identity(gpu_id) ||
        required_string(meta, "precision") != "fp16" ||
        required_string(meta, "role") != "emotion" ||
        required_string(meta, "model") != model ||
        required_string(meta, "layout") != "RGB-NCHW" ||
        !json_object_has_member(meta, "fp16_verified") ||
        !json_object_get_boolean_member(meta, "fp16_verified"))
        throw std::runtime_error("emotion engine runtime/model mismatch; rebuild it");
    if (sha256_file(path) != required_string(meta, "engine_sha256"))
        throw std::runtime_error("emotion engine checksum mismatch");
    const auto source = std::filesystem::path(path).parent_path() / required_string(meta, "onnx_path");
    if (std::filesystem::exists(source) && sha256_file(source.string()) !=
        required_string(meta, "onnx_sha256"))
        throw std::runtime_error("emotion ONNX changed; rebuild engine");

    const int side = input_size(model);
    auto *shape = required_array(meta, "input_shape", 4);
    if (json_array_get_int_element(shape, 0) != 1 || json_array_get_int_element(shape, 1) != 3 ||
        json_array_get_int_element(shape, 2) != side || json_array_get_int_element(shape, 3) != side)
        throw std::runtime_error("emotion input shape mismatch");
    const bool mbf = side == 112;
    const double expected_mean[] = {mbf ? .5 : .485, mbf ? .5 : .456, mbf ? .5 : .406};
    const double expected_std[] = {mbf ? .5 : .229, mbf ? .5 : .224, mbf ? .5 : .225};
    auto *mean = required_array(meta, "mean", 3);
    auto *stddev = required_array(meta, "std", 3);
    auto *labels = required_array(meta, "labels", kClasses);
    for (int i = 0; i < 3; ++i)
        if (std::abs(json_array_get_double_element(mean, i) - expected_mean[i]) > 1e-8 ||
            std::abs(json_array_get_double_element(stddev, i) - expected_std[i]) > 1e-8)
            throw std::runtime_error("emotion preprocessing metadata mismatch");
    for (int i = 0; i < kClasses; ++i)
        if (std::string(json_array_get_string_element(labels, i)) != label(i, false))
            throw std::runtime_error("emotion label metadata mismatch");
    const std::string input_name = required_string(meta, "input_name");
    auto *output_names = required_array(meta, "output_names", 1);
    const std::string output_name = json_array_get_string_element(output_names, 0);

    gchar *bytes = nullptr;
    gsize length = 0;
    if (!g_file_get_contents(path.c_str(), &bytes, &length, &error)) {
        std::string message = error ? error->message : "cannot read emotion engine";
        g_clear_error(&error);
        throw std::runtime_error(message);
    }
    p_->runtime.reset(nvinfer1::createInferRuntime(p_->logger));
    if (!p_->runtime) { g_free(bytes); throw std::runtime_error("cannot create TensorRT runtime"); }
    p_->engine.reset(p_->runtime->deserializeCudaEngine(bytes, length));
    g_free(bytes);
    if (!p_->engine) throw std::runtime_error("cannot deserialize emotion engine");
    p_->context.reset(p_->engine->createExecutionContext());
    if (!p_->context || !p_->context->setInputShape(input_name.c_str(), nvinfer1::Dims4(1, 3, side, side)))
        throw std::runtime_error("cannot set emotion input shape");
    if (p_->engine->getNbIOTensors() != 2)
        throw std::runtime_error("expected one emotion input and output");
    check(cudaStreamCreate(&p_->stream));
    for (int i = 0; i < p_->engine->getNbIOTensors(); ++i) {
        const char *name = p_->engine->getIOTensorName(i);
        auto buffer = std::make_unique<Buffer>();
        buffer->name = name;
        buffer->dtype = p_->engine->getTensorDataType(name);
        buffer->input = p_->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
        if (buffer->dtype != nvinfer1::DataType::kFLOAT && buffer->dtype != nvinfer1::DataType::kHALF)
            throw std::runtime_error("unsupported emotion engine tensor type");
        buffer->count = elements(p_->context->getTensorShape(name));
        buffer->bytes = buffer->count * (buffer->dtype == nvinfer1::DataType::kFLOAT ? 4 : 2);
        check(cudaMalloc(&buffer->device, buffer->bytes));
        check(cudaMallocHost(&buffer->host, buffer->bytes));
        if (!p_->context->setTensorAddress(name, buffer->device))
            throw std::runtime_error("cannot bind emotion tensor");
        if (buffer->input && buffer->name == input_name) p_->input = buffer.get();
        if (!buffer->input && buffer->name == output_name) p_->output = buffer.get();
        p_->buffers.push_back(std::move(buffer));
    }
    if (!p_->input || !p_->output || p_->input->count != size_t(3 * side * side) ||
        p_->output->count != size_t(mbf ? 10 : 8))
        throw std::runtime_error("emotion engine I/O mismatch");
}

Engine::~Engine() = default;

std::vector<float> Engine::run(const cv::Mat &blob) {
    if (blob.type() != CV_32F || !blob.isContinuous() || blob.total() != p_->input->count)
        throw std::runtime_error("emotion input type/shape mismatch");
    auto *input = p_->input;
    if (input->dtype == nvinfer1::DataType::kFLOAT)
        std::memcpy(input->host, blob.ptr<float>(), input->bytes);
    else
        for (size_t i = 0; i < input->count; ++i)
            static_cast<__half *>(input->host)[i] = __float2half(blob.ptr<float>()[i]);
    check(cudaMemcpyAsync(input->device, input->host, input->bytes, cudaMemcpyHostToDevice, p_->stream));
    if (!p_->context->enqueueV3(p_->stream)) throw std::runtime_error("emotion inference failed");
    auto *output = p_->output;
    check(cudaMemcpyAsync(output->host, output->device, output->bytes, cudaMemcpyDeviceToHost, p_->stream));
    check(cudaStreamSynchronize(p_->stream));
    std::vector<float> values(output->count);
    for (size_t i = 0; i < output->count; ++i)
        values[i] = output->dtype == nvinfer1::DataType::kFLOAT ?
            static_cast<float *>(output->host)[i] : __half2float(static_cast<__half *>(output->host)[i]);
    return values;
}

} // namespace emotieff
