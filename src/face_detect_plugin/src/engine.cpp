#include "face_core.hpp"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <filesystem>
#include <glib.h>
#include <json-glib/json-glib.h>
#include <stdexcept>

namespace face {
namespace {
void check(cudaError_t result) {
    if (result != cudaSuccess) throw std::runtime_error(cudaGetErrorString(result));
}
size_t elements(nvinfer1::Dims dims) {
    size_t size = 1;
    for (int i = 0; i < dims.nbDims; ++i) {
        if (dims.d[i] <= 0) throw std::runtime_error("unresolved engine dimension");
        size *= size_t(dims.d[i]);
    }
    return size;
}
class Logger : public nvinfer1::ILogger {
    void log(Severity severity, const char *message) noexcept override {
        if (severity <= Severity::kWARNING) g_warning("TensorRT: %s", message);
    }
};
struct Buffer {
    std::string name;
    nvinfer1::DataType type;
    size_t count = 0, bytes = 0;
    bool input = false;
    void *device = nullptr, *host = nullptr;
    ~Buffer() { if (device) cudaFree(device); if (host) cudaFreeHost(host); }
};
std::string required_string(JsonObject *obj, const char *key) {
    if (!json_object_has_member(obj, key) ||
        json_node_get_value_type(json_object_get_member(obj, key)) != G_TYPE_STRING)
        throw std::runtime_error(std::string("missing engine metadata: ") + key);
    return json_object_get_string_member(obj, key);
}
}
std::string sha256_file(const std::string &path) {
    gchar *data = nullptr;
    gsize size = 0;
    GError *error = nullptr;
    if (!g_file_get_contents(path.c_str(), &data, &size, &error)) {
        std::string message = error ? error->message : "file read failed";
        g_clear_error(&error);
        throw std::runtime_error(message);
    }
    gchar *digest = g_compute_checksum_for_data(G_CHECKSUM_SHA256,
        reinterpret_cast<guchar *>(data), size);
    std::string result = digest;
    g_free(digest); g_free(data);
    return result;
}
std::string runtime_identity(int gpu_id) {
    check(cudaSetDevice(gpu_id));
    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, gpu_id));
    return "TRT" + std::to_string(NV_TENSORRT_MAJOR) + "." +
        std::to_string(NV_TENSORRT_MINOR) + "." + std::to_string(NV_TENSORRT_PATCH) +
        "." + std::to_string(NV_TENSORRT_BUILD) + "|" + prop.name + "|SM" +
        std::to_string(prop.major) + std::to_string(prop.minor);
}
struct Engine::Impl {
    Logger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::vector<std::unique_ptr<Buffer>> buffers;
    std::vector<std::string> output_names;
    cudaStream_t stream = nullptr;
    std::string source_hash, role;
    float mean = 0, stddev = 0;
    ~Impl() {
        if (stream) { cudaStreamSynchronize(stream); cudaStreamDestroy(stream); }
    }
};
Engine::Engine(const std::string &path, int gpu_id) : p_(std::make_unique<Impl>()) {
    check(cudaSetDevice(gpu_id));
    JsonParser *parser = json_parser_new();
    GError *error = nullptr;
    if (!json_parser_load_from_file(parser, (path + ".json").c_str(), &error)) {
        std::string message = error ? error->message : "cannot load engine metadata";
        g_clear_error(&error); g_object_unref(parser);
        throw std::runtime_error(message);
    }
    try {
        JsonNode *root = json_parser_get_root(parser);
        if (!JSON_NODE_HOLDS_OBJECT(root)) throw std::runtime_error("invalid engine metadata");
        JsonObject *meta = json_node_get_object(root);
        if (json_object_get_int_member(meta, "schema") != 1 ||
            required_string(meta, "runtime") != runtime_identity(gpu_id) ||
            required_string(meta, "precision") != "fp16")
            throw std::runtime_error("engine runtime or precision mismatch; rebuild on this Jetson");
        if (sha256_file(path) != required_string(meta, "engine_sha256"))
            throw std::runtime_error("engine hash mismatch");
        const auto source = std::filesystem::path(path).parent_path() /
            required_string(meta, "onnx_path");
        p_->source_hash = required_string(meta, "onnx_sha256");
        if (std::filesystem::exists(source) && sha256_file(source) != p_->source_hash)
            throw std::runtime_error("ONNX model changed; rebuild engines");
        p_->role = required_string(meta, "role");
        p_->mean = json_object_get_double_member(meta, "mean");
        p_->stddev = json_object_get_double_member(meta, "std");
        JsonArray *outputs = json_object_get_array_member(meta, "output_names");
        for (guint i = 0; i < json_array_get_length(outputs); ++i)
            p_->output_names.emplace_back(json_array_get_string_element(outputs, i));
        JsonArray *shape_array = json_object_get_array_member(meta, "input_shape");
        if (!shape_array || json_array_get_length(shape_array) != 4)
            throw std::runtime_error("invalid input shape metadata");
        nvinfer1::Dims shape{}; shape.nbDims = 4;
        for (int i = 0; i < 4; ++i) shape.d[i] = json_array_get_int_element(shape_array, i);
        const auto input_name = required_string(meta, "input_name");
        gchar *bytes = nullptr; gsize length = 0;
        if (!g_file_get_contents(path.c_str(), &bytes, &length, &error)) {
            std::string message = error ? error->message : "cannot read engine";
            g_clear_error(&error); throw std::runtime_error(message);
        }
        p_->runtime.reset(nvinfer1::createInferRuntime(p_->logger));
        if (!p_->runtime) { g_free(bytes); throw std::runtime_error("cannot create TensorRT runtime"); }
        p_->engine.reset(p_->runtime->deserializeCudaEngine(bytes, length));
        g_free(bytes);
        if (!p_->engine) throw std::runtime_error("cannot deserialize engine");
        p_->context.reset(p_->engine->createExecutionContext());
        if (!p_->context || !p_->context->setInputShape(input_name.c_str(), shape))
            throw std::runtime_error("cannot set engine input shape");
        check(cudaStreamCreate(&p_->stream));
        for (int i = 0; i < p_->engine->getNbIOTensors(); ++i) {
            const char *name = p_->engine->getIOTensorName(i);
            auto buffer = std::make_unique<Buffer>();
            buffer->name = name;
            buffer->type = p_->engine->getTensorDataType(name);
            buffer->input = p_->engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            if (buffer->type != nvinfer1::DataType::kFLOAT &&
                buffer->type != nvinfer1::DataType::kHALF)
                throw std::runtime_error("unsupported engine I/O type");
            buffer->count = elements(p_->context->getTensorShape(name));
            buffer->bytes = buffer->count * (buffer->type == nvinfer1::DataType::kFLOAT ? 4 : 2);
            check(cudaMalloc(&buffer->device, buffer->bytes));
            check(cudaMallocHost(&buffer->host, buffer->bytes));
            if (!p_->context->setTensorAddress(name, buffer->device))
                throw std::runtime_error("cannot bind engine tensor");
            p_->buffers.push_back(std::move(buffer));
        }
        for (const auto &name : p_->output_names) {
            const auto found = std::find_if(p_->buffers.begin(), p_->buffers.end(),
                [&](const auto &b) { return !b->input && b->name == name; });
            if (found == p_->buffers.end()) throw std::runtime_error("missing engine output tensor");
        }
    } catch (...) { g_object_unref(parser); throw; }
    g_object_unref(parser);
}
Engine::~Engine() = default;
const std::string &Engine::source_hash() const { return p_->source_hash; }
const std::string &Engine::role() const { return p_->role; }
float Engine::mean() const { return p_->mean; }
float Engine::stddev() const { return p_->stddev; }
std::vector<std::vector<float>> Engine::run(const cv::Mat &blob) {
    for (const auto &b : p_->buffers) if (b->input) {
        if (blob.type() != CV_32F || !blob.isContinuous() || blob.total() != b->count)
            throw std::runtime_error("engine input shape/type mismatch");
        if (b->type == nvinfer1::DataType::kFLOAT)
            std::memcpy(b->host, blob.ptr<float>(), b->bytes);
        else for (size_t i = 0; i < b->count; ++i)
            static_cast<__half *>(b->host)[i] = __float2half(blob.ptr<float>()[i]);
        check(cudaMemcpyAsync(b->device, b->host, b->bytes, cudaMemcpyHostToDevice, p_->stream));
    }
    if (!p_->context->enqueueV3(p_->stream)) throw std::runtime_error("TensorRT enqueue failed");
    for (const auto &b : p_->buffers) if (!b->input)
        check(cudaMemcpyAsync(b->host, b->device, b->bytes, cudaMemcpyDeviceToHost, p_->stream));
    check(cudaStreamSynchronize(p_->stream));
    std::vector<std::vector<float>> result;
    for (const auto &name : p_->output_names) {
        const auto it = std::find_if(p_->buffers.begin(), p_->buffers.end(),
            [&](const auto &b) { return !b->input && b->name == name; });
        const auto &b = **it;
        std::vector<float> values(b.count);
        for (size_t i = 0; i < b.count; ++i) {
            values[i] = b.type == nvinfer1::DataType::kFLOAT ?
                static_cast<float *>(b.host)[i] : __half2float(static_cast<__half *>(b.host)[i]);
            if (!std::isfinite(values[i])) throw std::runtime_error("non-finite engine output");
        }
        result.push_back(std::move(values));
    }
    return result;
}
Pipeline::Pipeline(const Config &config, int gpu_id)
    : detector_(config.detector_engine, gpu_id), recognizer_(config.recognizer_engine, gpu_id),
      detection_threshold_(config.detection_threshold), nms_threshold_(config.nms_threshold) {
    if (detector_.role() != "scrfd" || recognizer_.role() != "arcface")
        throw std::runtime_error("wrong face model roles");
    mean_ = recognizer_.mean(); stddev_ = recognizer_.stddev();
    if (!(stddev_ > 0)) throw std::runtime_error("invalid recognizer normalization");
}
std::string Pipeline::model_id() const {
    char mean[32], stddev[32];
    g_ascii_dtostr(mean, sizeof(mean), mean_);
    g_ascii_dtostr(stddev, sizeof(stddev), stddev_);
    return recognizer_.source_hash() + ":arcface112-v1:mean" + mean + ":std" + stddev;
}
std::vector<Detection> Pipeline::analyze(const cv::Mat &image) {
    auto input = detector_input(image);
    auto faces = decode_scrfd(detector_.run(input.blob), input.scale,
                              detection_threshold_, nms_threshold_);
    for (auto &f : faces) {
        f.embedding = recognizer_.run(recognition_input(align_face(image, f), mean_, stddev_)).at(0);
        if (!normalize(f.embedding)) f.embedding.clear();
    }
    return faces;
}
void Pipeline::warmup() {
    detector_.run(detector_input(cv::Mat::zeros(640, 640, CV_8UC3)).blob);
    recognizer_.run(recognition_input(cv::Mat::zeros(112, 112, CV_8UC3), mean_, stddev_));
}
} // namespace face
