#include "emotieff_core.hpp"
#include "emotieff_metadata.h"
#include "face_metadata.h"
#include <algorithm>
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gstnvdsmeta.h>
#include <map>
#include <memory>
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <nvdsmeta.h>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <vector>
#include <cuda_runtime_api.h>

GST_DEBUG_CATEGORY_STATIC(emotieff_debug);
#define GST_CAT_DEFAULT emotieff_debug

namespace {
struct SourceState { guint64 last_frame = 0, seen = 0; };
struct Runtime {
    emotieff::Config config;
    std::unique_ptr<emotieff::Pipeline> pipeline;
    std::map<guint, SourceState> sources;
    NvBufSurface *converted = nullptr;
    guint width = 0, height = 0;
    ~Runtime() { if (converted) NvBufSurfaceDestroy(converted); }
};

void ensure_surface(Runtime &runtime, guint width, guint height, guint gpu_id) {
    if (runtime.converted && runtime.width == width && runtime.height == height) return;
    if (runtime.converted) { NvBufSurfaceDestroy(runtime.converted); runtime.converted = nullptr; }
    NvBufSurfaceCreateParams params{};
    params.gpuId = gpu_id;
    params.width = width;
    params.height = height;
    params.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
    params.layout = NVBUF_LAYOUT_PITCH;
#ifdef __aarch64__
    params.memType = NVBUF_MEM_SURFACE_ARRAY;
#else
    params.memType = NVBUF_MEM_CUDA_UNIFIED;
#endif
    if (NvBufSurfaceCreate(&runtime.converted, 1, &params) != 0)
        throw std::runtime_error("cannot allocate emotion RGBA surface");
    runtime.width = width;
    runtime.height = height;
}

cv::Mat convert_frame(Runtime &runtime, NvBufSurface *surface, guint batch_id, guint gpu_id) {
    auto &src = surface->surfaceList[batch_id];
    if (!src.width || !src.height) throw std::runtime_error("invalid frame dimensions");
    ensure_surface(runtime, src.width, src.height, gpu_id);
    NvBufSurfTransformConfigParams session{};
    session.compute_mode = NvBufSurfTransformCompute_Default;
    session.gpu_id = gpu_id;
    if (NvBufSurfTransformSetSessionParams(&session) != NvBufSurfTransformError_Success)
        throw std::runtime_error("NvBufSurfTransform session failed");
    NvBufSurface single = *surface;
    single.surfaceList = &src;
    single.batchSize = single.numFilled = 1;
    NvBufSurfTransformParams transform{};
    transform.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    transform.transform_filter = NvBufSurfTransformInter_Default;
    if (NvBufSurfTransform(&single, runtime.converted, &transform) != NvBufSurfTransformError_Success)
        throw std::runtime_error("NVMM to RGBA emotion conversion failed");
    if (NvBufSurfaceMap(runtime.converted, 0, 0, NVBUF_MAP_READ) != 0)
        throw std::runtime_error("cannot map emotion conversion surface");
    try {
        if (runtime.converted->memType == NVBUF_MEM_SURFACE_ARRAY &&
            NvBufSurfaceSyncForCpu(runtime.converted, 0, 0) != 0)
            throw std::runtime_error("emotion conversion CPU sync failed");
        const auto &dst = runtime.converted->surfaceList[0];
        cv::Mat rgba(dst.height, dst.width, CV_8UC4, dst.mappedAddr.addr[0], dst.pitch);
        cv::Mat bgr;
        cv::cvtColor(rgba, bgr, cv::COLOR_RGBA2BGR);
        NvBufSurfaceUnMap(runtime.converted, 0, 0);
        return bgr;
    } catch (...) {
        NvBufSurfaceUnMap(runtime.converted, 0, 0);
        throw;
    }
}

bool is_face(NvDsObjectMeta *object, guint operate_on_gie_id) {
    if (!object || object->unique_component_id != gint(operate_on_gie_id)) return false;
    for (NvDsMetaList *node = object->obj_user_meta_list; node; node = node->next) {
        auto *meta = static_cast<NvDsUserMeta *>(node->data);
        if (meta && meta->base_meta.meta_type ==
                        nvds_get_user_meta_type(const_cast<gchar *>(FACE_DETECT_META_TYPE)))
            return true;
    }
    return false;
}

void attach_result(NvDsBatchMeta *batch, NvDsObjectMeta *object,
                   const emotieff::Result &result, const std::string &model, guint unique_id) {
    auto *user = nvds_acquire_user_meta_from_pool(batch);
    if (!user) throw std::runtime_error("emotion user metadata pool exhausted");
    auto *value = g_new0(EmotieffMeta, 1);
    value->component_id = unique_id;
    g_strlcpy(value->model, model.c_str(), sizeof(value->model));
    value->valid = result.index >= 0;
    value->class_id = result.index;
    value->logits_count = result.logits.size();
    for (size_t i = 0; i < result.logits.size(); ++i) value->logits[i] = result.logits[i];
    if (value->valid) {
        value->confidence = result.probabilities[result.index];
        for (int i = 0; i < emotieff::kClasses; ++i)
            value->probabilities[i] = result.probabilities[i];
    }
    user->user_meta_data = value;
    user->base_meta.meta_type = nvds_get_user_meta_type(const_cast<gchar *>(EMOTIEFF_META_TYPE));
    user->base_meta.copy_func = emotieff_meta_copy;
    user->base_meta.release_func = emotieff_meta_release;
    nvds_add_user_meta_to_obj(object, user);
    if (!value->valid) return;

    auto *classifier = nvds_acquire_classifier_meta_from_pool(batch);
    auto *label = nvds_acquire_label_info_meta_from_pool(batch);
    if (!classifier || !label) throw std::runtime_error("emotion classifier metadata pool exhausted");
    classifier->unique_component_id = unique_id;
    classifier->num_labels = 1;
    classifier->classifier_type = "emotion";
    label->num_classes = emotieff::kClasses;
    label->result_class_id = value->class_id;
    label->result_prob = value->confidence;
    g_strlcpy(label->result_label, emotieff::label(value->class_id), sizeof(label->result_label));
    nvds_add_label_info_meta_to_classifier(classifier, label);
    nvds_add_classifier_meta_to_object(object, classifier);
}
} // namespace

typedef struct _GstEmotieff {
    GstBaseTransform parent;
    gchar *config_file;
    guint gpu_id, unique_id, operate_on_gie_id;
    Runtime *runtime;
} GstEmotieff;
typedef struct _GstEmotieffClass { GstBaseTransformClass parent_class; } GstEmotieffClass;

G_DEFINE_TYPE(GstEmotieff, gst_emotieff, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_CONFIG_FILE, PROP_GPU_ID, PROP_UNIQUE_ID, PROP_OPERATE_ON_GIE_ID };
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK,
    GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string){ NV12, RGBA }"));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC,
    GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string){ NV12, RGBA }"));

static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *spec) {
    auto *self = reinterpret_cast<GstEmotieff *>(object);
    switch (id) {
    case PROP_CONFIG_FILE: g_free(self->config_file); self->config_file = g_value_dup_string(value); break;
    case PROP_GPU_ID: self->gpu_id = g_value_get_uint(value); break;
    case PROP_UNIQUE_ID: self->unique_id = g_value_get_uint(value); break;
    case PROP_OPERATE_ON_GIE_ID: self->operate_on_gie_id = g_value_get_uint(value); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
    }
}

static void get_property(GObject *object, guint id, GValue *value, GParamSpec *spec) {
    auto *self = reinterpret_cast<GstEmotieff *>(object);
    switch (id) {
    case PROP_CONFIG_FILE: g_value_set_string(value, self->config_file); break;
    case PROP_GPU_ID: g_value_set_uint(value, self->gpu_id); break;
    case PROP_UNIQUE_ID: g_value_set_uint(value, self->unique_id); break;
    case PROP_OPERATE_ON_GIE_ID: g_value_set_uint(value, self->operate_on_gie_id); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
    }
}

static gboolean start(GstBaseTransform *transform) {
    auto *self = reinterpret_cast<GstEmotieff *>(transform);
    try {
        if (!self->config_file || !*self->config_file) throw std::runtime_error("config-file is required");
        if (self->unique_id == self->operate_on_gie_id)
            throw std::runtime_error("emotion unique-id conflicts with face component");
        if (cudaSetDevice(self->gpu_id) != cudaSuccess)
            throw std::runtime_error("cannot select configured GPU");
        auto runtime = std::make_unique<Runtime>();
        runtime->config = emotieff::load_config(self->config_file);
        runtime->pipeline = std::make_unique<emotieff::Pipeline>(runtime->config, self->gpu_id);
        runtime->pipeline->warmup();
        self->runtime = runtime.release();
        return TRUE;
    } catch (const std::exception &error) {
        GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS, ("emotion startup failed"), ("%s", error.what()));
        return FALSE;
    }
}

static gboolean stop(GstBaseTransform *transform) {
    auto *self = reinterpret_cast<GstEmotieff *>(transform);
    delete self->runtime;
    self->runtime = nullptr;
    return TRUE;
}

static gboolean sink_event(GstBaseTransform *transform, GstEvent *event) {
    auto *self = reinterpret_cast<GstEmotieff *>(transform);
    if (self->runtime && (GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_STOP ||
                          GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT))
        self->runtime->sources.clear();
    return GST_BASE_TRANSFORM_CLASS(gst_emotieff_parent_class)->sink_event(transform, event);
}

static GstFlowReturn transform_ip(GstBaseTransform *transform, GstBuffer *buffer) {
    auto *self = reinterpret_cast<GstEmotieff *>(transform);
    if (!self->runtime) return GST_FLOW_ERROR;
    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        GST_ELEMENT_ERROR(self, RESOURCE, READ, ("cannot map emotion input buffer"), (nullptr));
        return GST_FLOW_ERROR;
    }
    try {
        auto *surface = reinterpret_cast<NvBufSurface *>(map.data);
        auto *batch = gst_buffer_get_nvds_batch_meta(buffer);
        if (!surface || !surface->surfaceList || !batch)
            throw std::runtime_error("NvBufSurface and NvDsBatchMeta are required");
        if (surface->gpuId != self->gpu_id)
            throw std::runtime_error("input GPU does not match configured emotion GPU");
        for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
            auto *frame = static_cast<NvDsFrameMeta *>(node->data);
            if (!frame || frame->batch_id >= surface->numFilled) continue;
            auto &state = self->runtime->sources[frame->source_id];
            if (frame->frame_num <= state.last_frame) state.seen = 0;
            state.last_frame = frame->frame_num;
            const guint64 count = state.seen++;
            if (count % (guint64(self->runtime->config.interval) + 1) != 0) continue;
            std::vector<NvDsObjectMeta *> faces;
            for (NvDsMetaList *item = frame->obj_meta_list; item; item = item->next) {
                auto *object = static_cast<NvDsObjectMeta *>(item->data);
                if (is_face(object, self->operate_on_gie_id)) faces.push_back(object);
            }
            if (faces.empty()) continue;
            cv::Mat image = convert_frame(*self->runtime, surface, frame->batch_id, self->gpu_id);
            for (auto *object : faces) {
                auto &rect = object->detector_bbox_info.org_bbox_coords;
                cv::Rect2f box(rect.left, rect.top, rect.width, rect.height);
                const auto result = self->runtime->pipeline->analyze(image, box);
                attach_result(batch, object, result, self->runtime->config.model, self->unique_id);
                GST_DEBUG_OBJECT(self, "source=%u frame=%u class=%d score=%.3f",
                                 frame->source_id, frame->frame_num, result.index,
                                 result.index >= 0 ? result.probabilities[result.index] : 0.f);
            }
        }
    } catch (const std::exception &error) {
        gst_buffer_unmap(buffer, &map);
        GST_ELEMENT_ERROR(self, STREAM, FAILED, ("emotion processing failed"), ("%s", error.what()));
        return GST_FLOW_ERROR;
    }
    gst_buffer_unmap(buffer, &map);
    return GST_FLOW_OK;
}

static void finalize(GObject *object) {
    auto *self = reinterpret_cast<GstEmotieff *>(object);
    delete self->runtime;
    g_free(self->config_file);
    G_OBJECT_CLASS(gst_emotieff_parent_class)->finalize(object);
}

static void gst_emotieff_class_init(GstEmotieffClass *klass) {
    auto *obj = G_OBJECT_CLASS(klass);
    auto *element = GST_ELEMENT_CLASS(klass);
    auto *base = GST_BASE_TRANSFORM_CLASS(klass);
    obj->set_property = set_property;
    obj->get_property = get_property;
    obj->finalize = finalize;
    base->start = start;
    base->stop = stop;
    base->sink_event = sink_event;
    base->transform_ip = transform_ip;
    gst_element_class_add_static_pad_template(element, &sink_template);
    gst_element_class_add_static_pad_template(element, &src_template);
    gst_element_class_set_static_metadata(element, "DeepStream emotion recognition",
        "Filter/Effect/Video", "EmotiEff on facedetect objects", "deepstream-app-custom");
    const auto flags = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY);
    g_object_class_install_property(obj, PROP_CONFIG_FILE,
        g_param_spec_string("config-file", "Config file", "Emotion plugin YAML configuration", nullptr, flags));
    g_object_class_install_property(obj, PROP_GPU_ID,
        g_param_spec_uint("gpu-id", "GPU ID", "CUDA GPU ID", 0, G_MAXUINT, 0, flags));
    g_object_class_install_property(obj, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID", "Emotion classifier component ID", 1, G_MAXINT, 17, flags));
    g_object_class_install_property(obj, PROP_OPERATE_ON_GIE_ID,
        g_param_spec_uint("operate-on-gie-id", "Face component ID", "Upstream facedetect component ID",
                          1, G_MAXINT, 16, flags));
}

static void gst_emotieff_init(GstEmotieff *self) {
    self->gpu_id = 0;
    self->unique_id = 17;
    self->operate_on_gie_id = 16;
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}

static gboolean plugin_init(GstPlugin *plugin) {
    GST_DEBUG_CATEGORY_INIT(emotieff_debug, "emotieff", 0, "DeepStream emotion recognition");
    return gst_element_register(plugin, "emotieff", GST_RANK_NONE, gst_emotieff_get_type());
}

GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, emotieff,
                  "DeepStream emotion classifier", plugin_init, VERSION, LICENSE, BINARY_PACKAGE, URL)
