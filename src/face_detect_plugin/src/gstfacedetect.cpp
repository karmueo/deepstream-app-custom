#include "face_core.hpp"
#include "face_metadata.h"
#include <algorithm>
#include <cstring>
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

GST_DEBUG_CATEGORY_STATIC(face_detect_debug);
#define GST_CAT_DEFAULT face_detect_debug

namespace {
struct SourceState { guint64 last_frame = 0, seen = 0; };
struct Runtime {
    face::Config config;
    std::unique_ptr<face::Pipeline> pipeline;
    std::vector<face::Sample> gallery;
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
        throw std::runtime_error("cannot allocate RGBA conversion surface");
    runtime.width = width; runtime.height = height;
}
cv::Mat convert_frame(Runtime &runtime, NvBufSurface *input, guint batch_id, guint gpu_id) {
    auto &src = input->surfaceList[batch_id];
    if (src.width == 0 || src.height == 0) throw std::runtime_error("invalid frame dimensions");
    ensure_surface(runtime, src.width, src.height, gpu_id);
    NvBufSurfTransformConfigParams session{};
    session.compute_mode = NvBufSurfTransformCompute_Default;
    session.gpu_id = gpu_id;
    if (NvBufSurfTransformSetSessionParams(&session) != NvBufSurfTransformError_Success)
        throw std::runtime_error("NvBufSurfTransform session failed");
    NvBufSurface single = *input;
    single.surfaceList = &src;
    single.batchSize = single.numFilled = 1;
    NvBufSurfTransformParams transform{};
    transform.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
    transform.transform_filter = NvBufSurfTransformInter_Default;
    if (NvBufSurfTransform(&single, runtime.converted, &transform) !=
        NvBufSurfTransformError_Success)
        throw std::runtime_error("NVMM to RGBA conversion failed");
    if (NvBufSurfaceMap(runtime.converted, 0, 0, NVBUF_MAP_READ) != 0)
        throw std::runtime_error("cannot map converted surface");
    try {
        if (runtime.converted->memType == NVBUF_MEM_SURFACE_ARRAY &&
            NvBufSurfaceSyncForCpu(runtime.converted, 0, 0) != 0)
            throw std::runtime_error("converted surface CPU sync failed");
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
void add_face(NvDsBatchMeta *batch, NvDsFrameMeta *frame, const face::Detection &det,
              const face::Match &match, guint unique_id, guint width, guint height) {
    const float left = std::clamp(det.box.x, 0.f, float(width));
    const float top = std::clamp(det.box.y, 0.f, float(height));
    const float right = std::clamp(det.box.x + det.box.width, 0.f, float(width));
    const float bottom = std::clamp(det.box.y + det.box.height, 0.f, float(height));
    if (right <= left || bottom <= top) return;
    auto *obj = nvds_acquire_obj_meta_from_pool(batch);
    auto *user = nvds_acquire_user_meta_from_pool(batch);
    if (!obj || !user) throw std::runtime_error("DeepStream metadata pool exhausted");
    obj->unique_component_id = unique_id;
    obj->class_id = 0;
    obj->object_id = UNTRACKED_OBJECT_ID;
    obj->confidence = det.score;
    obj->rect_params.left = left;
    obj->rect_params.top = top;
    obj->rect_params.width = right - left;
    obj->rect_params.height = bottom - top;
    obj->rect_params.border_width = 2;
    obj->rect_params.border_color = {0.2f, 1.f, 0.4f, 1.f};
    obj->detector_bbox_info.org_bbox_coords.left = left;
    obj->detector_bbox_info.org_bbox_coords.top = top;
    obj->detector_bbox_info.org_bbox_coords.width = right - left;
    obj->detector_bbox_info.org_bbox_coords.height = bottom - top;
    g_strlcpy(obj->obj_label, "face", sizeof(obj->obj_label));
    auto *value = g_new0(FaceDetectMeta, 1);
    value->person_id = match.person;
    value->name = g_strdup(match.name.c_str());
    value->matched = match.matched;
    value->similarity = match.similarity;
    for (int i = 0; i < 5; ++i) {
        value->landmarks[2 * i] = det.points[i].x;
        value->landmarks[2 * i + 1] = det.points[i].y;
    }
    user->user_meta_data = value;
    user->base_meta.meta_type = nvds_get_user_meta_type(const_cast<gchar *>(FACE_DETECT_META_TYPE));
    user->base_meta.copy_func = face_detect_meta_copy;
    user->base_meta.release_func = face_detect_meta_release;
    nvds_add_user_meta_to_obj(obj, user);
    nvds_add_obj_meta_to_frame(frame, obj, nullptr);
}
} // namespace

typedef struct _GstFaceDetect {
    GstBaseTransform parent;
    gchar *config_file;
    guint gpu_id, unique_id;
    Runtime *runtime;
} GstFaceDetect;
typedef struct _GstFaceDetectClass { GstBaseTransformClass parent_class; } GstFaceDetectClass;

G_DEFINE_TYPE(GstFaceDetect, gst_face_detect, GST_TYPE_BASE_TRANSFORM)

enum { PROP_0, PROP_CONFIG_FILE, PROP_GPU_ID, PROP_UNIQUE_ID };
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE("sink", GST_PAD_SINK,
    GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string){ NV12, RGBA }"));
static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE("src", GST_PAD_SRC,
    GST_PAD_ALWAYS, GST_STATIC_CAPS("video/x-raw(memory:NVMM), format=(string){ NV12, RGBA }"));

static void set_property(GObject *object, guint id, const GValue *value, GParamSpec *spec) {
    auto *self = reinterpret_cast<GstFaceDetect *>(object);
    switch (id) {
    case PROP_CONFIG_FILE: g_free(self->config_file); self->config_file = g_value_dup_string(value); break;
    case PROP_GPU_ID: self->gpu_id = g_value_get_uint(value); break;
    case PROP_UNIQUE_ID: self->unique_id = g_value_get_uint(value); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
    }
}
static void get_property(GObject *object, guint id, GValue *value, GParamSpec *spec) {
    auto *self = reinterpret_cast<GstFaceDetect *>(object);
    switch (id) {
    case PROP_CONFIG_FILE: g_value_set_string(value, self->config_file); break;
    case PROP_GPU_ID: g_value_set_uint(value, self->gpu_id); break;
    case PROP_UNIQUE_ID: g_value_set_uint(value, self->unique_id); break;
    default: G_OBJECT_WARN_INVALID_PROPERTY_ID(object, id, spec);
    }
}
static gboolean start(GstBaseTransform *transform) {
    auto *self = reinterpret_cast<GstFaceDetect *>(transform);
    try {
        if (!self->config_file || !*self->config_file)
            throw std::runtime_error("config-file is required");
        if (cudaSetDevice(self->gpu_id) != cudaSuccess)
            throw std::runtime_error("cannot select configured GPU");
        auto runtime = std::make_unique<Runtime>();
        runtime->config = face::load_config(self->config_file);
        runtime->pipeline = std::make_unique<face::Pipeline>(runtime->config, self->gpu_id);
        runtime->pipeline->warmup();
        if (!runtime->config.gallery_file.empty()) {
            face::Gallery gallery(runtime->config.gallery_file, true);
            runtime->gallery = gallery.samples(runtime->pipeline->model_id());
            if (runtime->gallery.empty())
                GST_WARNING_OBJECT(self, "gallery has no samples for current recognition model");
        }
        self->runtime = runtime.release();
        return TRUE;
    } catch (const std::exception &e) {
        GST_ELEMENT_ERROR(self, RESOURCE, SETTINGS, ("face detection startup failed"), ("%s", e.what()));
        return FALSE;
    }
}
static gboolean stop(GstBaseTransform *transform) {
    auto *self = reinterpret_cast<GstFaceDetect *>(transform);
    delete self->runtime;
    self->runtime = nullptr;
    return TRUE;
}
static gboolean sink_event(GstBaseTransform *transform, GstEvent *event) {
    auto *self = reinterpret_cast<GstFaceDetect *>(transform);
    if (self->runtime && (GST_EVENT_TYPE(event) == GST_EVENT_FLUSH_STOP ||
                          GST_EVENT_TYPE(event) == GST_EVENT_SEGMENT))
        self->runtime->sources.clear();
    return GST_BASE_TRANSFORM_CLASS(gst_face_detect_parent_class)->sink_event(transform, event);
}
static GstFlowReturn transform_ip(GstBaseTransform *transform, GstBuffer *buffer) {
    auto *self = reinterpret_cast<GstFaceDetect *>(transform);
    if (!self->runtime) return GST_FLOW_ERROR;
    GstMapInfo map{};
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        GST_ELEMENT_ERROR(self, RESOURCE, READ, ("cannot map input GstBuffer"), (nullptr));
        return GST_FLOW_ERROR;
    }
    try {
        auto *surface = reinterpret_cast<NvBufSurface *>(map.data);
        auto *batch = gst_buffer_get_nvds_batch_meta(buffer);
        if (!surface || !surface->surfaceList || !batch)
            throw std::runtime_error("NvBufSurface and NvDsBatchMeta are required");
        if (surface->gpuId != self->gpu_id)
            throw std::runtime_error("input GPU does not match configured GPU");
        for (NvDsMetaList *node = batch->frame_meta_list; node; node = node->next) {
            auto *frame = static_cast<NvDsFrameMeta *>(node->data);
            if (!frame || frame->batch_id >= surface->numFilled) continue;
            auto &state = self->runtime->sources[frame->source_id];
            if (frame->frame_num <= state.last_frame) state.seen = 0;
            state.last_frame = frame->frame_num;
            const auto count = state.seen++;
            if (count % (self->runtime->config.interval + 1) != 0) continue;
            cv::Mat image = convert_frame(*self->runtime, surface, frame->batch_id, self->gpu_id);
            auto faces = self->runtime->pipeline->analyze(image);
            GST_INFO_OBJECT(self, "source=%u frame=%u faces=%zu", frame->source_id,
                            frame->frame_num, faces.size());
            for (const auto &det : faces) {
                const auto found = face::match(det.embedding, self->runtime->gallery,
                                                self->runtime->config.recognition_threshold);
                GST_DEBUG_OBJECT(self, "source=%u frame=%u identity=%s similarity=%.3f matched=%d",
                                 frame->source_id, frame->frame_num, found.name.c_str(),
                                 found.similarity, found.matched);
                add_face(batch, frame, det, found, self->unique_id, image.cols, image.rows);
            }
        }
    } catch (const std::exception &e) {
        gst_buffer_unmap(buffer, &map);
        GST_ELEMENT_ERROR(self, STREAM, FAILED, ("face detection processing failed"), ("%s", e.what()));
        return GST_FLOW_ERROR;
    }
    gst_buffer_unmap(buffer, &map);
    return GST_FLOW_OK;
}
static void finalize(GObject *object) {
    auto *self = reinterpret_cast<GstFaceDetect *>(object);
    delete self->runtime;
    g_free(self->config_file);
    G_OBJECT_CLASS(gst_face_detect_parent_class)->finalize(object);
}
static void gst_face_detect_class_init(GstFaceDetectClass *klass) {
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
    gst_element_class_set_static_metadata(element, "DeepStream face detection and recognition",
        "Filter/Effect/Video", "SCRFD and ArcFace on NVMM frames", "deepstream-app-custom");
    const auto flags = static_cast<GParamFlags>(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS | GST_PARAM_MUTABLE_READY);
    g_object_class_install_property(obj, PROP_CONFIG_FILE,
        g_param_spec_string("config-file", "Config file", "Face plugin YAML configuration",
                            nullptr, flags));
    g_object_class_install_property(obj, PROP_GPU_ID,
        g_param_spec_uint("gpu-id", "GPU ID", "CUDA GPU ID", 0, G_MAXUINT, 0, flags));
    g_object_class_install_property(obj, PROP_UNIQUE_ID,
        g_param_spec_uint("unique-id", "Unique ID", "DeepStream component ID", 1, G_MAXINT, 16, flags));
}
static void gst_face_detect_init(GstFaceDetect *self) {
    self->gpu_id = 0; self->unique_id = 16;
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
}
static gboolean plugin_init(GstPlugin *plugin) {
    GST_DEBUG_CATEGORY_INIT(face_detect_debug, "facedetect", 0, "DeepStream face detection");
    return gst_element_register(plugin, "facedetect", GST_RANK_NONE, gst_face_detect_get_type());
}
GST_PLUGIN_DEFINE(GST_VERSION_MAJOR, GST_VERSION_MINOR, facedetect,
                  "DeepStream face detector", plugin_init, VERSION, LICENSE, BINARY_PACKAGE, URL)
