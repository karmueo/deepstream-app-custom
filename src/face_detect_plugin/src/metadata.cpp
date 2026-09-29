#include "face_metadata.h"
#include <nvdsmeta.h>

extern "C" gpointer face_detect_meta_copy(gpointer data, gpointer) {
    const auto *src = static_cast<NvDsUserMeta *>(data);
    const auto *value = static_cast<FaceDetectMeta *>(src->user_meta_data);
    auto *copy = g_new0(FaceDetectMeta, 1);
    *copy = *value;
    copy->name = g_strdup(value->name);
    return copy;
}
extern "C" void face_detect_meta_release(gpointer data, gpointer) {
    auto *meta = static_cast<NvDsUserMeta *>(data);
    auto *value = static_cast<FaceDetectMeta *>(meta->user_meta_data);
    if (value) { g_free(value->name); g_free(value); meta->user_meta_data = nullptr; }
}
