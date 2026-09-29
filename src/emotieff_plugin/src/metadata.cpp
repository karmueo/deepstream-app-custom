#include "emotieff_metadata.h"
#include <nvdsmeta.h>

extern "C" gpointer emotieff_meta_copy(gpointer data, gpointer) {
    const auto *source = static_cast<NvDsUserMeta *>(data);
    const auto *value = static_cast<EmotieffMeta *>(source->user_meta_data);
    if (!value) return nullptr;
    auto *copy = g_new(EmotieffMeta, 1);
    *copy = *value;
    return copy;
}

extern "C" void emotieff_meta_release(gpointer data, gpointer) {
    auto *meta = static_cast<NvDsUserMeta *>(data);
    g_free(meta->user_meta_data);
    meta->user_meta_data = nullptr;
}
