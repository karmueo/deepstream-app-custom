#include "deepstream_emotieff.h"
#include "deepstream_common.h"

gboolean create_emotieff_bin(const NvDsEmotieffConfig *config, guint face_unique_id,
                             NvDsEmotieffBin *bin) {
    gboolean result = FALSE;
    bin->bin = gst_bin_new("emotieff_bin");
    bin->queue = gst_element_factory_make("queue", "emotieff_queue");
    bin->element = gst_element_factory_make("emotieff", "emotieff0");
    if (!bin->bin || !bin->queue || !bin->element) {
        g_printerr("Failed to create emotieff; check GST_PLUGIN_PATH\n");
        return FALSE;
    }
    gst_bin_add_many(GST_BIN(bin->bin), bin->queue, bin->element, NULL);
    if (!gst_element_link(bin->queue, bin->element)) goto done;
    g_object_set(bin->element, "config-file", config->config_file,
                 "gpu-id", config->gpu_id, "unique-id", config->unique_id,
                 "operate-on-gie-id", face_unique_id, NULL);
    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->queue, "sink");
    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->element, "src");
    result = TRUE;
done:
    if (!result) g_printerr("Failed to link emotion bin\n");
    return result;
}
