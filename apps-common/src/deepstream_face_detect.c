#include "deepstream_face_detect.h"
#include "deepstream_common.h"

gboolean create_face_detect_bin(const NvDsFaceDetectConfig *config, NvDsFaceDetectBin *bin) {
    gboolean ret = FALSE;
    bin->bin = gst_bin_new("face_detect_bin");
    bin->queue = gst_element_factory_make("queue", "face_detect_queue");
    bin->element = gst_element_factory_make("facedetect", "face_detect0");
    if (!bin->bin || !bin->queue || !bin->element) {
        g_printerr("Failed to create facedetect; check GST_PLUGIN_PATH\n");
        return FALSE;
    }
    gst_bin_add_many(GST_BIN(bin->bin), bin->queue, bin->element, NULL);
    if (!gst_element_link(bin->queue, bin->element)) goto done;
    g_object_set(bin->element, "config-file", config->config_file,
                 "gpu-id", config->gpu_id, "unique-id", config->unique_id, NULL);
    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->queue, "sink");
    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->element, "src");
    ret = TRUE;
done:
    if (!ret) g_printerr("Failed to link face detection bin\n");
    return ret;
}
