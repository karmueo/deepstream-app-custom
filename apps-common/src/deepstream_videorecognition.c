#include "deepstream_common.h"
#include "deepstream_videorecognition.h"

// Create bin, add queue and the element, link all elements and ghost pads,
// Set the element properties from the parsed config
gboolean
create_dsvideorecognition_bin(NvDsVideoRecognitionConfig *config, NvDsVideoRecognitionBin *bin)
{
    GstCaps *caps = NULL;
    gboolean ret = FALSE;

    bin->bin = gst_bin_new("videorecognition_bin");
    if (!bin->bin)
    {
        NVGSTDS_ERR_MSG_V("Failed to create 'videorecognition_bin'");
        goto done;
    }

    bin->queue = gst_element_factory_make(NVDS_ELEM_QUEUE, "videorecognition_queue");
    if (!bin->queue)
    {
        NVGSTDS_ERR_MSG_V("Failed to create 'videorecognition_queue'");
        goto done;
    }

    bin->elem_dsvideorecognition =
        gst_element_factory_make(NVDS_ELEM_DSVIDEORECOGNITION_ELEMENT, "videorecognition0");
    if (!bin->elem_dsvideorecognition)
    {
        NVGSTDS_ERR_MSG_V("Failed to create 'videorecognition0'");
        goto done;
    }

    bin->pre_conv =
        gst_element_factory_make(NVDS_ELEM_VIDEO_CONV, "videorecognition_conv0");
    if (!bin->pre_conv)
    {
        NVGSTDS_ERR_MSG_V("Failed to create 'videorecognition_conv0'");
        goto done;
    }

    bin->cap_filter =
        gst_element_factory_make(NVDS_ELEM_CAPS_FILTER, "videorecognition_caps");
    if (!bin->cap_filter)
    {
        NVGSTDS_ERR_MSG_V("Failed to create 'videorecognition_caps'");
        goto done;
    }

    gst_bin_add_many(GST_BIN(bin->bin), bin->queue,
                     bin->pre_conv, bin->cap_filter, bin->elem_dsvideorecognition, NULL);

    NVGSTDS_LINK_ELEMENT(bin->queue, bin->pre_conv);
    NVGSTDS_LINK_ELEMENT(bin->pre_conv, bin->cap_filter);
    NVGSTDS_LINK_ELEMENT(bin->cap_filter, bin->elem_dsvideorecognition);

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->queue, "sink");

    NVGSTDS_BIN_ADD_GHOST_PAD(bin->bin, bin->elem_dsvideorecognition, "src");

    g_object_set(G_OBJECT(bin->elem_dsvideorecognition),
                 "unique-id", config->unique_id,
                 "gpu-id", config->gpu_id,
                 NULL);
    if (config->batch_size)
    {
        g_object_set(G_OBJECT(bin->elem_dsvideorecognition), "batch-size", config->batch_size, NULL);
    }
    g_object_set(G_OBJECT(bin->pre_conv), "gpu-id", config->gpu_id, NULL);

    g_object_set(G_OBJECT(bin->pre_conv), "nvbuf-memory-type",
                 config->nvbuf_memory_type, NULL);

    ret = TRUE;

done:
    if (!ret)
    {
        NVGSTDS_ERR_MSG_V("%s failed", __func__);
    }

    return ret;
}
