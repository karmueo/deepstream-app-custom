#ifndef DEEPSTREAM_EMOTIEFF_H
#define DEEPSTREAM_EMOTIEFF_H

#include <gst/gst.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    gboolean enable;
    guint gpu_id;
    guint unique_id;
    gchar *config_file;
} NvDsEmotieffConfig;
typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *element;
} NvDsEmotieffBin;
gboolean create_emotieff_bin(const NvDsEmotieffConfig *config, guint face_unique_id,
                             NvDsEmotieffBin *bin);
#ifdef __cplusplus
}
#endif
#endif
