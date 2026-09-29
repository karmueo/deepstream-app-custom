#ifndef DEEPSTREAM_FACE_DETECT_H
#define DEEPSTREAM_FACE_DETECT_H

#include <gst/gst.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef struct {
    gboolean enable;
    guint gpu_id;
    guint unique_id;
    gchar *config_file;
} NvDsFaceDetectConfig;
typedef struct {
    GstElement *bin;
    GstElement *queue;
    GstElement *element;
} NvDsFaceDetectBin;
gboolean create_face_detect_bin(const NvDsFaceDetectConfig *config, NvDsFaceDetectBin *bin);
#ifdef __cplusplus
}
#endif
#endif
