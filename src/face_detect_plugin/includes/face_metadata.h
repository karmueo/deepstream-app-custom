#ifndef FACE_DETECT_METADATA_H
#define FACE_DETECT_METADATA_H

#include <glib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define FACE_DETECT_META_TYPE "DEEPSTREAM_FACE_DETECT_META"

/* Coordinates refer to the plugin input surface. name owns UTF-8 storage. */
typedef struct FaceDetectMeta {
    gint64 person_id;
    gchar *name;
    gboolean matched;
    gfloat similarity;
    gfloat landmarks[10];
} FaceDetectMeta;

gpointer face_detect_meta_copy(gpointer data, gpointer user_data);
void face_detect_meta_release(gpointer data, gpointer user_data);

#ifdef __cplusplus
}
#endif

#endif
