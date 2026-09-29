#ifndef EMOTIEFF_METADATA_H
#define EMOTIEFF_METADATA_H

#include <glib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define EMOTIEFF_META_TYPE "DEEPSTREAM_EMOTIEFF_META"
#define EMOTIEFF_CLASS_COUNT 8
#define EMOTIEFF_MAX_LOGITS 10

/* Attached to the original facedetect NvDsObjectMeta. Arrays own their data. */
typedef struct EmotieffMeta {
    guint component_id;
    gchar model[32];
    gboolean valid;
    gint class_id; /* -1 means invalid crop. */
    gfloat confidence;
    gfloat probabilities[EMOTIEFF_CLASS_COUNT];
    gfloat logits[EMOTIEFF_MAX_LOGITS];
    guint logits_count;
} EmotieffMeta;

gpointer emotieff_meta_copy(gpointer data, gpointer user_data);
void emotieff_meta_release(gpointer data, gpointer user_data);

#ifdef __cplusplus
}
#endif
#endif
