#ifndef EMOTIEFF_OSD_H
#define EMOTIEFF_OSD_H

#include "face_metadata.h"
#include <nvdsmeta.h>

#ifdef __cplusplus
extern "C" {
#endif
/* Caller owns the returned display string. */
gchar *format_face_emotion_label(const FaceDetectMeta *face,
                                 const NvDsObjectMeta *object, guint emotion_unique_id);
#ifdef __cplusplus
}
#endif
#endif
