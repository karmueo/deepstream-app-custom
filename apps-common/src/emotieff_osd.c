#include "emotieff_osd.h"
#include "emotieff_metadata.h"

gchar *format_face_emotion_label(const FaceDetectMeta *face,
                                 const NvDsObjectMeta *object, guint emotion_unique_id) {
    if (!face) return NULL;
    const gchar *name = face->name ? face->name : "陌生人";
    if (object && emotion_unique_id) {
        for (NvDsMetaList *node = object->obj_user_meta_list; node; node = node->next) {
            const NvDsUserMeta *user = (const NvDsUserMeta *)node->data;
            if (!user || user->base_meta.meta_type != nvds_get_user_meta_type(EMOTIEFF_META_TYPE))
                continue;
            const EmotieffMeta *emotion = (const EmotieffMeta *)user->user_meta_data;
            if (!emotion || emotion->component_id != emotion_unique_id) continue;
            if (!emotion->valid || emotion->class_id < 0 || emotion->class_id >= EMOTIEFF_CLASS_COUNT)
                return g_strdup_printf("%s %.3f 无法判断", name, face->similarity);
            static const gchar *labels[] = {"愤怒", "轻蔑", "厌恶", "恐惧", "高兴", "中性", "悲伤", "惊讶"};
            return g_strdup_printf("%s %.3f %s %.3f", name, face->similarity,
                                   labels[emotion->class_id], emotion->confidence);
        }
    }
    return g_strdup_printf("%s %.3f", name, face->similarity);
}
