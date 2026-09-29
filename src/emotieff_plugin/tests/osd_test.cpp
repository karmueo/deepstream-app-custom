#include "emotieff_osd.h"
#include "emotieff_metadata.h"
#include <cassert>
#include <cstring>

int main() {
    FaceDetectMeta face{};
    face.name = const_cast<gchar *>("陌生人");
    face.similarity = .5f;
    NvDsObjectMeta object{};
    gchar *text = format_face_emotion_label(&face, &object, 17);
    assert(std::strcmp(text, "陌生人 0.500") == 0);
    g_free(text);
    NvDsUserMeta user{};
    EmotieffMeta emotion{};
    emotion.component_id = 17;
    emotion.valid = TRUE;
    emotion.class_id = 4;
    emotion.confidence = .932f;
    user.base_meta.meta_type = nvds_get_user_meta_type(const_cast<gchar *>(EMOTIEFF_META_TYPE));
    user.user_meta_data = &emotion;
    GList node{};
    node.data = &user;
    object.obj_user_meta_list = &node;
    text = format_face_emotion_label(&face, &object, 17);
    assert(std::strcmp(text, "陌生人 0.500 高兴 0.932") == 0);
    g_free(text);
    emotion.valid = FALSE;
    text = format_face_emotion_label(&face, &object, 17);
    assert(std::strcmp(text, "陌生人 0.500 无法判断") == 0);
    g_free(text);
}
