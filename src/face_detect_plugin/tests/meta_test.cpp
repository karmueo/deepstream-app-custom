#include "face_metadata.h"
#include <nvdsmeta.h>
#include <cstring>
#include <stdexcept>

int main() {
    NvDsUserMeta original{};
    auto *value = g_new0(FaceDetectMeta, 1);
    value->person_id = 42;
    value->name = g_strdup("张三：来自较长的 UTF-8 人员姓名");
    value->matched = TRUE;
    value->similarity = .731f;
    value->landmarks[9] = 123.25f;
    original.user_meta_data = value;
    auto *copy = static_cast<FaceDetectMeta *>(face_detect_meta_copy(&original, nullptr));
    if (!copy || copy == value || copy->name == value->name ||
        std::strcmp(copy->name, value->name) || copy->person_id != value->person_id ||
        copy->matched != value->matched || copy->similarity != value->similarity ||
        copy->landmarks[9] != value->landmarks[9])
        throw std::runtime_error("face metadata deep copy failed");
    face_detect_meta_release(&original, nullptr);
    NvDsUserMeta copied{};
    copied.user_meta_data = copy;
    if (std::strcmp(copy->name, "张三：来自较长的 UTF-8 人员姓名"))
        throw std::runtime_error("copied name lost after original release");
    face_detect_meta_release(&copied, nullptr);
    return copied.user_meta_data == nullptr ? 0 : 1;
}
