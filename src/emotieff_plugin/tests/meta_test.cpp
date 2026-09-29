#include "emotieff_metadata.h"
#include <cassert>
#include <nvdsmeta.h>

int main() {
    NvDsUserMeta user{};
    auto *original = g_new0(EmotieffMeta, 1);
    original->component_id = 17;
    original->class_id = 4;
    original->probabilities[4] = .8f;
    original->logits_count = 10;
    original->logits[9] = 2.4f;
    g_strlcpy(original->model, "mbf_va_mtl", sizeof(original->model));
    user.user_meta_data = original;
    auto *copy = static_cast<EmotieffMeta *>(emotieff_meta_copy(&user, nullptr));
    assert(copy != original && copy->component_id == 17 && copy->class_id == 4);
    assert(copy->probabilities[4] == .8f && copy->logits[9] == 2.4f);
    emotieff_meta_release(&user, nullptr);
    assert(user.user_meta_data == nullptr && copy->logits_count == 10);
    user.user_meta_data = copy;
    emotieff_meta_release(&user, nullptr);
}
