#!/usr/bin/env bash
set -euo pipefail
plugin_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
model_dir="${plugin_dir}/model"
builder="${plugin_dir}/build/emotieff-build-engine"
if [[ ! -x "${builder}" ]]; then
    builder="${plugin_dir}/../bin/emotieff-build-engine"
fi
if [[ ! -x "${builder}" ]]; then
    echo "Build ${builder} first" >&2
    exit 1
fi
for model in enet_b0_8_best_vgaf mbf_va_mtl; do
    "${builder}" "${model}" "${model_dir}/${model}.onnx" \
        "${model_dir}/${model}.engine" "${EMOTIEFF_GPU_ID:-0}"
done
