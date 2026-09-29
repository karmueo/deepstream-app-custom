#!/usr/bin/env bash
set -euo pipefail
model_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_dir="${EMOTIEFF_MODEL_SOURCE:-}"
if [[ -z "${source_dir}" ]]; then
    echo "Set EMOTIEFF_MODEL_SOURCE to the directory containing the original ONNX models" >&2
    exit 1
fi
for model in enet_b0_8_best_vgaf mbf_va_mtl; do
    source_file="${source_dir}/${model}.onnx"
    if [[ ! -f "${source_file}" ]]; then
        echo "Missing original ONNX model: ${source_file}" >&2
        exit 1
    fi
    target_file="${model_dir}/${model}.onnx"
    if [[ ! -f "${target_file}" ]]; then
        cp -- "${source_file}" "${target_file}"
    elif ! cmp -s -- "${source_file}" "${target_file}"; then
        echo "Existing ONNX differs from source: ${target_file}" >&2
        exit 1
    fi
    sha256sum "${target_file}"
done
