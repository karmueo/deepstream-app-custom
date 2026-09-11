#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
trtexec_bin="${TRTEXEC_BIN:-/usr/src/tensorrt/bin/trtexec}"

if [[ ! -x "${trtexec_bin}" ]]; then
    echo "trtexec not found or not executable: ${trtexec_bin}" >&2
    exit 1
fi

build_fp16_engine() {
    local onnx_path="$1"
    local engine_path="$2"
    local temporary_engine="${engine_path}.building"

    if [[ ! -f "${onnx_path}" ]]; then
        echo "ONNX model not found: ${onnx_path}" >&2
        exit 1
    fi

    "${trtexec_bin}" \
        --onnx="${onnx_path}" \
        --saveEngine="${temporary_engine}" \
        --fp16 \
        --skipInference
    mv -f "${temporary_engine}" "${engine_path}"
    "${trtexec_bin}" --loadEngine="${engine_path}" --skipInference
}

nanotrack_dir="${repo_root}/sot_plugin/models"
build_fp16_engine \
    "${nanotrack_dir}/nanotrack_backbone.onnx" \
    "${nanotrack_dir}/nanotrack_backbone_fp16.engine"
build_fp16_engine \
    "${nanotrack_dir}/nanotrack_backbone_search.onnx" \
    "${nanotrack_dir}/nanotrack_backbone_search_fp16.engine"
build_fp16_engine \
    "${nanotrack_dir}/nanotrack_head.onnx" \
    "${nanotrack_dir}/nanotrack_head_fp16.engine"

yolo_dir="${repo_root}/src/deepstream-app/models"
build_fp16_engine \
    "${yolo_dir}/yolo26n_rgb_352_uav_no-p2_b4.onnx" \
    "${yolo_dir}/yolo26n_rgb_352_uav_no-p2_b4.engine"

sha256sum \
    "${nanotrack_dir}/nanotrack_backbone_fp16.engine" \
    "${nanotrack_dir}/nanotrack_backbone_search_fp16.engine" \
    "${nanotrack_dir}/nanotrack_head_fp16.engine" \
    "${yolo_dir}/yolo26n_rgb_352_uav_no-p2_b4.engine"
