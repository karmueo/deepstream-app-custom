#!/usr/bin/env bash
set -euo pipefail
model_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
archive="${model_dir}/buffalo_sc.zip"
if [[ ! -f "${archive}" ]]; then
  curl -fL --retry 3 --connect-timeout 20 --max-time 600 \
    -o "${archive}.part" \
    'https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_sc.zip'
  mv -- "${archive}.part" "${archive}"
fi
unzip -o "${archive}" -d "${model_dir}"
sha256sum "${model_dir}/det_500m.onnx" "${model_dir}/w600k_mbf.onnx"
