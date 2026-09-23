#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $0 <ONNX_PATH> <ENGINE_PATH> [fp16]

Examples:
  $0 model.onnx model.engine fp16
TRTEXEC can be used to override the trtexec executable path.
EOF
}

if [[ $# -lt 2 || $# -gt 3 || ( $# -eq 3 && $3 != fp16 ) ]]; then
    usage >&2
    exit 1
fi

onnx_path=$1
engine_path=$2
shift 2

if [[ ! -f "$onnx_path" ]]; then
    echo "ONNX model does not exist: $onnx_path" >&2
    exit 2
fi

trtexec=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
if [[ ! -x "$trtexec" ]]; then
    echo "trtexec is not executable: $trtexec" >&2
    exit 2
fi

args=(
    "--onnx=$onnx_path"
    "--saveEngine=$engine_path"
    --profilingVerbosity=detailed
    --verbose
)

if [[ "${1:-}" == "fp16" ]]; then
    args+=(--fp16)
fi

echo "Converting ONNX model to TensorRT engine:"
echo "  ONNX Path:   $onnx_path"
echo "  Engine Path: $engine_path"

"$trtexec" "${args[@]}"
echo "Conversion successful! TensorRT engine saved to: $engine_path"
