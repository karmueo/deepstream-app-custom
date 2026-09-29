#!/usr/bin/env bash
set -euo pipefail
model_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -x "${model_dir}/../build/face-build-engines" ]]; then
  builder="${model_dir}/../build/face-build-engines"
else
  builder="face-build-engines"
fi
exec "${builder}" --models "${model_dir}" --output "${model_dir}" "$@"
