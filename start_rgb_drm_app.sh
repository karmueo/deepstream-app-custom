#!/usr/bin/env bash
set -euo pipefail

runtime_root="/opt/deepstream-app-custom"
export GST_PLUGIN_PATH="${runtime_root}/gst-plugins:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="${runtime_root}/lib:${runtime_root}/gst-plugins:${LD_LIBRARY_PATH:-}"

if [[ -n "${STATE_DIRECTORY:-}" ]]; then
  runtime_state="${STATE_DIRECTORY}"
elif [[ -n "${XDG_STATE_HOME:-}" ]]; then
  runtime_state="${XDG_STATE_HOME}/deepstream-app-custom"
else
  runtime_state="${HOME:?HOME must be set}/.local/state/deepstream-app-custom"
fi
mkdir -p "${runtime_state}/smart_rec_rgb"
cd "${runtime_state}"

unset DISPLAY
unset XAUTHORITY
unset WAYLAND_DISPLAY

exec "${runtime_root}/bin/deepstream-app-custom.bin" \
  -c "${runtime_root}/configs/yml/app_config.yml"
