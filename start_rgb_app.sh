#!/usr/bin/env bash
set -euo pipefail

export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
export DISPLAY="${DISPLAY:-:0}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

if [[ -z "${XAUTHORITY:-}" ]]; then
  if [[ -f "/run/user/$(id -u)/gdm/Xauthority" ]]; then
    export XAUTHORITY="/run/user/$(id -u)/gdm/Xauthority"
  elif [[ -f "${HOME}/.Xauthority" ]]; then
    export XAUTHORITY="${HOME}/.Xauthority"
  fi
fi

cd /opt/nvidia/deepstream/deepstream
exec /opt/nvidia/deepstream/deepstream/bin/deepstream-app \
  -c /opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/yml/app_config.yml
