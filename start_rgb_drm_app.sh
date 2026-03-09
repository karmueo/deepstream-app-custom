#!/usr/bin/env bash
set -euo pipefail

export GST_PLUGIN_PATH="/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
unset DISPLAY
unset XAUTHORITY
unset WAYLAND_DISPLAY

cd /opt/nvidia/deepstream/deepstream
exec /opt/nvidia/deepstream/deepstream/bin/deepstream-app \
  -c /opt/nvidia/deepstream/deepstream/deepstream-app-custom/configs/yml/app_config.yml
