#!/usr/bin/env bash
set -euo pipefail

unset DISPLAY
unset XAUTHORITY
unset WAYLAND_DISPLAY

exec /opt/deepstream-app-custom/bin/deepstream-app-custom \
  -c /opt/deepstream-app-custom/configs/yml/app_config.yml
