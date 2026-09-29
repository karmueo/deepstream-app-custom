#!/usr/bin/env bash
set -euo pipefail
plugin_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${1:-${plugin_dir}/build}"
headers_dir="${build_dir}/opencv45"
downloads_dir="${build_dir}/opencv45-debs"
if [[ -f "${headers_dir}/usr/include/opencv4/opencv2/core/version.hpp" ]]; then
  echo "OpenCV headers already prepared: ${headers_dir}/usr/include/opencv4"
  exit 0
fi
mkdir -p "${headers_dir}" "${downloads_dir}"
cd "${downloads_dir}"
apt download \
  libopencv-core-dev=4.5.4+dfsg-9ubuntu4 \
  libopencv-imgproc-dev=4.5.4+dfsg-9ubuntu4 \
  libopencv-imgcodecs-dev=4.5.4+dfsg-9ubuntu4
for package in ./*.deb; do
  dpkg-deb -x "${package}" "${headers_dir}"
done
if [[ ! -f "${headers_dir}/usr/include/opencv4/opencv2/core/version.hpp" ]]; then
  echo "Failed to prepare OpenCV 4.5 headers" >&2
  exit 1
fi
echo "OpenCV 4.5 headers: ${headers_dir}/usr/include/opencv4"
