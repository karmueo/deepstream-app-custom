#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
plugin_dir="${repo_dir}/src/face_detect_plugin"
work_dir="${plugin_dir}/build/test-video"
mkdir -p "${work_dir}"
python3 - "${repo_dir}" "${work_dir}" <<'PY'
from pathlib import Path
import re
import os
import sys

root, work = map(Path, sys.argv[1:])
test_video = Path(os.environ.get(
    'FACE_TEST_VIDEO',
    '/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4',
)).expanduser().resolve()
if not test_video.is_file():
    raise RuntimeError(f'FACE_TEST_VIDEO does not exist: {test_video}')
config = (root / 'src/deepstream-app/configs/yml/app_config.yml').read_text()
plugin_root = root / 'src/face_detect_plugin'
sources = int(os.environ.get('FACE_TEST_SOURCES', '1'))
interval = int(os.environ.get('FACE_TEST_INTERVAL', '0'))
with_yolo = os.environ.get('FACE_TEST_WITH_YOLO', '0') == '1'
with_preprocess = os.environ.get('FACE_TEST_WITH_PREPROCESS', '1' if with_yolo else '0') == '1'
with_tracker = os.environ.get('FACE_TEST_WITH_TRACKER', '1' if with_yolo else '0') == '1'
with_face = os.environ.get('FACE_TEST_WITH_FACE', '1') == '1'
if sources not in (1, 2) or interval < 0 or interval > 10000:
    raise RuntimeError('FACE_TEST_SOURCES must be 1 or 2; interval must be 0..10000')

def set_value(group: str, key: str, value: str) -> None:
    global config
    section = re.compile(rf'(?ms)^{re.escape(group)}:\n(.*?)(?=^[^\s#][^\n]*:|\Z)')
    found = section.search(config)
    if not found:
        raise RuntimeError(f'missing section {group}')
    block = found.group(1)
    entry = re.compile(rf'(?m)^  {re.escape(key)}:.*$')
    if not entry.search(block):
        raise RuntimeError(f'missing {group}.{key}')
    block = entry.sub(f'  {key}: {value}', block, count=1)
    config = config[:found.start(1)] + block + config[found.end(1):]

set_value('source', 'csv-file-path', str(work / 'file_sources.csv'))
set_value('pre-process', 'enable', '1' if with_preprocess else '0')
set_value('primary-gie', 'enable', '1' if with_yolo else '0')
if with_yolo and not with_preprocess:
    set_value('primary-gie', 'input-tensor-meta', '0')
set_value('tracker', 'enable', '1' if with_tracker else '0')
if with_yolo:
    set_value('pre-process', 'config-file', str(root / 'src/deepstream-app/configs/config_preprocess_rgb_352_primary.txt'))
    set_value('primary-gie', 'config-file', str(root / 'src/deepstream-app/configs/yml/config_infer_primary_yolo_352_rgb.yml'))
set_value('face-detect', 'enable', '1' if with_face else '0')
set_value('emotieff', 'enable', '0')
set_value('face-detect', 'config-file', str(work / 'face_plugin.yml'))
set_value('sink0', 'enable', '0')
set_value('tiled-display', 'rows', '1')
set_value('tiled-display', 'columns', str(sources))
set_value('streammux', 'live-source', '0')
set_value('streammux', 'batch-size', str(sources))
set_value('tests', 'file-loop', '0')
plugin_config = (plugin_root / 'config_face_detect.yml').read_text()
plugin_config = plugin_config.replace('model/detector.engine', str(plugin_root / 'model/detector.engine'))
plugin_config = plugin_config.replace('model/recognizer.engine', str(plugin_root / 'model/recognizer.engine'))
plugin_config = re.sub(r'(?m)^  gallery-file:.*$',
                       f'  gallery-file: "{plugin_root / "data/gallery.sqlite"}"',
                       plugin_config)
plugin_config = plugin_config.replace('interval: 0', f'interval: {interval}')
if os.environ.get('FACE_TEST_EMPTY_GALLERY', '0') == '1':
    plugin_config = re.sub(r'(?m)^  gallery-file:.*$', '  gallery-file: ""', plugin_config)
(work / 'face_plugin.yml').write_text(plugin_config)
sink = os.environ.get('FACE_TEST_SINK', 'file')
if sink == 'file':
    set_value('sink1', 'enable', '0')
    set_value('sink3', 'enable', '1')
    set_value('sink3', 'output-file', str(work / 'annotated.mkv'))
elif sink == 'fake':
    set_value('sink1', 'enable', '0')
    set_value('sink4', 'enable', '1')
elif sink != 'egl':
    raise RuntimeError('FACE_TEST_SINK must be egl, file or fake')
(work / 'face_test.yml').write_text(config)
csv = ('enable,type,uri,num-sources,gpu-id,cudadec-memtype,rtsp-reconnect-interval-sec,'
       'rtsp-reconnect-attempts,select-rtp-protocol,smart-record,smart-rec-dir-path,'
       'smart-rec-duration,smart-rec-start-time\n')
csv += f'1,2,{test_video.as_uri()},1,0,0,0,0,0,0,{work},30,3\n'
if sources == 2:
    csv += f'1,2,{test_video.as_uri()},1,0,0,0,0,0,0,{work},30,3\n'
(work / 'file_sources.csv').write_text(csv)
PY
export DISPLAY="${DISPLAY:-:10.0}"
export GST_PLUGIN_PATH="${plugin_dir}/build/lib:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:/opt/deepstream-app-custom/lib:${LD_LIBRARY_PATH:-}"
exec "${repo_dir}/build/deepstream-app-custom.bin" -c "${work_dir}/face_test.yml"
