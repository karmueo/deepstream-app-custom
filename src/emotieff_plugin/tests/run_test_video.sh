#!/usr/bin/env bash
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
plugin_dir="${repo_dir}/src/emotieff_plugin"
work_dir="${plugin_dir}/build/test-video"
mkdir -p "${work_dir}"
python3 - "${repo_dir}" "${work_dir}" <<'PY'
from pathlib import Path
import os
import re
import sys

root, work = map(Path, sys.argv[1:])
face_root = root / 'src/face_detect_plugin'
emotion_root = root / 'src/emotieff_plugin'
video = Path(os.environ.get(
    'EMOTIEFF_TEST_VIDEO', '/opt/nvidia/deepstream/deepstream/samples/streams/sample_720p.mp4',
)).expanduser().resolve()
if not video.is_file():
    raise RuntimeError(f'Video does not exist: {video}')
model = os.environ.get('EMOTIEFF_TEST_MODEL', 'enet_b0_8_best_vgaf')
if model not in ('enet_b0_8_best_vgaf', 'mbf_va_mtl'):
    raise RuntimeError('Unknown EMOTIEFF_TEST_MODEL')
sources = int(os.environ.get('EMOTIEFF_TEST_SOURCES', '1'))
interval = int(os.environ.get('EMOTIEFF_TEST_INTERVAL', '0'))
loop = int(os.environ.get('EMOTIEFF_TEST_LOOP', '0'))
with_yolo = os.environ.get('EMOTIEFF_TEST_WITH_YOLO', '0') == '1'
with_preprocess = os.environ.get('EMOTIEFF_TEST_WITH_PREPROCESS',
                                 '1' if with_yolo else '0') == '1'
with_tracker = os.environ.get('EMOTIEFF_TEST_WITH_TRACKER',
                              '1' if with_yolo else '0') == '1'
if sources not in (1, 2) or not 0 <= interval <= 10000 or loop not in (0, 1):
    raise RuntimeError('Sources must be 1 or 2; interval 0..10000; loop 0 or 1')
if with_tracker and not with_yolo:
    raise RuntimeError('Tracker test requires YOLO')
config = (root / 'src/deepstream-app/configs/yml/app_config.yml').read_text()

def set_value(group: str, key: str, value: str) -> None:
    global config
    section = re.compile(rf'(?ms)^{re.escape(group)}:\n(.*?)(?=^[^\s#][^\n]*:|\Z)')
    match = section.search(config)
    if not match:
        raise RuntimeError(f'Missing section {group}')
    block = match.group(1)
    entry = re.compile(rf'(?m)^  {re.escape(key)}:.*$')
    if not entry.search(block):
        raise RuntimeError(f'Missing {group}.{key}')
    block = entry.sub(f'  {key}: {value}', block, count=1)
    config = config[:match.start(1)] + block + config[match.end(1):]

set_value('source', 'csv-file-path', str(work / 'file_sources.csv'))
set_value('pre-process', 'enable', '1' if with_preprocess else '0')
set_value('primary-gie', 'enable', '1' if with_yolo else '0')
set_value('tracker', 'enable', '1' if with_tracker else '0')
if with_yolo:
    set_value('pre-process', 'config-file',
              str(root / 'src/deepstream-app/configs/config_preprocess_rgb_352_primary.txt'))
    set_value('primary-gie', 'config-file',
              str(root / 'src/deepstream-app/configs/yml/config_infer_primary_yolo_352_rgb.yml'))
    if not with_preprocess:
        set_value('primary-gie', 'input-tensor-meta', '0')
set_value('face-detect', 'enable', '1')
set_value('face-detect', 'config-file', str(work / 'face_plugin.yml'))
set_value('emotieff', 'enable', '1')
set_value('emotieff', 'config-file', str(work / 'emotion_plugin.yml'))
set_value('sink0', 'enable', '0')
set_value('tiled-display', 'rows', '1')
set_value('tiled-display', 'columns', str(sources))
set_value('streammux', 'live-source', '0')
set_value('streammux', 'batch-size', str(sources))
set_value('tests', 'file-loop', str(loop))

face_config = (face_root / 'config_face_detect.yml').read_text()
face_config = face_config.replace('model/detector.engine', str(face_root / 'model/detector.engine'))
face_config = face_config.replace('model/recognizer.engine', str(face_root / 'model/recognizer.engine'))
face_config = re.sub(r'(?m)^  gallery-file:.*$', '  gallery-file: ""', face_config)
(work / 'face_plugin.yml').write_text(face_config)
emotion_config = (emotion_root / 'configs/config_emotieff.yml').read_text()
emotion_config = re.sub(r'(?m)^  emotion-model:.*$', f'  emotion-model: {model}', emotion_config)
emotion_config = re.sub(r'(?m)^  emotion-engine:.*$',
                        f'  emotion-engine: {emotion_root / "model" / (model + ".engine")}',
                        emotion_config)
emotion_config = re.sub(r'(?m)^  interval:.*$', f'  interval: {interval}', emotion_config)
(work / 'emotion_plugin.yml').write_text(emotion_config)

sink = os.environ.get('EMOTIEFF_TEST_SINK', 'file')
if sink == 'file':
    set_value('sink1', 'enable', '0')
    set_value('sink3', 'enable', '1')
    set_value('sink3', 'output-file', str(work / 'annotated.mkv'))
elif sink == 'fake':
    set_value('sink1', 'enable', '0')
    set_value('sink3', 'enable', '1')
elif sink != 'egl':
    raise RuntimeError('EMOTIEFF_TEST_SINK must be file, fake or egl')
(work / 'emotion_test.yml').write_text(config)
csv = ('enable,type,uri,num-sources,gpu-id,cudadec-memtype,rtsp-reconnect-interval-sec,'
       'rtsp-reconnect-attempts,select-rtp-protocol,smart-record,smart-rec-dir-path,'
       'smart-rec-duration,smart-rec-start-time\n')
for _ in range(sources):
    csv += f'1,2,{video.as_uri()},1,0,0,0,0,0,0,{work},30,3\n'
(work / 'file_sources.csv').write_text(csv)
PY
export DISPLAY="${DISPLAY:-:10.0}"
export GST_PLUGIN_PATH="${plugin_dir}/build/lib:${repo_dir}/src/face_detect_plugin/build/lib:/opt/nvidia/deepstream/deepstream/lib/gst-plugins:${GST_PLUGIN_PATH:-}"
export LD_LIBRARY_PATH="/opt/nvidia/deepstream/deepstream/lib:/opt/deepstream-app-custom/lib:${LD_LIBRARY_PATH:-}"
exec "${repo_dir}/build/deepstream-app-custom.bin" -c "${work_dir}/emotion_test.yml"
