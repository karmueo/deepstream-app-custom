#!/usr/bin/env python3
"""Verify an extracted deepstream-app-custom release tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Iterable


REQUIRED = (
    "start_rgb_app.sh",
    "start_rgb_drm_app.sh",
    "bin/deepstream-app-custom.bin",
    "configs/yml/app_config.yml",
    "configs/yml/config_infer_primary_yolo_352_rgb.yml",
    "configs/yml/file_sources.csv",
    "configs/config_preprocess_rgb_352_primary.txt",
    "configs/config_sot.yml",
    "configs/labels.txt",
    "face_detect_plugin/config_face_detect.yml",
    "face_detect_plugin/model/det_500m.onnx",
    "face_detect_plugin/model/w600k_mbf.onnx",
    "face_detect_plugin/model/detector.engine",
    "face_detect_plugin/model/recognizer.engine",
    "face_detect_plugin/model/detector.engine.json",
    "face_detect_plugin/model/recognizer.engine.json",
    "models/yolo26n_rgb_352_uav_no-p2_b4.engine",
    "models/yolo26n_rgb_352_uav_no-p2_b4.onnx",
    "models/nanotrack_head_fp16.engine",
    "models/nanotrack_head.onnx",
    "models/nanotrack_backbone_fp16.engine",
    "models/nanotrack_backbone.onnx",
    "models/nanotrack_backbone_search_fp16.engine",
    "models/nanotrack_backbone_search.onnx",
    "models/convert2trt.sh",
    "lib/libsot.so",
    "lib/libnvdsinfer_custom_impl_Yolo.so",
    "lib/libcustom2d_preprocess.so",
    "gst-plugins/libgstfacedetect.so",
)
ALLOWED_SHELL_SCRIPTS = {
    "start_rgb_app.sh",
    "start_rgb_drm_app.sh",
    "models/convert2trt.sh",
}
FORBIDDEN_SUFFIXES = {
    ".c",
    ".cc",
    ".cpp",
    ".cu",
    ".h",
    ".hpp",
    ".py",
    ".sh",
}
TEXT_SUFFIXES = {"", ".conf", ".csv", ".ini", ".json", ".txt", ".yaml", ".yml"}
RTSP_CREDENTIALS = re.compile(rb"rtsp://[^\s,/:]+:[^\s,@]+@", re.IGNORECASE)
RUNTIME_PATHS = re.compile(
    rb"(?:file://)?/opt/deepstream-app-custom/[A-Za-z0-9._/-]+"
)


def _files(root: Path) -> Iterable[Path]:
    return (path for path in root.rglob("*") if path.is_file())


def _contains_bytes(path: Path, needle: bytes) -> bool:
    overlap = b""
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            window = overlap + chunk
            if needle in window:
                return True
            overlap = window[-(len(needle) - 1) :]
    return False


def _run(command: list[str], env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed:\n{result.stdout}")
    return result.stdout


def verify(root: Path, runtime_checks: bool, development: bool) -> list[str]:
    errors: list[str] = []
    elf_files: list[Path] = []
    for relative in REQUIRED:
        if not (root / relative).is_file():
            errors.append(f"missing required file: {relative}")

    for path in _files(root):
        relative = path.relative_to(root)
        if relative.parent == Path("face_detect_plugin/model") and path.name.endswith(
            ".engine.json"
        ):
            try:
                metadata = json.loads(path.read_text(encoding="utf-8"))
                onnx_path = metadata["onnx_path"]
                if not isinstance(onnx_path, str) or not onnx_path:
                    raise ValueError("onnx_path must be a nonempty string")
                if Path(onnx_path).is_absolute():
                    raise ValueError("onnx_path must be relative to the engine")
                source = (path.parent / onnx_path).resolve()
                if not source.is_relative_to(root) or not source.is_file():
                    raise ValueError("onnx_path does not point to a packaged model")
            except (OSError, KeyError, ValueError, TypeError) as error:
                errors.append(f"invalid face engine metadata in {relative}: {error}")
        if (
            path.suffix.lower() in FORBIDDEN_SUFFIXES
            and str(relative) not in ALLOWED_SHELL_SCRIPTS
        ):
            errors.append(f"forbidden development file: {relative}")
        if _contains_bytes(path, b"/home/nvidia/"):
            errors.append(f"developer absolute path found: {relative}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            data = path.read_bytes()
            if RTSP_CREDENTIALS.search(data):
                errors.append(f"RTSP credentials found: {relative}")
            for match in RUNTIME_PATHS.finditer(data):
                runtime_path = match.group().decode("utf-8")
                if runtime_path.startswith("file://"):
                    runtime_path = runtime_path.removeprefix("file://")
                referenced = root / Path(runtime_path).relative_to(
                    "/opt/deepstream-app-custom"
                )
                if not referenced.exists():
                    errors.append(
                        f"missing runtime path referenced by {relative}: "
                        f"{runtime_path}"
                    )
        with path.open("rb") as stream:
            is_elf = stream.read(4) == b"\x7fELF"
        if is_elf:
            elf_files.append(path)
            sections = _run(["readelf", "-S", str(path)])
            if not development and ".debug_" in sections:
                errors.append(f"embedded debug section found: {relative}")

    if runtime_checks and not errors:
        environment = os.environ.copy()
        environment["LD_LIBRARY_PATH"] = (
            f"{root / 'lib'}:{root / 'gst-plugins'}:"
            f"{environment.get('LD_LIBRARY_PATH', '')}"
        )
        environment["GST_PLUGIN_PATH"] = (
            f"{root / 'gst-plugins'}:"
            "/opt/nvidia/deepstream/deepstream/lib/gst-plugins:"
            f"{environment.get('GST_PLUGIN_PATH', '')}"
        )
        for elf_file in elf_files:
            dependencies = _run(["ldd", str(elf_file)], environment)
            if "not found" in dependencies:
                relative = elf_file.relative_to(root)
                errors.append(
                    f"unresolved dependency in {relative}:\n{dependencies}"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="extracted /opt/deepstream-app-custom")
    parser.add_argument(
        "--runtime-checks",
        action="store_true",
        help="also run ldd on a compatible Jetson",
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="allow debug sections in a development installation",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    if not root.is_dir():
        parser.error(f"not a directory: {root}")
    errors = verify(root, args.runtime_checks, args.development)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(f"release verification passed: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
