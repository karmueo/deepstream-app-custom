#!/usr/bin/env python3
"""Compare both ported engines on the same face crops as C++ and ONNX FP32."""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


MODELS = ("enet_b0_8_best_vgaf", "mbf_va_mtl")


def run_json(command: list[str]) -> dict:
    return json.loads(subprocess.run(command, check=True, capture_output=True, text=True).stdout)


def read_frames(images: list[Path], video: Path | None, limit: int) -> list[np.ndarray]:
    frames = []
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {path}")
        frames.append(frame)
    if video:
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for index in np.linspace(0, count - 1, min(limit, count), dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if ok:
                frames.append(frame)
        cap.release()
    if not frames:
        raise RuntimeError("Provide --image or --video")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser()
    plugin = Path(__file__).resolve().parents[1]
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--reference-bin", type=Path, required=True)
    parser.add_argument("--reference-face-engines", type=Path,
                        default=Path.home() / ".local/share/jetson-face-recognition/engines")
    parser.add_argument("--reference-emotion-engines", type=Path,
                        default=Path.home() / ".local/share/jetson-video-emotion/engines")
    parser.add_argument("--port-bin", type=Path, default=plugin / "build/emotieff-image-check")
    parser.add_argument("--report", type=Path, default=plugin / "build/reference-report.json")
    args = parser.parse_args()
    frames = read_frames(args.image, args.video, args.limit)
    report = {}
    with tempfile.TemporaryDirectory(prefix="emotieff-compare-") as temporary:
        work = Path(temporary)
        for model in MODELS:
            side = 112 if model == "mbf_va_mtl" else 224
            mean = np.array([.5] * 3 if side == 112 else [.485, .456, .406])
            stddev = np.array([.5] * 3 if side == 112 else [.229, .224, .225])
            session = ort.InferenceSession(str(plugin / "model" / f"{model}.onnx"),
                                           providers=["CPUExecutionProvider"])
            config = work / "config.yml"
            config.write_text("property:\n"
                              f"  emotion-model: {model}\n"
                              f"  emotion-engine: {plugin / 'model' / (model + '.engine')}\n"
                              "  interval: 0\n")
            differences = []
            disagreements = []
            for frame_index, frame in enumerate(frames):
                image_path = work / "frame.png"
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError("Cannot write temporary frame")
                reference_faces = run_json([
                    str(args.reference_bin), "--headless", "--image", str(image_path),
                    "--face-engines", str(args.reference_face_engines),
                    "--emotion-model", model,
                    "--emotion-engine", str(args.reference_emotion_engines / f"{model}.engine"),
                ])["faces"]
                for face in reference_faces:
                    x, y, width, height = map(int, face["crop"])
                    if width <= 0 or height <= 0:
                        continue
                    face_crop = frame[y:y + height, x:x + width]
                    crop_path = work / "crop.png"
                    if not cv2.imwrite(str(crop_path), face_crop):
                        raise RuntimeError("Cannot write temporary crop")
                    actual = run_json([str(args.port_bin), "--config", str(config), str(crop_path)])
                    reference_result = run_json([
                        str(args.reference_bin), "--headless", "--crop", str(crop_path),
                        "--emotion-model", model,
                        "--emotion-engine", str(args.reference_emotion_engines / f"{model}.engine"),
                    ])
                    rgb = cv2.cvtColor(cv2.resize(face_crop, (side, side)), cv2.COLOR_BGR2RGB)
                    blob = ((rgb / 255. - mean) / stddev).transpose(2, 0, 1).astype("float32")[None]
                    logits = session.run(None, {session.get_inputs()[0].name: blob})[0][0]
                    expected = np.exp(logits[:8] - np.max(logits[:8]))
                    expected /= expected.sum()
                    actual_prob = np.asarray(actual["probabilities"])
                    reference_prob = np.asarray(reference_result["probabilities"])
                    error = max(float(np.max(np.abs(actual_prob - expected))),
                                float(np.max(np.abs(actual_prob - reference_prob))))
                    differences.append(error)
                    if (int(np.argmax(actual_prob)) != int(np.argmax(expected)) and
                            float(np.sort(expected)[-1] - np.sort(expected)[-2]) > .04):
                        disagreements.append({"frame": frame_index, "crop": face["crop"],
                                              "probability_error": error})
            if not differences:
                raise AssertionError(f"No valid face crops found for {model}")
            report[model] = {"face_crops": len(differences),
                             "max_probability_error": max(differences),
                             "confident_disagreements": disagreements,
                             "passed": max(differences) <= .02 and not disagreements}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not all(row["passed"] for row in report.values()):
        raise AssertionError("EmotiEff port differs from reference")


if __name__ == "__main__":
    main()
