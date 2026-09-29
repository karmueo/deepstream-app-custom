#!/usr/bin/env python3
"""Compare the port against the original C++ program on identical BGR images."""

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np


def run_json(args: list[str]) -> dict:
    result = subprocess.run(args, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax1 + aw, bx1 + bw), min(ay1 + ah, by1 + bh)
    overlap = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - overlap
    return overlap / union if union > 0 else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--reference-bin", type=Path, required=True)
    parser.add_argument("--reference-engines", type=Path, required=True)
    parser.add_argument("--reference-workspace", type=Path, required=True)
    parser.add_argument("--port-bin", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("images", nargs="+", type=Path)
    args = parser.parse_args()

    report = []
    for image in args.images:
        reference = run_json([
            str(args.reference_bin), "--headless", "--workspace",
            str(args.reference_workspace), "--engines", str(args.reference_engines),
            "--image", str(image),
        ])
        actual = run_json([str(args.port_bin), "--config", str(args.config), str(image)])
        if actual["model_id"] != reference["model_id"]:
            raise AssertionError("model ID differs")
        expected_faces = reference["faces"]
        actual_faces = actual["faces"]
        if len(actual_faces) != len(expected_faces):
            raise AssertionError(f"{image}: face count differs")
        unused = set(range(len(expected_faces)))
        results = []
        for face in actual_faces:
            index = max(unused, key=lambda i: box_iou(face["bbox"], expected_faces[i]["bbox"]))
            unused.remove(index)
            expected = expected_faces[index]
            bbox_error = float(np.max(np.abs(np.array(face["bbox"]) - expected["bbox"])))
            landmark_error = float(np.max(np.abs(
                np.array(face["landmarks"]) - expected["landmarks"])))
            embedding = np.array(face["embedding"], dtype=np.float64)
            reference_embedding = np.array(expected["embedding"], dtype=np.float64)
            similarity = float(np.dot(embedding, reference_embedding))
            results.append({
                "bbox_max_abs_error": bbox_error,
                "landmark_max_abs_error": landmark_error,
                "embedding_cosine": similarity,
            })
            if bbox_error > 1 or landmark_error > 1 or similarity <= 0.99:
                raise AssertionError(f"{image}: port differs from original: {results[-1]}")
        report.append({"image": str(image), "faces": len(actual_faces), "comparisons": results})
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(f"Compared {len(report)} images and {sum(row['faces'] for row in report)} faces")


if __name__ == "__main__":
    main()
