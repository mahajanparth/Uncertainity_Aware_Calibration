#!/usr/bin/env python3
"""Detect a Kalibr-style AprilGrid in one image and save a corner overlay."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from calibration_board_utils import detect_geometric_aprilgrid, detect_kalibr_aprilgrid, detect_true_apriltag_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--cols", type=int, required=True)
    parser.add_argument("--backend", choices=["opencv", "geometric", "apriltag"], default="opencv")
    parser.add_argument("--tag-size", type=float, default=0.088)
    parser.add_argument("--tag-spacing", type=float, default=0.3)
    parser.add_argument("--board-size", type=float, default=None)
    parser.add_argument("--json-output", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = pathlib.Path(args.image)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    if args.backend == "geometric":
        result = detect_geometric_aprilgrid(
            gray,
            rows=args.rows,
            cols=args.cols,
            tag_size=args.tag_size,
            tag_spacing=args.tag_spacing,
            board_width=args.board_size,
            board_height=args.board_size,
        )
    elif args.backend == "apriltag":
        result = detect_true_apriltag_grid(
            gray,
            rows=args.rows,
            cols=args.cols,
        )
    else:
        result = detect_kalibr_aprilgrid(gray, rows=args.rows, cols=args.cols)

    annotated = image_bgr.copy()
    for detection in result["detections"]:
        pts = np.round(detection["corners"]).astype(int)
        cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
        for (x, y), point_idx in zip(pts, detection["point_indices"]):
            cv2.circle(annotated, (int(x), int(y)), 5, (0, 0, 255), thickness=-1)
            cv2.putText(
                annotated,
                str(point_idx),
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )
        center = pts.mean(axis=0).astype(int)
        cv2.putText(
            annotated,
            f"id={int(detection['id'])}",
            (int(center[0]) - 20, int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)

    summary = {
        "success": bool(result["success"]),
        "num_markers": int(result["num_markers"]),
        "num_observed_corners": int(np.count_nonzero(result["observed"])),
        "ids": [int(detection["id"]) for detection in result["detections"]],
        "backend": args.backend,
    }
    print(json.dumps(summary, indent=2))

    if args.json_output:
        json_path = pathlib.Path(args.json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(summary, indent=2), encoding="ascii")


if __name__ == "__main__":
    main()
