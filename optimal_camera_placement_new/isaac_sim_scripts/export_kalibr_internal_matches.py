#!/usr/bin/env python3
"""Export Kalibr AprilGrid correspondences from a ROS bag to JSON.

This script is meant to run inside a Kalibr/ROS environment, for example:

docker run --rm -v "$PWD":/workspace -w /workspace --entrypoint /bin/bash kalibr:latest -lc '
  set -eo pipefail
  source /catkin_ws/devel/setup.bash 2>/dev/null || source /kalibr_ws/devel/setup.bash 2>/dev/null || source /root/catkin_ws/devel/setup.bash 2>/dev/null || true
  python3 isaac_sim_scripts/export_kalibr_internal_matches.py \
    --summary isaac_outputs/.../interactive_summary.json \
    --bag isaac_outputs/.../cam_april.bag \
    --target isaac_outputs/.../april_6x6.yaml \
    --output isaac_outputs/.../kalibr_matches.json \
    --topic /cam0/image_raw \
    --model pinhole-radtan
'
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to Isaac interactive_summary.json.")
    parser.add_argument("--bag", required=True, help="Path to ROS bag used by Kalibr.")
    parser.add_argument("--target", required=True, help="Kalibr target YAML path.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--topic", required=True, help="ROS image topic inside the bag.")
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "pinhole-radtan",
            "pinhole-equi",
            "pinhole-fov",
            "omni-none",
            "omni-radtan",
            "eucm-none",
            "ds-none",
        ],
        help="Kalibr camera model name.",
    )
    parser.add_argument("--bag-from-to", metavar=("START", "END"), type=float, nargs=2, default=None)
    parser.add_argument("--bag-freq", type=float, default=None)
    return parser.parse_args()


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def iter_selected_captures(summary: dict) -> Iterable[dict]:
    for entry in summary.get("history", []):
        if entry.get("step_type") == "seed":
            yield entry
            continue
        if entry.get("step_type") == "recommended_batch":
            for capture in entry.get("captures", []):
                yield capture


def collect_summary_images(summary: dict) -> list[dict]:
    images: list[dict] = []
    for order, capture in enumerate(iter_selected_captures(summary)):
        image_path = pathlib.Path(capture["image_path"])
        images.append(
            {
                "summary_order": int(order),
                "selection_order": int(capture.get("selection_order", order)),
                "candidate_index": int(capture["candidate_index"]),
                "image_name": image_path.name,
                "image_path": str(image_path),
            }
        )
    return images


def main() -> None:
    args = parse_args()
    summary_path = pathlib.Path(args.summary).expanduser().resolve()
    bag_path = pathlib.Path(args.bag).expanduser().resolve()
    target_path = pathlib.Path(args.target).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()

    summary = load_json(summary_path)
    summary_images = collect_summary_images(summary)
    if not summary_images:
        raise RuntimeError(f"No selected images found in {summary_path}")

    import aslam_cv_backend as acvb
    import kalibr_common as kc
    import kalibr_camera_calibration as kcc

    camera_models = {
        "pinhole-radtan": acvb.DistortedPinhole,
        "pinhole-equi": acvb.EquidistantPinhole,
        "pinhole-fov": acvb.FovPinhole,
        "omni-none": acvb.Omni,
        "omni-radtan": acvb.DistortedOmni,
        "eucm-none": acvb.ExtendedUnified,
        "ds-none": acvb.DoubleSphere,
    }

    target_config = kc.CalibrationTargetParameters(str(target_path))
    dataset = kc.BagImageDatasetReader(str(bag_path), args.topic, bag_from_to=args.bag_from_to, bag_freq=args.bag_freq)
    camera = kcc.CameraGeometry(camera_models[args.model], target_config, dataset, verbose=False)

    num_images = dataset.numImages()
    if num_images != len(summary_images):
        raise RuntimeError(
            f"Bag image count ({num_images}) does not match summary-selected image count ({len(summary_images)})."
        )

    records: list[dict] = []
    detected_count = 0
    total_points = 0

    for idx, ((timestamp, image), image_meta) in enumerate(zip(dataset.readDataset(), summary_images)):
        success, obs = camera.ctarget.detector.findTargetNoTransformation(timestamp, np.array(image))
        record = dict(image_meta)
        record["bag_index"] = int(idx)
        record["timestamp_ns"] = int(timestamp.toSec() * 1e9)

        if success:
            corner_ids = np.asarray(obs.getCornersIdx(), dtype=int).reshape(-1)
            uv = np.asarray(obs.getCornersImageFrame(), dtype=float).reshape(-1, 2)
            xyz = np.asarray(obs.getCornersTargetFrame(), dtype=float).reshape(-1, 3)
            record["detected"] = True
            record["corner_ids"] = corner_ids.tolist()
            record["uv"] = uv.tolist()
            record["xyz_target"] = xyz.tolist()
            record["num_points"] = int(corner_ids.shape[0])
            detected_count += 1
            total_points += int(corner_ids.shape[0])
        else:
            record["detected"] = False
            record["corner_ids"] = []
            record["uv"] = []
            record["xyz_target"] = []
            record["num_points"] = 0
        records.append(record)

    payload = {
        "source": "kalibr_internal_detector",
        "summary_path": str(summary_path),
        "bag_path": str(bag_path),
        "target_path": str(target_path),
        "topic": args.topic,
        "model": args.model,
        "num_images": len(records),
        "num_detected_images": int(detected_count),
        "total_detected_points": int(total_points),
        "images": records,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="ascii")

    print(f"Saved Kalibr internal matches to: {output_path}")
    print(f"Images processed: {len(records)}")
    print(f"Images with detections: {detected_count}")
    print(f"Total detected points: {total_points}")


if __name__ == "__main__":
    main()
