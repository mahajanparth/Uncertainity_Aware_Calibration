#!/usr/bin/env python3
"""Build an OASIS CalibrationProblem from Isaac poses and external 2D-3D matches.

This bridges the workflow:
1. Isaac Sim chooses/captures views and stores the camera poses.
2. An external pipeline (for example Kalibr running in Docker) extracts AprilGrid matches.
3. OASIS consumes the Isaac poses plus those measured correspondences.

Expected matches JSON format:
{
  "images": [
    {
      "image_name": "seed_000_pose_0000.png",
      "detected": true,
      "corner_ids": [0, 1, 2],
      "uv": [[612.1, 301.2], [640.8, 301.6], [669.0, 302.0]]
    }
  ]
}

`image_path` may be used instead of `image_name`. Unknown images are ignored.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Iterable

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal_analysis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to interactive_summary.json from Isaac.")
    parser.add_argument(
        "--base-problem",
        required=True,
        help="Path to Isaac's all_candidate_problem.npz. Poses and target geometry are copied from here.",
    )
    parser.add_argument(
        "--matches",
        required=True,
        help="Path to external matches JSON. Records should contain image_name/image_path, corner_ids, and uv.",
    )
    parser.add_argument("--output", required=True, help="Output CalibrationProblem .npz path.")
    parser.add_argument(
        "--pixel-noise-sigma",
        type=float,
        default=None,
        help="Override the recorded pixel noise sigma in the exported problem.",
    )
    parser.add_argument(
        "--retain-unselected-measurements",
        action="store_true",
        help="Keep synthetic measurements for unselected poses instead of clearing them to NaN.",
    )
    parser.add_argument(
        "--retain-selected-synthetic-when-missing",
        action="store_true",
        help="If a selected image has no external match record, keep the base measurement instead of leaving it empty.",
    )
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


def build_image_lookup(summary: dict) -> tuple[list[int], dict[str, dict]]:
    selected_indices: list[int] = []
    lookup: dict[str, dict] = {}
    for capture in iter_selected_captures(summary):
        candidate_index = int(capture["candidate_index"])
        image_path = pathlib.Path(capture["image_path"])
        record = {
            "candidate_index": candidate_index,
            "image_path": str(image_path),
            "image_name": image_path.name,
        }
        selected_indices.append(candidate_index)
        lookup[image_path.name] = record
        lookup[str(image_path)] = record
    return selected_indices, lookup


def iter_match_records(payload: dict | list) -> Iterable[dict]:
    if isinstance(payload, list):
        for record in payload:
            if isinstance(record, dict):
                yield record
        return
    if isinstance(payload, dict):
        images = payload.get("images", [])
        if isinstance(images, list):
            for record in images:
                if isinstance(record, dict):
                    yield record


def record_key(record: dict) -> str | None:
    image_path = record.get("image_path")
    if image_path:
        return str(image_path)
    image_name = record.get("image_name")
    if image_name:
        return pathlib.Path(str(image_name)).name
    return None


def build_measurement_array(record: dict, num_target_points: int) -> tuple[np.ndarray, int]:
    measurements = np.full((num_target_points, 2), np.nan, dtype=float)
    if record.get("detected", True) is False:
        return measurements, 0

    uv = np.asarray(record.get("uv", []), dtype=float).reshape(-1, 2)
    corner_ids = np.asarray(record.get("corner_ids", []), dtype=int).reshape(-1)
    if uv.shape[0] == 0 or corner_ids.shape[0] == 0:
        return measurements, 0

    count = 0
    for corner_id, point in zip(corner_ids, uv):
        corner_idx = int(corner_id)
        if 0 <= corner_idx < num_target_points and np.isfinite(point).all():
            measurements[corner_idx] = point
            count += 1
    return measurements, count


def main() -> None:
    args = parse_args()
    summary_path = pathlib.Path(args.summary).expanduser().resolve()
    base_problem_path = pathlib.Path(args.base_problem).expanduser().resolve()
    matches_path = pathlib.Path(args.matches).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()

    summary = load_json(summary_path)
    matches_payload = load_json(matches_path)
    base_problem = fim.load_calibration_problem_npz(str(base_problem_path))

    selected_indices, image_lookup = build_image_lookup(summary)
    if not selected_indices:
        raise RuntimeError(f"No selected captures found in {summary_path}")

    if args.retain_unselected_measurements:
        measurements = np.asarray(base_problem.measurements, dtype=float).copy()
    else:
        measurements = np.full_like(base_problem.measurements, np.nan, dtype=float)

    matched_candidate_indices: list[int] = []
    unmatched_selected_indices: list[int] = []
    unknown_records: list[str] = []
    used_records = 0
    matched_points_total = 0

    for record in iter_match_records(matches_payload):
        key = record_key(record)
        if key is None:
            continue
        mapped = image_lookup.get(key) or image_lookup.get(pathlib.Path(key).name)
        if mapped is None:
            unknown_records.append(key)
            continue
        candidate_index = int(mapped["candidate_index"])
        record_measurements, num_points = build_measurement_array(record, base_problem.target_points.shape[0])
        measurements[candidate_index] = record_measurements
        matched_candidate_indices.append(candidate_index)
        matched_points_total += num_points
        used_records += 1

    matched_index_set = set(matched_candidate_indices)
    for candidate_index in selected_indices:
        if candidate_index in matched_index_set:
            continue
        unmatched_selected_indices.append(int(candidate_index))
        if args.retain_selected_synthetic_when_missing:
            measurements[candidate_index] = base_problem.measurements[candidate_index]

    pixel_noise_sigma = (
        float(args.pixel_noise_sigma)
        if args.pixel_noise_sigma is not None
        else float(base_problem.pixel_noise_sigma)
    )
    problem = fim.CalibrationProblem(
        target_points=np.asarray(base_problem.target_points, dtype=float),
        candidate_rotations=np.asarray(base_problem.candidate_rotations, dtype=float),
        candidate_translations=np.asarray(base_problem.candidate_translations, dtype=float),
        measurements=measurements,
        intrinsics_gt=np.asarray(base_problem.intrinsics_gt, dtype=float),
        intrinsics_init=np.asarray(base_problem.intrinsics_init, dtype=float),
        image_size=tuple(int(v) for v in base_problem.image_size),
        pixel_noise_sigma=pixel_noise_sigma,
        candidate_board_rotations=(
            None if base_problem.candidate_board_rotations is None else np.asarray(base_problem.candidate_board_rotations, dtype=float)
        ),
        candidate_board_translations=(
            None if base_problem.candidate_board_translations is None else np.asarray(base_problem.candidate_board_translations, dtype=float)
        ),
        camera_is_fixed=bool(base_problem.camera_is_fixed),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fim.save_calibration_problem_npz(str(output_path), problem)

    prior = fim.build_prior_blocks(problem)
    after_report = cal_analysis.evaluate_selection(problem, selected_indices, prior=prior)
    before_after = cal_analysis.before_after_calibration_summary(problem, selected_indices, prior=prior)

    report = {
        "summary_path": str(summary_path),
        "base_problem_path": str(base_problem_path),
        "matches_path": str(matches_path),
        "output_path": str(output_path),
        "selected_indices": [int(idx) for idx in selected_indices],
        "matched_candidate_indices": sorted(int(idx) for idx in matched_index_set),
        "unmatched_selected_indices": unmatched_selected_indices,
        "used_match_records": int(used_records),
        "ignored_unknown_match_records": int(len(unknown_records)),
        "ignored_unknown_record_examples": unknown_records[:10],
        "matched_points_total": int(matched_points_total),
        "num_valid_measurements": int(np.count_nonzero(np.isfinite(measurements).all(axis=2))),
        "pixel_noise_sigma": float(problem.pixel_noise_sigma),
        "retain_unselected_measurements": bool(args.retain_unselected_measurements),
        "retain_selected_synthetic_when_missing": bool(args.retain_selected_synthetic_when_missing),
        "selection_report": after_report,
        "before_after_calibration": before_after,
    }
    report_path = output_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="ascii")

    print(f"Saved CalibrationProblem to: {output_path}")
    print(f"Saved bridge report to: {report_path}")
    print(f"Selected captures: {len(selected_indices)}")
    print(f"Matched selected captures: {len(matched_index_set)}")
    print(f"Total valid matched points: {matched_points_total}")
    print(f"After-selection min_eig: {after_report['min_eig']:.6f}")


if __name__ == "__main__":
    main()
