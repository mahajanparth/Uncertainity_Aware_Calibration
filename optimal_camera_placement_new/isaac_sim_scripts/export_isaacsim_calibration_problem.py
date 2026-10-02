#!/usr/bin/env python3
"""Convert an Isaac Sim capture dataset into a CalibrationProblem .npz file."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from OASIS.FIM import CalibrationProblem, default_intrinsics_init_offset, project_points, save_calibration_problem_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="Path to dataset_metadata.json from Isaac Sim capture.")
    parser.add_argument("--detections", help="Optional path to detections.json from detect_checkerboard_uv.py.")
    parser.add_argument("--output", required=True, help="Output .npz file for the CalibrationProblem.")
    parser.add_argument(
        "--pixel-noise-sigma",
        type=float,
        default=1.0,
        help="Noise level recorded in the exported CalibrationProblem.",
    )
    parser.add_argument(
        "--init-offset",
        type=float,
        nargs="+",
        default=None,
        help="Perturbation added to the ground-truth intrinsics to form intrinsics_init.",
    )
    parser.add_argument(
        "--use-detections",
        action="store_true",
        help="Use detections.json to build sparse measurements instead of exact geometric projections.",
    )
    return parser.parse_args()


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def augment_intrinsics_with_zero_distortion(intrinsics: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=float).reshape(-1)
    if intrinsics.size >= 9:
        return intrinsics
    if intrinsics.size < 4:
        raise ValueError("Metadata intrinsics must contain at least [fx, fy, cx, cy].")
    return np.concatenate([intrinsics[:4], np.zeros(5, dtype=float)])


def frame_sort_key(frame: dict) -> tuple[int, str]:
    return int(frame.get("frame_index", 0)), str(frame.get("image_relative_path", ""))


def _legacy_usd_rotation_to_calibration_rotation(rotation_usd: np.ndarray) -> np.ndarray:
    return rotation_usd @ np.diag([1.0, -1.0, -1.0])


def frame_rotation_wc(frame: dict) -> np.ndarray:
    if "rotation_wc" in frame:
        return np.asarray(frame["rotation_wc"], dtype=float)
    if "rotation_wc_usd" in frame:
        return _legacy_usd_rotation_to_calibration_rotation(np.asarray(frame["rotation_wc_usd"], dtype=float))
    raise KeyError("Frame metadata is missing both rotation_wc and rotation_wc_usd.")


def build_measurements_from_metadata(metadata: dict, target_points: np.ndarray) -> np.ndarray:
    intrinsics = augment_intrinsics_with_zero_distortion(np.asarray(metadata["camera"]["intrinsics_px"], dtype=float))
    image_size = tuple(int(v) for v in metadata["image_size"])
    width, height = image_size
    frames = sorted(metadata["frames"], key=frame_sort_key)
    measurements = np.full((len(frames), target_points.shape[0], 2), np.nan, dtype=float)
    for idx, frame in enumerate(frames):
        rotation_wc = frame_rotation_wc(frame)
        translation_wc = np.asarray(frame["translation_wc"], dtype=float)
        projected = project_points(target_points, rotation_wc, translation_wc, intrinsics)
        valid = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        projected[~valid] = np.nan
        measurements[idx] = projected
    return measurements


def build_measurements_from_detections(metadata: dict, detections: dict, target_points: np.ndarray) -> np.ndarray:
    frames = sorted(metadata["frames"], key=frame_sort_key)
    frame_name_to_index = {
        pathlib.Path(frame["image_relative_path"]).name: idx
        for idx, frame in enumerate(frames)
    }
    measurements = np.full((len(frames), target_points.shape[0], 2), np.nan, dtype=float)
    for record in detections.get("images", []):
        image_name = pathlib.Path(record.get("image_name", "")).name
        if image_name not in frame_name_to_index or not record.get("detected", False):
            continue
        frame_idx = frame_name_to_index[image_name]
        uv = np.asarray(record.get("uv", []), dtype=float).reshape(-1, 2)
        if uv.size == 0:
            continue
        corner_ids = np.asarray(record.get("corner_ids", list(range(uv.shape[0]))), dtype=int).reshape(-1)
        for corner_id, point in zip(corner_ids, uv):
            if 0 <= int(corner_id) < target_points.shape[0]:
                measurements[frame_idx, int(corner_id)] = point
    return measurements


def main() -> None:
    args = parse_args()
    metadata_path = pathlib.Path(args.metadata).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()
    detections_path = pathlib.Path(args.detections).expanduser().resolve() if args.detections else None

    metadata = load_json(metadata_path)
    detections = load_json(detections_path) if detections_path else None

    target_points = np.asarray(metadata["board"]["corner_points_world"], dtype=float)
    frames = sorted(metadata["frames"], key=frame_sort_key)
    candidate_rotations = np.asarray([frame_rotation_wc(frame) for frame in frames], dtype=float)
    candidate_translations = np.asarray([frame["translation_wc"] for frame in frames], dtype=float)
    intrinsics_gt = augment_intrinsics_with_zero_distortion(np.asarray(metadata["camera"]["intrinsics_px"], dtype=float))
    if args.init_offset is None:
        init_offset = default_intrinsics_init_offset(intrinsics_gt.size)
    else:
        init_offset = np.zeros(intrinsics_gt.size, dtype=float)
        raw = np.asarray(args.init_offset, dtype=float).reshape(-1)
        init_offset[: min(init_offset.size, raw.size)] = raw[: min(init_offset.size, raw.size)]
    intrinsics_init = intrinsics_gt + init_offset
    image_size = tuple(int(v) for v in metadata["image_size"])

    if args.use_detections:
        if detections is None:
            raise ValueError("--use-detections was set but no --detections file was provided.")
        measurements = build_measurements_from_detections(metadata, detections, target_points)
    else:
        measurements = build_measurements_from_metadata(metadata, target_points)

    problem = CalibrationProblem(
        target_points=target_points,
        candidate_rotations=candidate_rotations,
        candidate_translations=candidate_translations,
        measurements=measurements,
        intrinsics_gt=intrinsics_gt,
        intrinsics_init=intrinsics_init,
        image_size=image_size,
        pixel_noise_sigma=float(args.pixel_noise_sigma),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_calibration_problem_npz(str(output_path), problem)

    summary = {
        "metadata_path": str(metadata_path),
        "detections_path": str(detections_path) if detections_path else None,
        "output_path": str(output_path),
        "num_frames": len(frames),
        "num_target_points": int(target_points.shape[0]),
        "num_valid_measurements": int(np.count_nonzero(np.isfinite(measurements).all(axis=2))),
        "used_detections": bool(args.use_detections),
        "image_size": list(image_size),
        "intrinsics_gt": intrinsics_gt.tolist(),
        "intrinsics_init": intrinsics_init.tolist(),
    }
    summary_path = output_path.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    print(f"Saved CalibrationProblem to: {output_path}")
    print(f"Saved export summary to: {summary_path}")


if __name__ == "__main__":
    main()
