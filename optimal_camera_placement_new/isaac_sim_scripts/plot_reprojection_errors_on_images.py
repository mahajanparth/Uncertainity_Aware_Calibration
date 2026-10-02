#!/usr/bin/env python3
"""Plot per-point reprojection errors on calibration images.

Typical usage:

python3 isaac_sim_scripts/plot_reprojection_errors_on_images.py \
  --summary isaac_outputs/interactive_run_aprilgrid_opencv_100_noiseless_ransac_hard_subpix/interactive_summary.json \
  --output-dir isaac_outputs/interactive_run_aprilgrid_opencv_100_noiseless_ransac_hard_subpix/reprojection_overlays
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import cv2
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from calibration_board_utils import detect_kalibr_aprilgrid
from detect_checkerboard_uv import detect_charuco
from OASIS import FIM as fim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to interactive_summary.json.")
    parser.add_argument("--output-dir", required=True, help="Directory for overlay images and report JSON.")
    parser.add_argument(
        "--report-key",
        choices=("final_intrinsics_report", "seed_intrinsics_report"),
        default="final_intrinsics_report",
        help="Which intrinsics report inside interactive_summary.json to use.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional limit on the number of images to process.",
    )
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.5,
        help="Only draw numeric error labels for points above this reprojection error in pixels.",
    )
    parser.add_argument(
        "--board-type",
        choices=("auto", "aprilgrid", "charuco"),
        default="auto",
        help="Board detector to use. 'auto' infers it from the summary/report.",
    )
    parser.add_argument("--board-rows", type=int, default=None, help="Optional override for board rows.")
    parser.add_argument("--board-cols", type=int, default=None, help="Optional override for board cols.")
    parser.add_argument("--board-square-size", type=float, default=None, help="Optional override for board square size in meters.")
    parser.add_argument("--board-marker-size", type=float, default=None, help="Optional override for ChArUco/AprilGrid marker size in meters.")
    parser.add_argument("--aruco-dict", default="DICT_4X4_250", help="ArUco dictionary used for ChArUco detection.")
    return parser.parse_args()


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def flatten_history_entries(history: list[dict]) -> list[dict]:
    flat = []
    for item in history:
        if "image_path" in item:
            flat.append(item)
            continue
        captures = item.get("captures")
        if isinstance(captures, list):
            for capture in captures:
                if isinstance(capture, dict) and "image_path" in capture:
                    flat.append(capture)
    return flat


def build_aprilgrid_points(
    rows: int,
    cols: int,
    tag_size: float,
    tag_spacing: float,
) -> np.ndarray:
    pitch = tag_size * (1.0 + tag_spacing)
    points = []
    for grid_row in range(2 * rows):
        for grid_col in range(2 * cols):
            x = (grid_col // 2) * pitch + (grid_col % 2) * tag_size
            y = (grid_row // 2) * pitch + (grid_row % 2) * tag_size
            points.append((x, y, 0.0))
    return np.asarray(points, dtype=np.float64)


def camera_from_intrinsics(intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray(intrinsics, dtype=float).reshape(-1)
    camera_matrix = np.array(
        [
            [intrinsics[0], 0.0, intrinsics[2]],
            [0.0, intrinsics[1], intrinsics[3]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist = np.zeros((5, 1), dtype=np.float64)
    if intrinsics.size > 4:
        tail = intrinsics[4 : min(intrinsics.size, 9)]
        dist[: tail.size, 0] = tail
    return camera_matrix, dist


def error_color(error_px: float, max_error_px: float) -> tuple[int, int, int]:
    ratio = float(np.clip(error_px / max(max_error_px, 1.0e-9), 0.0, 1.0))
    blue = int(round(255.0 * (1.0 - ratio)))
    red = int(round(255.0 * ratio))
    green = int(round(255.0 * (1.0 - abs(ratio - 0.5) * 2.0)))
    return blue, green, red


def infer_aprilgrid_geometry(problem: fim.CalibrationProblem) -> tuple[int, int, float, float]:
    points = np.asarray(problem.target_points, dtype=np.float64)
    xs = np.unique(np.round(points[:, 0], decimals=9))
    ys = np.unique(np.round(points[:, 1], decimals=9))
    if xs.size % 2 != 0 or ys.size % 2 != 0:
        raise RuntimeError("Could not infer AprilGrid rows/cols from target points.")
    cols = int(xs.size // 2)
    rows = int(ys.size // 2)
    x_diffs = np.diff(xs)
    x_diffs = x_diffs[x_diffs > 1.0e-9]
    if x_diffs.size == 0:
        raise RuntimeError("Could not infer AprilGrid spacing from target points.")
    tag_size = float(np.min(x_diffs))
    pitch = float(xs[2] - xs[0]) if xs.size >= 3 else tag_size
    tag_spacing = max(pitch / max(tag_size, 1.0e-12) - 1.0, 0.0)
    return rows, cols, tag_size, tag_spacing


def infer_charuco_geometry(problem: fim.CalibrationProblem) -> tuple[int, int, float]:
    points = np.asarray(problem.target_points, dtype=np.float64)
    xs = np.unique(np.round(points[:, 0], decimals=9))
    ys = np.unique(np.round(points[:, 1], decimals=9))
    if xs.size < 2 or ys.size < 2:
        raise RuntimeError("Could not infer ChArUco geometry from target points.")
    square_size = float(np.min(np.diff(xs)))
    return int(ys.size + 1), int(xs.size + 1), square_size


def detect_aprilgrid_points(
    image_gray: np.ndarray,
    summary: dict,
    problem: fim.CalibrationProblem,
) -> tuple[np.ndarray, np.ndarray]:
    config = summary.get("config", {})
    rows, cols, tag_size, tag_spacing = infer_aprilgrid_geometry(problem)

    result = detect_kalibr_aprilgrid(
        image_gray,
        rows=rows,
        cols=cols,
        min_tags_for_valid_obs=1,
        do_subpix_refinement=True,
        max_subpix_displacement2=float(config.get("aprilgrid_max_subpix_displacement2", 1.5)),
        subpix_window_half_width=int(config.get("aprilgrid_subpix_window", 2)),
        subpix_max_iters=int(config.get("aprilgrid_subpix_max_iters", 30)),
        subpix_epsilon=float(config.get("aprilgrid_subpix_epsilon", 0.1)),
    )
    observed = np.asarray(result["observed"], dtype=bool)
    image_points = np.asarray(result["image_points"], dtype=np.float64)
    object_points = build_aprilgrid_points(rows, cols, tag_size, tag_spacing)
    return object_points[observed], image_points[observed]


def detect_charuco_points(
    image_bgr: np.ndarray,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    aruco_dict: str,
    problem: fim.CalibrationProblem,
) -> tuple[np.ndarray, np.ndarray]:
    found, corners, corner_ids = detect_charuco(
        image_bgr,
        rows=rows,
        cols=cols,
        square_size=square_size,
        marker_size=marker_size,
        aruco_dict_name=aruco_dict,
    )
    if not found or corners is None or corner_ids is None:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    image_points = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    ids = np.asarray(corner_ids, dtype=np.int32).reshape(-1)
    object_points = np.asarray(problem.target_points, dtype=np.float64)[ids]
    return object_points, image_points


def draw_overlay(
    image_bgr: np.ndarray,
    observed_uv: np.ndarray,
    projected_uv: np.ndarray,
    errors_px: np.ndarray,
    label_threshold: float,
) -> np.ndarray:
    annotated = image_bgr.copy()
    max_error = float(np.max(errors_px)) if errors_px.size else 1.0
    for obs, proj, err in zip(observed_uv, projected_uv, errors_px):
        obs_pt = tuple(int(round(v)) for v in obs)
        proj_pt = tuple(int(round(v)) for v in proj)
        color = error_color(float(err), max_error)
        cv2.line(annotated, obs_pt, proj_pt, color, 1, cv2.LINE_AA)
        cv2.circle(annotated, obs_pt, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, proj_pt, 3, (0, 255, 0), 1, cv2.LINE_AA)
        if float(err) >= float(label_threshold):
            cv2.putText(
                annotated,
                f"{float(err):.2f}",
                (obs_pt[0] + 4, obs_pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated


def generate_reprojection_overlays(
    summary_path: str | pathlib.Path,
    output_dir: str | pathlib.Path,
    report_key: str = "final_intrinsics_report",
    max_images: int | None = None,
    label_threshold: float = 0.5,
    board_type: str = "auto",
    board_rows: int | None = None,
    board_cols: int | None = None,
    board_square_size: float | None = None,
    board_marker_size: float | None = None,
    aruco_dict: str = "DICT_4X4_250",
) -> dict:
    summary_path = pathlib.Path(summary_path).expanduser().resolve()
    output_dir = pathlib.Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_json(summary_path)
    report = summary.get(report_key)
    if report is None:
        raise RuntimeError(f"{report_key} was not found in {summary_path}")
    problem_path = summary_path.parent / "all_candidate_problem.npz"
    if not problem_path.exists():
        raise RuntimeError(f"Expected {problem_path} next to the summary so AprilGrid geometry can be inferred.")
    problem = fim.load_calibration_problem_npz(str(problem_path))
    inferred_board_type = board_type
    if inferred_board_type == "auto":
        if "aprilgrid_detector_backend" in report or summary.get("config", {}).get("aprilgrid_detector_backend") is not None:
            inferred_board_type = "aprilgrid"
        else:
            inferred_board_type = "charuco"

    if inferred_board_type == "aprilgrid":
        rows, cols, inferred_square, _tag_spacing = infer_aprilgrid_geometry(problem)
        marker_size = board_marker_size if board_marker_size is not None else inferred_square
        square_size = board_square_size if board_square_size is not None else marker_size
        board_rows = board_rows if board_rows is not None else rows
        board_cols = board_cols if board_cols is not None else cols
    else:
        rows, cols, inferred_square = infer_charuco_geometry(problem)
        board_rows = board_rows if board_rows is not None else rows
        board_cols = board_cols if board_cols is not None else cols
        square_size = board_square_size if board_square_size is not None else inferred_square
        marker_size = board_marker_size if board_marker_size is not None else square_size * 0.76

    intrinsics = np.asarray(report["intrinsics_vector"], dtype=float)
    camera_matrix, dist_coeffs = camera_from_intrinsics(intrinsics)

    history = flatten_history_entries(list(summary.get("history", [])))
    if max_images is not None:
        history = history[: max(0, int(max_images))]

    report_rows = []
    for item in history:
        image_path = pathlib.Path(item["image_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            report_rows.append({"image_path": str(image_path), "used": False, "reason": "image_not_readable"})
            continue
        if inferred_board_type == "aprilgrid":
            image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            object_points, image_points = detect_aprilgrid_points(image_gray, summary, problem)
        else:
            object_points, image_points = detect_charuco_points(
                image_bgr,
                rows=board_rows,
                cols=board_cols,
                square_size=float(square_size),
                marker_size=float(marker_size),
                aruco_dict=aruco_dict,
                problem=problem,
            )
        if object_points.shape[0] < 4:
            report_rows.append(
                {
                    "image_path": str(image_path),
                    "used": False,
                    "reason": "too_few_detected_points",
                    "num_points": int(object_points.shape[0]),
                }
            )
            continue

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            report_rows.append(
                {"image_path": str(image_path), "used": False, "reason": "solvepnp_failed", "num_points": int(object_points.shape[0])}
            )
            continue

        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points, axis=1)
        annotated = draw_overlay(image_bgr, image_points, projected, errors, label_threshold)

        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        rms_error = float(math.sqrt(np.mean(errors * errors)))
        cv2.putText(
            annotated,
            f"mean={mean_error:.3f}px rms={rms_error:.3f}px max={max_error:.3f}px",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), annotated)
        report_rows.append(
            {
                "image_path": str(image_path),
                "overlay_path": str(output_path),
                "used": True,
                "num_points": int(object_points.shape[0]),
                "mean_error_px": mean_error,
                "rms_error_px": rms_error,
                "max_error_px": max_error,
            }
        )

    summary_report = {
        "summary_path": str(summary_path),
        "output_dir": str(output_dir),
        "report_key": report_key,
        "board_type": inferred_board_type,
        "board_rows": int(board_rows),
        "board_cols": int(board_cols),
        "board_square_size": float(square_size),
        "board_marker_size": float(marker_size),
        "intrinsics_vector": intrinsics.tolist(),
        "images": report_rows,
    }
    (output_dir / "reprojection_overlay_report.json").write_text(json.dumps(summary_report, indent=2), encoding="ascii")
    num_used = sum(1 for row in report_rows if row.get("used"))
    print(f"Saved reprojection overlays for {num_used} images to: {output_dir}")
    return summary_report


def main() -> None:
    args = parse_args()
    generate_reprojection_overlays(
        summary_path=args.summary,
        output_dir=args.output_dir,
        report_key=args.report_key,
        max_images=args.max_images,
        label_threshold=args.label_threshold,
        board_type=args.board_type,
        board_rows=args.board_rows,
        board_cols=args.board_cols,
        board_square_size=args.board_square_size,
        board_marker_size=args.board_marker_size,
        aruco_dict=args.aruco_dict,
    )


if __name__ == "__main__":
    main()
