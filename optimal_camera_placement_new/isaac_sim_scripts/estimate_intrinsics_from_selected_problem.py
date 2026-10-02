#!/usr/bin/env python3
"""Estimate intrinsics from a selected subset of matched calibration views.

Typical usage:

python3 isaac_sim_scripts/estimate_intrinsics_from_selected_problem.py \
  --problem isaac_outputs/.../kalibr_matched_problem.npz \
  --selection-summary results/isaacsim_problem_YYYYMMDD_HHMMSS/summary.json \
  --kalibr-camchain isaac_outputs/.../cam_april-camchain.yaml \
  --output isaac_outputs/.../oasis_selected_intrinsics.json
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime
import json
import pathlib
import sys

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from OASIS import FIM as fim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problem", required=True, help="Path to a CalibrationProblem .npz file.")
    parser.add_argument(
        "--selection-summary",
        required=True,
        help="Path to an OASIS summary.json containing selected_indices.",
    )
    parser.add_argument(
        "--seed-selection-summary",
        default=None,
        help="Optional summary.json whose selected_indices will be used to estimate a seed intrinsics_init before the final fit.",
    )
    parser.add_argument(
        "--kalibr-camchain",
        default=None,
        help="Optional Kalibr camchain YAML for comparison.",
    )
    parser.add_argument(
        "--solver-backend",
        choices=("opencv", "gtsam"),
        default="opencv",
        help="Backend used for the post-hoc intrinsic fit.",
    )
    parser.add_argument("--output", required=True, help="Output JSON report path.")
    parser.add_argument(
        "--ransac-filter-correspondences",
        action="store_true",
        help="Apply planar homography RANSAC before calibration.",
    )
    parser.add_argument("--ransac-reproj-threshold", type=float, default=3.0)
    parser.add_argument("--ransac-min-inliers", type=int, default=8)
    return parser.parse_args()


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def load_selected_indices(path: pathlib.Path) -> list[int]:
    payload = load_json(path)
    indices = payload.get("selected_indices", [])
    return [int(idx) for idx in indices]


def filter_planar_correspondences_ransac(
    object_points: np.ndarray,
    image_points: np.ndarray,
    reproj_threshold: float,
    min_inliers: int,
) -> dict:
    if object_points.shape[0] < 4 or image_points.shape[0] < 4:
        return {"used": False, "reason": "too_few_points"}

    planar_xy = np.asarray(object_points[:, :2], dtype=np.float32)
    image_xy = np.asarray(image_points, dtype=np.float32)
    homography, mask = cv2.findHomography(planar_xy, image_xy, cv2.RANSAC, ransacReprojThreshold=float(reproj_threshold))
    if homography is None or mask is None:
        return {"used": False, "reason": "homography_failed"}

    mask = np.asarray(mask, dtype=bool).reshape(-1)
    inliers = int(np.count_nonzero(mask))
    if inliers < int(min_inliers):
        return {"used": False, "reason": "too_few_inliers", "mask": mask.tolist(), "num_inliers": inliers}
    return {"used": True, "mask": mask.tolist(), "num_inliers": inliers}


def estimate_intrinsics_from_selection(
    problem: fim.CalibrationProblem,
    selected_indices: list[int],
    use_ransac_filter: bool = False,
    ransac_reproj_threshold: float = 3.0,
    ransac_min_inliers: int = 8,
) -> dict:
    object_points = []
    image_points = []
    view_reports = []
    planar_points = np.asarray(problem.target_points, dtype=np.float32).copy()
    planar_points[:, 2] = 0.0

    for idx in selected_indices:
        measurements = np.asarray(problem.measurements[idx], dtype=np.float32)
        valid = np.isfinite(measurements).all(axis=1)
        if np.count_nonzero(valid) < 4:
            view_reports.append(
                {"selection_index": int(idx), "used": False, "reason": "too_few_valid_measurements", "num_points": int(np.count_nonzero(valid))}
            )
            continue
        object_view = planar_points[valid]
        image_view = measurements[valid]
        ransac_report = None
        if use_ransac_filter:
            ransac_report = filter_planar_correspondences_ransac(
                object_view,
                image_view,
                reproj_threshold=ransac_reproj_threshold,
                min_inliers=ransac_min_inliers,
            )
            if ransac_report["used"]:
                inlier_mask = np.asarray(ransac_report["mask"], dtype=bool)
                object_view = object_view[inlier_mask]
                image_view = image_view[inlier_mask]
        if object_view.shape[0] < 4:
            view_reports.append(
                {
                    "selection_index": int(idx),
                    "used": False,
                    "reason": "too_few_points_after_ransac" if use_ransac_filter else "too_few_valid_measurements",
                    "num_points": int(object_view.shape[0]),
                    "ransac_report": ransac_report,
                }
            )
            continue
        object_points.append(object_view.reshape(-1, 1, 3))
        image_points.append(image_view.reshape(-1, 1, 2))
        view_reports.append(
            {
                "selection_index": int(idx),
                "used": True,
                "num_points": int(object_view.shape[0]),
                "ransac_report": ransac_report,
            }
        )

    if not object_points:
        raise RuntimeError("No valid selected views were available for calibration.")

    fx, fy, cx, cy, dist_guess = fim.split_intrinsics(problem.intrinsics_init)
    camera_matrix_init = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs_init = np.asarray(dist_guess, dtype=np.float64).reshape(-1, 1)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    (
        retval,
        camera_matrix,
        dist_coeffs,
        _rvecs,
        _tvecs,
        std_deviations_intrinsics,
        _std_deviations_extrinsics,
        _per_view_errors,
    ) = cv2.calibrateCameraExtended(
        object_points,
        image_points,
        tuple(int(v) for v in problem.image_size),
        camera_matrix_init,
        dist_coeffs_init,
        flags=flags,
    )

    dist_vector = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    if dist_vector.size < 5:
        dist_vector = np.pad(dist_vector, (0, 5 - dist_vector.size))
    std_intrinsics = np.asarray(std_deviations_intrinsics, dtype=float).reshape(-1)
    std_intrinsics_named = np.zeros(9, dtype=float)
    if std_intrinsics.size >= 4:
        std_intrinsics_named[:4] = std_intrinsics[:4]
    # OpenCV returns [fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6, ...]
    if std_intrinsics.size >= 9:
        std_intrinsics_named[4:] = std_intrinsics[4:9]
    elif std_intrinsics.size > 4:
        tail = min(5, std_intrinsics.size - 4)
        std_intrinsics_named[4 : 4 + tail] = std_intrinsics[4 : 4 + tail]
    covariance_diag = std_intrinsics_named * std_intrinsics_named
    intrinsics = np.array(
        [camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2], *dist_vector[:5]],
        dtype=float,
    )
    return {
        "solver_backend": "opencv",
        "num_views_requested": len(selected_indices),
        "num_views_used": len(object_points),
        "rms_reprojection_error": float(retval),
        "camera_matrix": camera_matrix.astype(float).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist(),
        "intrinsics_vector": intrinsics.tolist(),
        "intrinsics_gt": np.asarray(problem.intrinsics_gt, dtype=float).tolist(),
        "intrinsics_init": np.asarray(problem.intrinsics_init, dtype=float).tolist(),
        "intrinsics_error_vs_gt": (intrinsics - np.asarray(problem.intrinsics_gt, dtype=float)).tolist(),
        "parameter_labels": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"],
        "std_dev_intrinsics": std_intrinsics_named.tolist(),
        "covariance_diag_intrinsics": covariance_diag.tolist(),
        "ransac_filter_used": bool(use_ransac_filter),
        "ransac_reproj_threshold": float(ransac_reproj_threshold),
        "ransac_min_inliers": int(ransac_min_inliers),
        "view_reports": view_reports,
    }


def estimate_intrinsics_from_selection_gtsam(
    problem: fim.CalibrationProblem,
    selected_indices: list[int],
    use_ransac_filter: bool = False,
    ransac_reproj_threshold: float = 3.0,
    ransac_min_inliers: int = 8,
) -> dict:
    import gtsam

    X = gtsam.symbol_shorthand.X
    K = gtsam.symbol_shorthand.K

    graph = gtsam.NonlinearFactorGraph()
    initial = gtsam.Values()
    view_reports = []
    used_view_indices: list[int] = []

    planar_points = np.asarray(problem.target_points, dtype=np.float64).copy()
    planar_points[:, 2] = 0.0

    intr = np.asarray(problem.intrinsics_init, dtype=float).reshape(-1)
    k1 = float(intr[4]) if intr.size > 4 else 0.0
    k2 = float(intr[5]) if intr.size > 5 else 0.0
    p1 = float(intr[6]) if intr.size > 6 else 0.0
    p2 = float(intr[7]) if intr.size > 7 else 0.0
    k3_fixed = float(intr[8]) if intr.size > 8 else 0.0
    calib_init = gtsam.Cal3DS2(float(intr[0]), float(intr[1]), 0.0, float(intr[2]), float(intr[3]), k1, k2, p1, p2)
    calib_key = K(0)
    initial.insert(calib_key, calib_init)

    sigma_px = max(float(problem.pixel_noise_sigma), 1.0e-9)
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma_px)
    # Keep skew pinned to zero. Cal3DS2 does not include k3, so we optimize the default 8+skew layout only.
    calib_prior_sigma = np.array([30.0, 30.0, 1.0e-9, 20.0, 20.0, 0.25, 0.25, 0.25, 0.25], dtype=float)
    graph.push_back(gtsam.PriorFactorCal3DS2(calib_key, calib_init, gtsam.noiseModel.Diagonal.Sigmas(calib_prior_sigma)))

    camera_matrix_init = np.array([[intr[0], 0.0, intr[2]], [0.0, intr[1], intr[3]], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs_init = np.asarray([k1, k2, p1, p2, 0.0], dtype=np.float64).reshape(-1, 1)

    for idx in selected_indices:
        measurements = np.asarray(problem.measurements[idx], dtype=np.float64)
        valid = np.isfinite(measurements).all(axis=1)
        if np.count_nonzero(valid) < 4:
            view_reports.append(
                {"selection_index": int(idx), "used": False, "reason": "too_few_valid_measurements", "num_points": int(np.count_nonzero(valid))}
            )
            continue
        object_view = planar_points[valid]
        image_view = measurements[valid]
        corner_ids = np.flatnonzero(valid)
        ransac_report = None
        if use_ransac_filter:
            ransac_report = filter_planar_correspondences_ransac(
                object_view.astype(np.float32),
                image_view.astype(np.float32),
                reproj_threshold=ransac_reproj_threshold,
                min_inliers=ransac_min_inliers,
            )
            if ransac_report["used"]:
                inlier_mask = np.asarray(ransac_report["mask"], dtype=bool)
                object_view = object_view[inlier_mask]
                image_view = image_view[inlier_mask]
                corner_ids = corner_ids[inlier_mask]
        if object_view.shape[0] < 4:
            view_reports.append(
                {
                    "selection_index": int(idx),
                    "used": False,
                    "reason": "too_few_points_after_ransac" if use_ransac_filter else "too_few_valid_measurements",
                    "num_points": int(object_view.shape[0]),
                    "ransac_report": ransac_report,
                }
            )
            continue

        pose_key = X(int(idx))
        pose_init = None
        success, rvec, tvec = cv2.solvePnP(
            object_view.astype(np.float64),
            image_view.astype(np.float64),
            camera_matrix_init,
            dist_coeffs_init,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if success:
            rot_cw, _ = cv2.Rodrigues(rvec)
            t_cw = np.asarray(tvec, dtype=float).reshape(3)
            rot_wc = rot_cw.T
            trans_wc = -(rot_cw.T @ t_cw)
            pose_init = gtsam.Pose3(gtsam.Rot3(rot_wc), trans_wc)
        else:
            pose_init = gtsam.Pose3(gtsam.Rot3(np.asarray(problem.candidate_rotations[idx], dtype=float)), np.asarray(problem.candidate_translations[idx], dtype=float))
        initial.insert(pose_key, pose_init)

        for point_w, uv in zip(object_view, image_view):
            point_fixed = np.asarray(point_w, dtype=np.float64).reshape(3)
            measurement = np.asarray(uv, dtype=np.float64).reshape(2)
            keys = gtsam.KeyVector()
            keys.append(pose_key)
            keys.append(calib_key)

            def make_error_func(point_local: np.ndarray, measurement_local: np.ndarray):
                def error_func(this: gtsam.CustomFactor, values: gtsam.Values, jacobians: list[np.ndarray] | None) -> np.ndarray:
                    pose_val = values.atPose3(this.keys()[0])
                    calib_val = values.atCal3DS2(this.keys()[1])
                    cam = gtsam.PinholeCameraCal3DS2(pose_val, calib_val)
                    if jacobians is not None:
                        dpose = np.zeros((2, 6), dtype=float, order="F")
                        dpoint = np.zeros((2, 3), dtype=float, order="F")
                        dcal = np.zeros((2, 9), dtype=float, order="F")
                        pred = np.asarray(cam.project(point_local, dpose, dpoint, dcal), dtype=float).reshape(2)
                        jacobians[0] = dpose
                        jacobians[1] = dcal
                    else:
                        pred = np.asarray(cam.project(point_local), dtype=float).reshape(2)
                    return pred - measurement_local

                return error_func

            graph.push_back(gtsam.CustomFactor(measurement_noise, keys, make_error_func(point_fixed, measurement)))

        used_view_indices.append(int(idx))
        view_reports.append(
            {
                "selection_index": int(idx),
                "used": True,
                "num_points": int(object_view.shape[0]),
                "ransac_report": ransac_report,
            }
        )

    if not used_view_indices:
        raise RuntimeError("No valid selected views were available for GTSAM calibration.")

    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()

    calib = result.atCal3DS2(calib_key)
    calib_vec = np.asarray(calib.vector(), dtype=float).reshape(-1)
    intrinsics = np.array(
        [
            calib_vec[0],
            calib_vec[1],
            calib_vec[3],
            calib_vec[4],
            calib_vec[5],
            calib_vec[6],
            calib_vec[7],
            calib_vec[8],
            k3_fixed,
        ],
        dtype=float,
    )

    # Compute RMS reprojection error on the used correspondences with the optimized state.
    squared_error_sum = 0.0
    residual_count = 0
    for idx in used_view_indices:
        pose = result.atPose3(X(int(idx)))
        rot = pose.rotation().matrix()
        trans = np.asarray(pose.translation(), dtype=float)
        measurements = np.asarray(problem.measurements[idx], dtype=np.float64)
        valid = np.isfinite(measurements).all(axis=1)
        object_view = planar_points[valid]
        image_view = measurements[valid]
        if use_ransac_filter:
            ransac_report = filter_planar_correspondences_ransac(
                object_view.astype(np.float32),
                image_view.astype(np.float32),
                reproj_threshold=ransac_reproj_threshold,
                min_inliers=ransac_min_inliers,
            )
            if ransac_report["used"]:
                inlier_mask = np.asarray(ransac_report["mask"], dtype=bool)
                object_view = object_view[inlier_mask]
                image_view = image_view[inlier_mask]
        reproj = fim.project_points(object_view, rot, trans, intrinsics)
        residuals = reproj - image_view
        squared_error_sum += float(np.sum(residuals * residuals))
        residual_count += int(residuals.size)
    rms = float(np.sqrt(squared_error_sum / max(residual_count, 1)))

    covariance_diag = np.full(9, np.nan, dtype=float)
    std_intrinsics = np.full(9, np.nan, dtype=float)
    covariance_note = None
    try:
        marginals = gtsam.Marginals(graph, result)
        calib_cov = np.asarray(marginals.marginalCovariance(calib_key), dtype=float)
        std_calib = np.sqrt(np.maximum(np.diag(calib_cov), 0.0))
        covariance_diag[:8] = np.array(
            [
                calib_cov[0, 0],
                calib_cov[1, 1],
                calib_cov[3, 3],
                calib_cov[4, 4],
                calib_cov[5, 5],
                calib_cov[6, 6],
                calib_cov[7, 7],
                calib_cov[8, 8],
            ],
            dtype=float,
        )
        std_intrinsics[:8] = np.array(
            [
                std_calib[0],
                std_calib[1],
                std_calib[3],
                std_calib[4],
                std_calib[5],
                std_calib[6],
                std_calib[7],
                std_calib[8],
            ],
            dtype=float,
        )
    except RuntimeError as exc:
        covariance_note = f"GTSAM marginal covariance failed: {exc}"

    return {
        "solver_backend": "gtsam",
        "num_views_requested": len(selected_indices),
        "num_views_used": len(used_view_indices),
        "graph_factor_count": int(graph.size()),
        "rms_reprojection_error": rms,
        "camera_matrix": [[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]],
        "dist_coeffs": intrinsics[4:9].tolist(),
        "intrinsics_vector": intrinsics.tolist(),
        "intrinsics_gt": np.asarray(problem.intrinsics_gt, dtype=float).tolist(),
        "intrinsics_init": np.asarray(problem.intrinsics_init, dtype=float).tolist(),
        "intrinsics_error_vs_gt": (intrinsics - np.asarray(problem.intrinsics_gt, dtype=float)).tolist(),
        "parameter_labels": ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"],
        "std_dev_intrinsics": std_intrinsics.tolist(),
        "covariance_diag_intrinsics": covariance_diag.tolist(),
        "covariance_note": covariance_note,
        "k3_handling": "fixed_from_intrinsics_init_not_optimized_by_gtsam_cal3ds2",
        "ransac_filter_used": bool(use_ransac_filter),
        "ransac_reproj_threshold": float(ransac_reproj_threshold),
        "ransac_min_inliers": int(ransac_min_inliers),
        "view_reports": view_reports,
    }


def load_kalibr_camchain(path: pathlib.Path) -> dict:
    intrinsics = None
    distortion = None
    resolution = None
    for line in path.read_text(encoding="ascii").splitlines():
        stripped = line.strip()
        if stripped.startswith("intrinsics:"):
            intrinsics = list(ast.literal_eval(stripped.split(":", 1)[1].strip()))
        elif stripped.startswith("distortion_coeffs:"):
            distortion = list(ast.literal_eval(stripped.split(":", 1)[1].strip()))
        elif stripped.startswith("resolution:"):
            resolution = list(ast.literal_eval(stripped.split(":", 1)[1].strip()))
    if intrinsics is None or distortion is None:
        raise RuntimeError(f"Failed to parse intrinsics/distortion from {path}")
    full = intrinsics + distortion
    while len(full) < 9:
        full.append(0.0)
    return {
        "intrinsics_vector": full[:9],
        "resolution": resolution,
        "path": str(path),
    }


def format_intrinsics_comparison(labels: list[str], estimate: np.ndarray, reference: np.ndarray) -> list[str]:
    lines = []
    for label, est_value, ref_value in zip(labels, estimate, reference):
        lines.append(
            f"  {label}: estimate={float(est_value):.10f}, reference={float(ref_value):.10f}, error={float(est_value - ref_value):+.10f}"
        )
    return lines


def main() -> None:
    args = parse_args()
    problem_path = pathlib.Path(args.problem).expanduser().resolve()
    selection_summary_path = pathlib.Path(args.selection_summary).expanduser().resolve()
    seed_selection_summary_path = None
    if args.seed_selection_summary:
        seed_selection_summary_path = pathlib.Path(args.seed_selection_summary).expanduser().resolve()
    output_path = pathlib.Path(args.output).expanduser().resolve()

    problem = fim.load_calibration_problem_npz(str(problem_path))
    selection_summary_payload = load_json(selection_summary_path)
    selected_indices = load_selected_indices(selection_summary_path)
    if not selected_indices:
        raise RuntimeError(f"No selected_indices found in {selection_summary_path}")

    seed_estimate = None
    seed_indices: list[int] = []
    if seed_selection_summary_path is not None:
        seed_indices = load_selected_indices(seed_selection_summary_path)
        if not seed_indices:
            raise RuntimeError(f"No selected_indices found in {seed_selection_summary_path}")
        if args.solver_backend == "gtsam":
            seed_estimate = estimate_intrinsics_from_selection_gtsam(
                problem,
                seed_indices,
                use_ransac_filter=args.ransac_filter_correspondences,
                ransac_reproj_threshold=args.ransac_reproj_threshold,
                ransac_min_inliers=args.ransac_min_inliers,
            )
        else:
            seed_estimate = estimate_intrinsics_from_selection(
                problem,
                seed_indices,
                use_ransac_filter=args.ransac_filter_correspondences,
                ransac_reproj_threshold=args.ransac_reproj_threshold,
                ransac_min_inliers=args.ransac_min_inliers,
            )
        problem = fim.CalibrationProblem(
            target_points=np.asarray(problem.target_points, dtype=float),
            candidate_rotations=np.asarray(problem.candidate_rotations, dtype=float),
            candidate_translations=np.asarray(problem.candidate_translations, dtype=float),
            measurements=np.asarray(problem.measurements, dtype=float),
            intrinsics_gt=np.asarray(problem.intrinsics_gt, dtype=float),
            intrinsics_init=np.asarray(seed_estimate["intrinsics_vector"], dtype=float),
            image_size=tuple(int(v) for v in problem.image_size),
            pixel_noise_sigma=float(problem.pixel_noise_sigma),
            candidate_board_rotations=None
            if problem.candidate_board_rotations is None
            else np.asarray(problem.candidate_board_rotations, dtype=float),
            candidate_board_translations=None
            if problem.candidate_board_translations is None
            else np.asarray(problem.candidate_board_translations, dtype=float),
            camera_is_fixed=bool(problem.camera_is_fixed),
        )

    if args.solver_backend == "gtsam":
        estimate = estimate_intrinsics_from_selection_gtsam(
            problem,
            selected_indices,
            use_ransac_filter=args.ransac_filter_correspondences,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
            ransac_min_inliers=args.ransac_min_inliers,
        )
    else:
        estimate = estimate_intrinsics_from_selection(
            problem,
            selected_indices,
            use_ransac_filter=args.ransac_filter_correspondences,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
            ransac_min_inliers=args.ransac_min_inliers,
        )

    uncertainty_report = selection_summary_payload.get("before_after_calibration")
    selected_report = selection_summary_payload.get("selected_report")
    report = {
        "created_at": datetime.now().isoformat(),
        "problem_path": str(problem_path),
        "selection_summary_path": str(selection_summary_path),
        "solver_backend": args.solver_backend,
        "num_candidates": int(problem.num_candidates),
        "selected_indices": selected_indices,
        "image_size": [int(v) for v in problem.image_size],
        "intrinsics_gt": np.asarray(problem.intrinsics_gt, dtype=float).tolist(),
        "intrinsics_init": np.asarray(problem.intrinsics_init, dtype=float).tolist(),
        "pixel_noise_sigma": float(problem.pixel_noise_sigma),
        "config": {
            "solver_backend": args.solver_backend,
            "ransac_filter_correspondences": bool(args.ransac_filter_correspondences),
            "ransac_reproj_threshold": float(args.ransac_reproj_threshold),
            "ransac_min_inliers": int(args.ransac_min_inliers),
            "seed_selection_summary_path": None if seed_selection_summary_path is None else str(seed_selection_summary_path),
        },
        "selection_report": selected_report,
        "uncertainty_report": uncertainty_report,
        "final_intrinsics_report": estimate,
    }
    if seed_selection_summary_path is not None and seed_estimate is not None:
        report["seed_estimate"] = seed_estimate
        report["seed_selection_summary_path"] = str(seed_selection_summary_path)
        report["seed_selected_indices"] = seed_indices

    if args.kalibr_camchain:
        kalibr = load_kalibr_camchain(pathlib.Path(args.kalibr_camchain).expanduser().resolve())
        est_vec = np.asarray(estimate["intrinsics_vector"], dtype=float)
        kal_vec = np.asarray(kalibr["intrinsics_vector"], dtype=float)
        report["kalibr_reference"] = kalibr
        report["final_intrinsics_report"]["intrinsics_error_vs_kalibr"] = (est_vec - kal_vec).tolist()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="ascii")

    est_vec = np.asarray(estimate["intrinsics_vector"], dtype=float)
    gt_vec = np.asarray(problem.intrinsics_gt, dtype=float)
    labels = list(estimate.get("parameter_labels", ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]))
    print(f"Saved selected-view intrinsic estimate to: {output_path}")
    print(f"Views requested: {estimate['num_views_requested']}")
    print(f"Views used: {estimate['num_views_used']}")
    print(f"RMS reprojection error: {estimate['rms_reprojection_error']:.6f}")
    print("Final optimized parameters vs ground truth:")
    for line in format_intrinsics_comparison(labels, est_vec, gt_vec):
        print(line)
    if seed_estimate is not None:
        print(f"Seed views used: {seed_estimate['num_views_used']}")
        print(f"Seed RMS reprojection error: {seed_estimate['rms_reprojection_error']:.6f}")
        print(f"Seed intrinsics_init for final fit: {seed_estimate['intrinsics_vector']}")
    if "intrinsics_error_vs_kalibr" in estimate:
        kal_vec = np.asarray(report["kalibr_reference"]["intrinsics_vector"], dtype=float)
        print("Final optimized parameters vs Kalibr:")
        for line in format_intrinsics_comparison(labels, est_vec, kal_vec):
            print(line)


if __name__ == "__main__":
    main()
