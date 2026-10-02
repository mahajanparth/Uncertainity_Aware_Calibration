from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import FIM as infmat


def selection_vector(num_candidates: int, indices: Sequence[int]) -> np.ndarray:
    weights = np.zeros(num_candidates, dtype=float)
    weights[np.asarray(indices, dtype=int)] = 1.0
    return weights


def evaluate_selection(
    problem: infmat.CalibrationProblem,
    indices: Sequence[int],
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
) -> Dict[str, float]:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    weights = selection_vector(problem.num_candidates, indices)
    h_cal = infmat.compute_calibration_schur_compact(problem, weights, info_blocks, prior=prior)
    eigvals = np.linalg.eigvalsh(h_cal)
    reg = 1e-12 * np.eye(h_cal.shape[0], dtype=float)
    cov = np.linalg.pinv(h_cal + reg)

    visible_points = 0
    for idx in indices:
        visible_points += int(info_blocks.visible_counts[int(idx)])

    return {
        "min_eig": float(eigvals[0]),
        "max_eig": float(eigvals[-1]),
        "logdet": float(np.linalg.slogdet(h_cal + reg)[1]),
        "trace_cov": float(np.trace(cov)),
        "cond": float(eigvals[-1] / max(abs(eigvals[0]), 1e-12)),
        "visible_points": float(visible_points),
    }


def candidate_min_eig_scores(
    problem: infmat.CalibrationProblem,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
) -> np.ndarray:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    scores = np.zeros(problem.num_candidates, dtype=float)
    for idx in range(problem.num_candidates):
        weights = np.zeros(problem.num_candidates, dtype=float)
        weights[idx] = 1.0
        scores[idx] = infmat.compute_min_eig_score(problem, weights, info_blocks, prior=prior)
    return scores


def candidate_eigenvalue_spectra(
    problem: infmat.CalibrationProblem,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
) -> np.ndarray:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    spectra = np.zeros((problem.num_candidates, problem.intrinsics_dim), dtype=float)
    for idx in range(problem.num_candidates):
        weights = np.zeros(problem.num_candidates, dtype=float)
        weights[idx] = 1.0
        h_cal = infmat.compute_calibration_schur_compact(problem, weights, info_blocks, prior=prior)
        spectra[idx] = np.linalg.eigvalsh(h_cal)
    return spectra


def before_after_calibration_summary(
    problem: infmat.CalibrationProblem,
    selected_indices: Sequence[int],
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
) -> Dict[str, object]:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    before_selection = np.zeros(problem.num_candidates, dtype=float)
    after_selection = selection_vector(problem.num_candidates, selected_indices)

    before_h_cal = infmat.compute_calibration_schur_compact(problem, before_selection, info_blocks, prior=prior)
    after_h_cal = infmat.compute_calibration_schur_compact(problem, after_selection, info_blocks, prior=prior)

    reg = 1e-12 * np.eye(problem.intrinsics_dim, dtype=float)
    before_eigvals = np.linalg.eigvalsh(before_h_cal)
    after_eigvals = np.linalg.eigvalsh(after_h_cal)
    before_cov = np.linalg.pinv(before_h_cal + reg)
    after_cov = np.linalg.pinv(after_h_cal + reg)
    before_std = np.sqrt(np.maximum(np.diag(before_cov), 0.0))
    after_std = np.sqrt(np.maximum(np.diag(after_cov), 0.0))

    return {
        "parameter_labels": intrinsic_parameter_labels(problem.intrinsics_dim),
        "before": {
            "h_cal": before_h_cal.tolist(),
            "eigvals": before_eigvals.tolist(),
            "min_eig": float(before_eigvals[0]),
            "covariance_diag": np.diag(before_cov).tolist(),
            "std_dev": before_std.tolist(),
        },
        "after": {
            "h_cal": after_h_cal.tolist(),
            "eigvals": after_eigvals.tolist(),
            "min_eig": float(after_eigvals[0]),
            "covariance_diag": np.diag(after_cov).tolist(),
            "std_dev": after_std.tolist(),
        },
    }


def intrinsic_parameter_labels(intrinsics_dim: int) -> List[str]:
    default = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"]
    if intrinsics_dim <= len(default):
        return default[:intrinsics_dim]
    extra = [f"theta_{idx}" for idx in range(len(default), intrinsics_dim)]
    return default + extra


def random_baseline_report(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
    num_trials: int = 50,
    seed: int = 0,
) -> Dict[str, object]:
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    rng = np.random.default_rng(seed)
    trials: List[Dict[str, float]] = []
    selections: List[List[int]] = []
    for _ in range(num_trials):
        indices = sorted(rng.choice(problem.num_candidates, size=select_k, replace=False).tolist())
        selections.append(indices)
        trials.append(evaluate_selection(problem, indices, prior=prior))

    min_eigs = np.array([trial["min_eig"] for trial in trials], dtype=float)
    best_idx = int(np.argmax(min_eigs))
    avg_report = {
        key: float(np.mean([trial[key] for trial in trials]))
        for key in trials[0].keys()
    }
    best_report = trials[best_idx]
    return {
        "average": avg_report,
        "best": best_report,
        "best_indices": selections[best_idx],
        "trials": trials,
    }


def calibrate_opencv(
    problem: infmat.CalibrationProblem,
    selected_indices: Sequence[int],
) -> Tuple[Optional[np.ndarray], float]:
    """Calibrate intrinsics with OpenCV on the selected subset.

    Returns (intrinsics_9vec, train_rms) where intrinsics_9vec = [fx,fy,cx,cy,k1,k2,p1,p2,k3].
    Returns (None, inf) if fewer than 3 views have valid measurements.
    """
    result = calibrate_opencv_full(problem, selected_indices)
    if not result["success"]:
        return None, float("inf")
    return result["intrinsics"], result["train_rms"]


def calibrate_opencv_full(
    problem: infmat.CalibrationProblem,
    selected_indices: Sequence[int],
) -> dict:
    """Calibrate intrinsics with OpenCV, returning poses and valid index list too.

    Returns a dict with keys:
      success      : bool
      intrinsics   : ndarray [fx,fy,cx,cy,k1,k2,p1,p2,k3]
      train_rms    : float (reprojection RMS on training views, px)
      K_mat        : (3,3) camera matrix
      dist         : (5,) distortion coefficients
      rvecs        : list of (3,1) Rodrigues vectors (world-to-camera, one per view)
      tvecs        : list of (3,1) translation vectors (world-to-camera, m)
      valid_indices: list of candidate indices that were passed to cv2
    """
    import cv2

    obj_points: List[np.ndarray] = []
    img_points: List[np.ndarray] = []
    valid_indices: List[int] = []
    for idx in selected_indices:
        meas = problem.measurements[int(idx)]
        valid = ~np.any(np.isnan(meas), axis=1)
        if valid.sum() < 4:
            continue
        obj_points.append(problem.target_points[valid].astype(np.float32))
        img_points.append(meas[valid].astype(np.float32).reshape(-1, 1, 2))
        valid_indices.append(int(idx))

    if len(obj_points) < 3:
        return {"success": False}

    w, h = int(problem.image_size[0]), int(problem.image_size[1])
    fx0 = float(max(w, h)) * 0.7
    K_init = np.array([[fx0, 0.0, w / 2.0], [0.0, fx0, h / 2.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    try:
        rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, (w, h), K_init.copy(), None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS,
        )
    except cv2.error:
        return {"success": False}

    d = dist.flatten()
    while len(d) < 5:
        d = np.append(d, 0.0)
    intrinsics = np.array([K[0, 0], K[1, 1], K[0, 2], K[1, 2], d[0], d[1], d[2], d[3], d[4]])
    return {
        "success": True,
        "intrinsics": intrinsics,
        "train_rms": float(rms),
        "K_mat": K,
        "dist": d,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "valid_indices": valid_indices,
    }


def compute_groupwise_errors(
    problem: infmat.CalibrationProblem,
    intrinsics_est: np.ndarray,
    valid_indices: Sequence[int],
    rvecs: list,
    tvecs: list,
) -> dict:
    """Compute physically interpretable group-wise calibration errors.

    Returns:
      focal_err_px  : L2 norm of (fx_err, fy_err) in pixels
      pp_err_px     : L2 norm of (cx_err, cy_err) in pixels
      dist_err      : L2 norm of distortion-coefficient vector error (unitless)
      rot_err_deg   : mean geodesic rotation error over calibration poses (degrees)
      trans_err_cm  : mean translation error over calibration poses (cm)
    """
    import cv2

    gt = np.asarray(problem.intrinsics_gt, dtype=float)
    est = intrinsics_est[: len(gt)]

    focal_err = float(np.sqrt((est[0] - gt[0]) ** 2 + (est[1] - gt[1]) ** 2))
    pp_err = float(np.sqrt((est[2] - gt[2]) ** 2 + (est[3] - gt[3]) ** 2))

    if len(gt) >= 9:
        dist_err = float(np.linalg.norm(est[4:9] - gt[4:9]))
    elif len(gt) > 4:
        dist_err = float(np.linalg.norm(est[4:] - gt[4:]))
    else:
        dist_err = float("nan")

    rot_errors_deg: List[float] = []
    trans_errors_cm: List[float] = []
    for i, prob_idx in enumerate(valid_indices):
        R_wc_gt = problem.candidate_rotations[int(prob_idx)]
        t_wc_gt = problem.candidate_translations[int(prob_idx)]
        R_cw_gt = R_wc_gt.T
        t_cw_gt = (-R_cw_gt @ t_wc_gt).flatten()

        R_est, _ = cv2.Rodrigues(rvecs[i])
        t_est = tvecs[i].flatten()

        # Geodesic rotation error
        cos_angle = float(np.clip((np.trace(R_est @ R_cw_gt.T) - 1.0) / 2.0, -1.0, 1.0))
        rot_errors_deg.append(float(np.degrees(np.arccos(cos_angle))))

        # Translation error metres → cm
        trans_errors_cm.append(float(np.linalg.norm(t_est - t_cw_gt) * 100.0))

    param_err = float(np.sqrt(focal_err**2 + pp_err**2 + dist_err**2)) if np.isfinite(dist_err) else float("nan")

    return {
        "focal_err_px": focal_err,
        "pp_err_px": pp_err,
        "dist_err": dist_err,
        "param_err": param_err,
        "rot_err_deg": float(np.mean(rot_errors_deg)) if rot_errors_deg else float("nan"),
        "trans_err_cm": float(np.mean(trans_errors_cm)) if trans_errors_cm else float("nan"),
    }


def heldout_reprojection_error(
    problem: infmat.CalibrationProblem,
    selected_indices: Sequence[int],
    K_mat: np.ndarray,
    dist_mat: np.ndarray,
) -> float:
    """Mean per-point reprojection error on poses NOT in selected_indices.

    Uses the known candidate poses (R_wc, t_wc) from the problem — no PnP needed.
    """
    import cv2

    selected_set = set(int(i) for i in selected_indices)
    errors = []
    for idx in range(problem.num_candidates):
        if idx in selected_set:
            continue
        meas = problem.measurements[idx]              # (M, 2)
        valid = ~np.any(np.isnan(meas), axis=1)
        if valid.sum() < 4:
            continue
        obj_pts = problem.target_points[valid].astype(np.float32)
        img_pts = meas[valid].astype(np.float32)

        R_wc = problem.candidate_rotations[idx]
        t_wc = problem.candidate_translations[idx]
        R_cw = R_wc.T
        t_cw = (-R_cw @ t_wc).reshape(3, 1)
        rvec, _ = cv2.Rodrigues(R_cw)
        projected, _ = cv2.projectPoints(obj_pts, rvec, t_cw, K_mat, dist_mat)
        projected = projected.reshape(-1, 2)
        errors.append(float(np.linalg.norm(projected - img_pts, axis=1).mean()))

    return float(np.mean(errors)) if errors else float("inf")


def compare_selected_vs_random(
    problem: infmat.CalibrationProblem,
    selected_indices: Sequence[int],
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
    num_random_trials: int = 50,
    seed: int = 0,
) -> Dict[str, object]:
    selected_report = evaluate_selection(problem, selected_indices, prior=prior)
    random_report = random_baseline_report(
        problem,
        select_k=len(selected_indices),
        prior=prior,
        num_trials=num_random_trials,
        seed=seed,
    )
    return {
        "selected": selected_report,
        "random_average": random_report["average"],
        "random_best": random_report["best"],
        "random_best_indices": random_report["best_indices"],
        "random_trials": random_report["trials"],
    }
