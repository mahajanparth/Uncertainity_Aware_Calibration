"""Real-world ChArUco calibration benchmark using live-demo captures.

Loads corner detections from a captures directory, estimates camera poses
via solvePnP, reserves a fixed held-out test set (farthest-point sampled for
diversity), then runs all selection methods on the remaining candidate pool and
reports held-out reprojection error (no ground-truth intrinsics needed).

Usage:
    python3 -m Experiments.main_real_charuco \
        [--captures-dir live_demo_charuco/captures] \
        [--summary-json live_demo_charuco/live_session_summary.json] \
        [--select-k 20] [--held-out-count 20] [--min-corners 20] \
        [--output-dir results/real]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import List, Tuple

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal
from OASIS import methods as sel


# ---------------------------------------------------------------------------
# Board helpers
# ---------------------------------------------------------------------------

def _build_global_corners(rows: int, cols: int, sq: float) -> np.ndarray:
    """All inner ChArUco corners in board frame, shape ((rows-1)*(cols-1), 3)."""
    pts = []
    for r in range(rows - 1):
        for c in range(cols - 1):
            pts.append([(c + 1) * sq, (r + 1) * sq, 0.0])
    return np.array(pts, dtype=float)


# ---------------------------------------------------------------------------
# Data loading + pose estimation
# ---------------------------------------------------------------------------

def load_and_estimate(
    captures_dir: pathlib.Path,
    rows: int,
    cols: int,
    sq: float,
    min_corners: int,
    K_mat: np.ndarray,
    dist: np.ndarray,
    min_depth_m: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Load JSON captures and estimate poses via solvePnP.

    Filters frames where solvePnP fails, board is behind camera, or any
    visible board point is closer than min_depth_m in camera frame.

    Returns:
        measurements  : (N, M, 2) float, NaN for undetected corners
        rotations     : (N, 3, 3) R_wc
        translations  : (N, 3)    t_wc (camera centre in world)
        raw_detections: list of (obj_pts, img_pts, R_wc, t_wc) for held-out eval
    """
    M = (rows - 1) * (cols - 1)
    n_cols_inner = cols - 1
    n_rows_inner = rows - 1

    json_files = sorted(captures_dir.glob("*.json"))

    meas_list: List[np.ndarray] = []
    rot_list:  List[np.ndarray] = []
    trans_list: List[np.ndarray] = []
    raw_list:  list = []
    n_rejected = 0

    for jf in json_files:
        d = json.loads(jf.read_text())
        if d["num_points"] < min_corners:
            n_rejected += 1
            continue

        obj_pts = np.array(d["object_points"], dtype=np.float32)
        img_pts = np.array(d["image_points"],  dtype=np.float32)

        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts,
            K_mat.astype(np.float32),
            dist.astype(np.float32),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            n_rejected += 1
            continue

        R_cw, _ = cv2.Rodrigues(rvec)
        pts_c = (R_cw @ obj_pts.T).T + tvec.flatten()

        # All visible board points must be at least min_depth_m in front
        if pts_c[:, 2].min() < min_depth_m:
            n_rejected += 1
            continue

        R_wc = R_cw.T
        t_wc = -R_wc @ tvec.flatten()

        meas = np.full((M, 2), np.nan, dtype=float)
        for op, ip in zip(obj_pts, img_pts):
            c = round(float(op[0]) / sq) - 1
            r = round(float(op[1]) / sq) - 1
            if 0 <= r < n_rows_inner and 0 <= c < n_cols_inner:
                meas[r * n_cols_inner + c] = ip

        meas_list.append(meas)
        rot_list.append(R_wc)
        trans_list.append(t_wc)
        raw_list.append((obj_pts.copy(), img_pts.copy(), R_wc.copy(), t_wc.copy()))

    print(f"  {len(meas_list)} frames kept, {n_rejected} rejected")
    return (
        np.array(meas_list,  dtype=float),
        np.array(rot_list,   dtype=float),
        np.array(trans_list, dtype=float),
        raw_list,
    )


# ---------------------------------------------------------------------------
# Held-out selection (farthest-point sampling on camera positions)
# ---------------------------------------------------------------------------

def farthest_point_sample(positions: np.ndarray, k: int, seed: int = 0) -> List[int]:
    """Select k indices that maximally cover the position space."""
    rng = np.random.default_rng(seed)
    n = len(positions)
    selected = [int(rng.integers(n))]
    dists = np.full(n, np.inf)
    for _ in range(k - 1):
        d = np.linalg.norm(positions - positions[selected[-1]], axis=1)
        dists = np.minimum(dists, d)
        selected.append(int(np.argmax(dists)))
    return selected


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _fim_metrics(problem, indices, prior):
    try:
        return cal.evaluate_selection(problem, indices, prior=prior)
    except Exception as e:
        print(f"    [warn] FIM metrics failed: {e}")
        return {"min_eig": float("nan"), "logdet": float("nan"),
                "trace_cov": float("nan"), "cond": float("nan"), "visible_points": 0.0}


def _calib_metrics(problem, candidate_indices, held_out_poses, K_seed, dist_seed,
                   fixed_poses: bool = True):
    """Calibrate on selected subset, evaluate on held-out frames.

    fixed_poses=True  : use poses estimated once at load time (honest metric).
    fixed_poses=False : re-run solvePnP per held-out frame with calibrated K
                        (soft metric — matches original paper protocol).
    """
    result = cal.calibrate_opencv_full(problem, candidate_indices)
    if not result["success"]:
        nan = float("nan")
        return {"train_rms": nan, "heldout_rms": nan}

    K_cal = result["K_mat"]
    dist_cal = result["dist"][:5].reshape(1, 5)

    heldout_errors = []
    if fixed_poses:
        for obj_pts, img_pts, R_wc, t_wc in held_out_poses:
            R_cw = R_wc.T
            t_cw = (-R_cw @ t_wc).reshape(3, 1)
            rvec, _ = cv2.Rodrigues(R_cw)
            proj, _ = cv2.projectPoints(obj_pts.astype(np.float32), rvec, t_cw, K_cal, dist_cal)
            proj = proj.reshape(-1, 2)
            heldout_errors.append(float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1)))))
    else:
        for obj_pts, img_pts, *_ in held_out_poses:
            ok, rvec, tvec = cv2.solvePnP(
                obj_pts, img_pts, K_cal, dist_cal, flags=cv2.SOLVEPNP_ITERATIVE)
            if not ok:
                continue
            proj, _ = cv2.projectPoints(obj_pts, rvec, tvec, K_cal, dist_cal)
            proj = proj.reshape(-1, 2)
            heldout_errors.append(float(np.sqrt(np.mean(np.sum((proj - img_pts) ** 2, axis=1)))))

    heldout_rms = float(np.mean(heldout_errors)) if heldout_errors else float("nan")
    return {"train_rms": result["train_rms"], "heldout_rms": heldout_rms}


def _ser(v):
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    return v


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures-dir", type=pathlib.Path,
                        default="live_demo_charuco/captures")
    parser.add_argument("--summary-json", type=pathlib.Path,
                        default="live_demo_charuco/live_session_summary.json")
    parser.add_argument("--select-k", type=int, default=20)
    parser.add_argument("--held-out-count", type=int, default=20,
                        help="Number of frames to reserve as a fixed test set")
    parser.add_argument("--min-corners", type=int, default=20)
    parser.add_argument("--min-depth-m", type=float, default=0.05,
                        help="Minimum camera-frame depth (m) for all visible board points")
    parser.add_argument("--random-trials", type=int, default=25)
    parser.add_argument("--held-out-seed", type=int, default=42,
                        help="RNG seed for farthest-point held-out sampling")
    parser.add_argument("--fixed-poses", action="store_true", default=False,
                        help="Use fixed held-out poses (honest). Default: solvePnP re-fit (paper protocol).")
    parser.add_argument("--output-dir", type=pathlib.Path, default="results/real")
    args = parser.parse_args()

    fim.set_information_backend("numeric")

    # ── Load session summary ────────────────────────────────────────────────
    summary = json.loads(args.summary_json.read_text())
    board_rows   = summary["board_rows"]
    board_cols   = summary["board_cols"]
    sq           = summary["board_square_size"]
    image_size   = tuple(summary["image_size"])          # (width, height)
    intr         = np.array(summary["final_intrinsics"], dtype=float)
    K_seed = np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1.]])
    dist_seed = intr[4:9].reshape(1, 5)

    print(f"Board   : {board_rows}x{board_cols}, square={sq*100:.1f} cm")
    print(f"Image   : {image_size[0]}x{image_size[1]} px")
    print(f"Intr    : fx={intr[0]:.1f}  fy={intr[1]:.1f}  "
          f"cx={intr[2]:.1f}  cy={intr[3]:.1f}")
    print(f"Dist    : {intr[4:9].tolist()}")

    # ── Build global target points ──────────────────────────────────────────
    target_pts = _build_global_corners(board_rows, board_cols, sq)
    M = target_pts.shape[0]
    print(f"Corners : {M} total inner corners")

    # ── Load captures + estimate poses ────────────────────────────────────
    print(f"\nLoading captures from {args.captures_dir} "
          f"(min_corners={args.min_corners}, min_depth={args.min_depth_m:.2f}m) ...")
    measurements, rotations, translations, raw_detections = load_and_estimate(
        args.captures_dir, board_rows, board_cols, sq, args.min_corners,
        K_seed, intr[4:9], min_depth_m=args.min_depth_m,
    )
    N_total = measurements.shape[0]
    if N_total < args.held_out_count + args.select_k:
        print(f"ERROR: only {N_total} valid frames, need at least "
              f"{args.held_out_count + args.select_k}")
        sys.exit(1)

    # ── Split held-out / candidate pool ────────────────────────────────────
    held_out_idx = farthest_point_sample(translations, args.held_out_count, seed=args.held_out_seed)
    held_out_set = set(held_out_idx)
    candidate_idx = [i for i in range(N_total) if i not in held_out_set]

    # ── Estimate best-available intrinsics from ALL frames ──────────────────
    # Use all N frames (candidates + held-out) to get the most accurate
    # intrinsics for re-estimating held-out poses. This avoids baking seed
    # intrinsic error into the fixed held-out poses.
    all_obj = [rd[0] for rd in raw_detections]
    all_img = [rd[1].reshape(-1, 1, 2) for rd in raw_detections]
    w, h = int(image_size[0]), int(image_size[1])
    rms_all, K_all, dist_all, _, _ = cv2.calibrateCamera(
        all_obj, all_img, (w, h), K_seed.copy(), None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS,
    )
    dist_all = dist_all.flatten()
    while len(dist_all) < 5:
        dist_all = np.append(dist_all, 0.0)
    dist_all_mat = dist_all[:5].reshape(1, 5)
    print(f"All-frame calibration: RMS={rms_all:.4f} px  "
          f"fx={K_all[0,0]:.1f}  fy={K_all[1,1]:.1f}  "
          f"cx={K_all[0,2]:.1f}  cy={K_all[1,2]:.1f}")

    # Re-estimate held-out poses with the all-frame intrinsics, then fix them
    held_out_poses = []
    for i in held_out_idx:
        obj_pts, img_pts = raw_detections[i][0], raw_detections[i][1]
        ok, rvec, tvec = cv2.solvePnP(
            obj_pts, img_pts, K_all, dist_all_mat, flags=cv2.SOLVEPNP_ITERATIVE)
        if ok:
            R_cw, _ = cv2.Rodrigues(rvec)
            R_wc = R_cw.T
            t_wc = (-R_wc @ tvec.flatten())
        else:
            R_wc, t_wc = raw_detections[i][2], raw_detections[i][3]
        held_out_poses.append((obj_pts, img_pts, R_wc, t_wc))

    # Build CalibrationProblem over candidate pool only
    cand_meas  = measurements[candidate_idx]
    cand_rots  = rotations[candidate_idx]
    cand_trans = translations[candidate_idx]
    N_cand = len(candidate_idx)

    print(f"\nHeld-out : {args.held_out_count} frames (farthest-point sampled)")
    print(f"Candidate: {N_cand} frames")

    intr_init = intr + fim.default_intrinsics_init_offset(intr.size)
    problem = fim.CalibrationProblem(
        target_points=target_pts,
        candidate_rotations=cand_rots,
        candidate_translations=cand_trans,
        measurements=cand_meas,
        intrinsics_gt=intr,          # best-estimate proxy (no true GT)
        intrinsics_init=intr_init,
        image_size=image_size,
        pixel_noise_sigma=1.0,
    )

    K   = args.select_k
    prior = fim.build_prior_blocks(problem)
    print(f"\nRunning selection methods  K={K}, N_cand={N_cand}\n")

    results = []

    # ── Random baseline ─────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    rand_rows = []
    for trial in range(args.random_trials):
        idx = list(rng.choice(N_cand, K, replace=False))
        r = {}
        r.update(_fim_metrics(problem, idx, prior))
        r.update(_calib_metrics(problem, idx, held_out_poses, K_seed, intr[4:9], fixed_poses=args.fixed_poses))
        rand_rows.append(r)
    rand_mean = {
        "method": "random",
        "selected_indices": [],
        "min_eig":    float(np.nanmean([r["min_eig"]    for r in rand_rows])),
        "logdet":     float(np.nanmean([r["logdet"]     for r in rand_rows])),
        "train_rms":  float(np.nanmean([r["train_rms"]  for r in rand_rows])),
        "heldout_rms":float(np.nanmean([r["heldout_rms"]for r in rand_rows])),
    }
    results.append(rand_mean)
    print(f"  random          λ_min={rand_mean['min_eig']:.4f}  "
          f"held-out={rand_mean['heldout_rms']:.4f} px")

    # ── Deterministic methods ───────────────────────────────────────────────
    det_methods = [
        ("coverage",         lambda: sel.coverage_selection(problem, K) + (None,)),
        ("motion-diversity", lambda: sel.motion_diversity_selection(problem, K) + (None,)),
        ("a-optimal",        lambda: sel.greedy_selection_a_optimal(problem, K, prior=prior) + (None,)),
        ("d-optimal",        lambda: sel.greedy_selection_d_optimal(problem, K, prior=prior) + (None,)),
        ("greedy-e",         lambda: sel.greedy_selection(problem, K, prior=prior) + (None,)),
        ("fw-e",             lambda: sel.frank_wolfe_selection(problem, K, prior=prior)),
        ("fw-a",             lambda: sel.frank_wolfe_a_optimal_selection(problem, K, prior=prior)),
        ("fw-d",             lambda: sel.frank_wolfe_d_optimal_selection(problem, K, prior=prior)),
    ]

    for name, fn in det_methods:
        print(f"  [{name}]", end=" ", flush=True)
        t0 = time.time()
        try:
            _, selected_indices, best_score, _, *_ = fn()
        except Exception as e:
            print(f"FAILED: {e}")
            results.append({"method": name, "min_eig": float("nan"),
                            "logdet": float("nan"), "train_rms": float("nan"),
                            "heldout_rms": float("nan"), "time_s": float("nan")})
            continue
        elapsed = time.time() - t0
        r = {"method": name, "selected_indices": list(selected_indices), "time_s": elapsed}
        r.update(_fim_metrics(problem, selected_indices, prior))
        r.update(_calib_metrics(problem, selected_indices, held_out_poses, K_seed, intr[4:9]))
        results.append({k: _ser(v) for k, v in r.items()})
        print(f"done {elapsed:.1f}s  λ_min={r['min_eig']:.4f}  "
              f"logdet={r['logdet']:.1f}  held-out={r['heldout_rms']:.4f} px")

    # ── Save JSON ────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "real_benchmark_results.json"
    out_path.write_text(json.dumps({
        "captures_dir": str(args.captures_dir),
        "n_total": N_total,
        "n_candidates": N_cand,
        "n_held_out": args.held_out_count,
        "held_out_pool_indices": held_out_idx,
        "select_k": K,
        "board_rows": board_rows,
        "board_cols": board_cols,
        "square_size": sq,
        "image_size": list(image_size),
        "random_trials": args.random_trials,
        "results": results,
    }, indent=2))
    print(f"\nSaved → {out_path}")

    # ── Print summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print(f"{'Method':<18} {'λ_min':>10} {'logdet':>8} {'Train':>8} {'Held-out':>10}")
    print("=" * 65)
    for r in results:
        print(f"  {r['method']:<16} "
              f"{r.get('min_eig', float('nan')):>10.4f} "
              f"{r.get('logdet', float('nan')):>8.1f} "
              f"{r.get('train_rms', float('nan')):>8.4f} "
              f"{r.get('heldout_rms', float('nan')):>10.4f}")
    print("=" * 65)
    print(f"\nHeld-out set: {args.held_out_count} frames fixed across all methods.")


if __name__ == "__main__":
    main()
