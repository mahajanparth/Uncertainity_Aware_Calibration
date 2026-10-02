"""Compare view-selection methods using Kalibr as the downstream calibration backend.

Pipeline
--------
1. Generate N=720 hemisphere poses (5 radii × 6 elevations × 24 azimuths).
2. Render a synthetic AprilGrid image at each pose (perspective-warped texture).
3. Build a CalibrationProblem from projected corners + Gaussian pixel noise.
4. Reserve 20 held-out views (farthest-point sampling on translations).
5. For each selection method → K=20 indices from remaining N-20 candidates:
   a. Write images to cam0/ subfolder for kalibr_bagcreater.
   b. docker run kalibr  (bagcreater + kalibr_calibrate_cameras --mi-tol -1).
   c. Parse <bag>-camchain.yaml for intrinsics and distortion.
   d. Evaluate held-out RMS with solvePnP(K_hat, dist_hat).
6. Save benchmark_results.json and summary CSV.

Usage
-----
    python3 -m Experiments.main_kalibr_aprilgrid \\
        --output-dir results/kalibr_aprilgrid \\
        [--select-k 20] [--held-out-count 20] \\
        [--noise-sigma 0.5] [--docker-image kalibr]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal
from OASIS import methods as sel
from isaac_sim_scripts.calibration_board_utils import ensure_aprilgrid_texture

# ---------------------------------------------------------------------------
# Board & camera constants (match generate_synthetic_dataset.py conventions)
# ---------------------------------------------------------------------------

_ROWS = 6
_COLS = 6
_TAG_SIZE = 0.033        # metres
_TAG_SPACING = 0.3       # ratio of tag_size
_TAG_PITCH = _TAG_SIZE * (1.0 + _TAG_SPACING)  # = 0.0429 m
_BOARD_W = _COLS * _TAG_PITCH                   # = 0.2574 m
_BOARD_H = _ROWS * _TAG_PITCH                   # = 0.2574 m
_PPP = 200               # pixels per tag pitch in texture

_FX = _FY = 1649.248
_CX, _CY = 960.0, 540.0
_IMAGE_W, _IMAGE_H = 1920, 1080
_IMAGE_SIZE = (_IMAGE_W, _IMAGE_H)
_INTRINSICS = np.array([_FX, _FY, _CX, _CY, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

_TARGET_CENTER = np.array([0.4, 0.0, 0.0], dtype=float)
_Z_BOARD = _TARGET_CENTER[2] + 0.0038  # board plane z in world frame

_RADII = [0.28, 0.34, 0.40, 0.46, 0.52]
_ELEVATIONS_DEG = [12, 22, 32, 42, 52, 62]
_AZIMUTH_COUNT = 24

# ---------------------------------------------------------------------------
# Hemisphere pose helpers (inlined from generate_synthetic_dataset.py)
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = _normalize(axis)
    c, s, omc = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    x, y, z = axis
    return np.array([
        [c + x*x*omc,    x*y*omc - z*s, x*z*omc + y*s],
        [y*x*omc + z*s,  c + y*y*omc,   y*z*omc - x*s],
        [z*x*omc - y*s,  z*y*omc + x*s, c + z*z*omc],
    ], dtype=float)


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    trace = np.trace(R)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        return np.array([0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s])
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
        return np.array([(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s])
    elif R[1,1] > R[2,2]:
        s = math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
        return np.array([(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s])
    else:
        s = math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
        return np.array([(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s])


def _quat_to_rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),  2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),  1-2*(x*x+y*y)],
    ], dtype=float)


def _look_at(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    fwd = _normalize(target - pos)
    up = np.array([0., 0., 1.])
    if abs(np.dot(fwd, up)) > 0.98:
        up = np.array([0., 1., 0.])
    right = _normalize(np.cross(fwd, up))
    up2 = _normalize(np.cross(right, fwd))
    return np.column_stack([right, up2, -fwd])   # R_usd (world→camera in USD convention)


def _usd_to_calib(R_usd: np.ndarray) -> np.ndarray:
    return R_usd @ np.diag([1., -1., -1.])


def generate_hemisphere_poses() -> tuple[np.ndarray, np.ndarray]:
    """Return (candidate_rotations, candidate_translations) in calibration convention."""
    rotations, translations = [], []
    for radius in sorted(_RADII):
        for elev_deg in sorted(_ELEVATIONS_DEG):
            elev = math.radians(elev_deg)
            xy_r = radius * math.cos(elev)
            z_off = radius * math.sin(elev)
            for az_idx in range(_AZIMUTH_COUNT):
                az = 2.0 * math.pi * az_idx / _AZIMUTH_COUNT
                pos = _TARGET_CENTER + np.array([
                    xy_r * math.cos(az), xy_r * math.sin(az), z_off], dtype=float)
                R_usd = _look_at(pos, _TARGET_CENTER)
                R_cal = _usd_to_calib(R_usd)
                rotations.append(R_cal)
                translations.append(pos)
    return np.array(rotations), np.array(translations)


# ---------------------------------------------------------------------------
# AprilGrid corner points (for FIM / held-out evaluation)
# ---------------------------------------------------------------------------

def build_aprilgrid_world_points() -> np.ndarray:
    """Inner corners of the AprilGrid in world coordinates.

    Kalibr's AprilGrid uses the 4 inner corners of each tag.
    local_x increases right, local_y increases down → world_y flipped.
    """
    pts = []
    for row in range(_ROWS):
        for col in range(_COLS):
            cx_local = (col + 0.5) * _TAG_PITCH  # centre of tag
            cy_local = (row + 0.5) * _TAG_PITCH
            half = _TAG_SIZE / 2.0
            corners_local = [
                (cx_local - half, cy_local - half),
                (cx_local + half, cy_local - half),
                (cx_local + half, cy_local + half),
                (cx_local - half, cy_local + half),
            ]
            for lx, ly in corners_local:
                wx = _TARGET_CENTER[0] - _BOARD_W / 2.0 + lx
                wy = _TARGET_CENTER[1] + _BOARD_H / 2.0 - ly
                pts.append([wx, wy, _Z_BOARD])
    return np.array(pts, dtype=float)


# ---------------------------------------------------------------------------
# Image rendering
# ---------------------------------------------------------------------------

def _board_world_corners() -> np.ndarray:
    """4 outer corners of the board in world frame (TL, TR, BR, BL)."""
    x0 = _TARGET_CENTER[0] - _BOARD_W / 2.0
    x1 = _TARGET_CENTER[0] + _BOARD_W / 2.0
    y0 = _TARGET_CENTER[1] - _BOARD_H / 2.0
    y1 = _TARGET_CENTER[1] + _BOARD_H / 2.0
    z  = _Z_BOARD
    return np.array([
        [x0, y1, z],  # TL
        [x1, y1, z],  # TR
        [x1, y0, z],  # BR
        [x0, y0, z],  # BL
    ], dtype=float)


_BOARD_CORNERS_WORLD = _board_world_corners()


def render_image(texture_gray: np.ndarray, R_wc: np.ndarray, t_wc: np.ndarray,
                 target_pts: np.ndarray, rng: np.random.Generator,
                 noise_sigma: float = 0.5,
                 min_visible_corners: int = 20) -> np.ndarray | None:
    """Perspective-warp the texture into a 1920×1080 grayscale image.

    Returns None if fewer than min_visible_corners are in-bounds, or the
    camera is below the minimum depth.
    """
    # Reject views with camera too close to the board plane
    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc
    pts_c = (target_pts @ R_cw.T) + t_cw
    if pts_c[:, 2].min() < 0.05:
        return None

    # Count corners within image bounds
    proj_corners = fim.project_points(target_pts, R_wc, t_wc, _INTRINSICS)
    in_bounds = (
        np.isfinite(proj_corners).all(axis=1)
        & (proj_corners[:, 0] >= 0) & (proj_corners[:, 0] < _IMAGE_W)
        & (proj_corners[:, 1] >= 0) & (proj_corners[:, 1] < _IMAGE_H)
    )
    if int(in_bounds.sum()) < min_visible_corners:
        return None

    # Compute homography from texture pixels → image pixels using the 4 board corners
    proj_board = fim.project_points(_BOARD_CORNERS_WORLD, R_wc, t_wc, _INTRINSICS)
    if not np.all(np.isfinite(proj_board)):
        return None

    tex_h, tex_w = texture_gray.shape
    src_pts = np.array([[0, 0], [tex_w, 0], [tex_w, tex_h], [0, tex_h]], dtype=np.float32)
    dst_pts = proj_board.astype(np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return None

    warped = cv2.warpPerspective(texture_gray, H, (_IMAGE_W, _IMAGE_H), borderValue=200)
    if noise_sigma > 0:
        noise = rng.normal(0, noise_sigma, warped.shape)
        warped = np.clip(warped.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return warped


# ---------------------------------------------------------------------------
# Farthest-point sampling (for held-out set selection)
# ---------------------------------------------------------------------------

def farthest_point_sample(positions: np.ndarray, k: int, seed: int = 42) -> list[int]:
    rng = np.random.default_rng(seed)
    n = len(positions)
    chosen = [int(rng.integers(n))]
    dists = np.full(n, np.inf)
    for _ in range(k - 1):
        last = positions[chosen[-1]]
        d = np.linalg.norm(positions - last, axis=1)
        dists = np.minimum(dists, d)
        dists[chosen] = -np.inf
        chosen.append(int(np.argmax(dists)))
    return chosen


# ---------------------------------------------------------------------------
# Kalibr Docker runner
# ---------------------------------------------------------------------------

def _write_target_yaml(path: pathlib.Path) -> None:
    path.write_text(
        "target_type: 'aprilgrid'\n"
        f"tagCols: {_COLS}\n"
        f"tagRows: {_ROWS}\n"
        f"tagSize: {_TAG_SIZE:.12g}\n"
        f"tagSpacing: {_TAG_SPACING:.12g}\n",
        encoding="ascii",
    )


def run_kalibr(
    image_dir: pathlib.Path,
    work_dir: pathlib.Path,
    target_yaml: pathlib.Path,
    docker_image: str = "kalibr",
    timeout: int = 600,
) -> dict[str, Any] | None:
    """Run Kalibr in Docker. Returns parsed intrinsics dict or None on failure.

    image_dir  — host dir containing images (will be mounted into cam0/).
    work_dir   — host dir where bag + results will be written.
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    # Kalibr bagcreater needs a folder with cam0/ subfolder
    cam0_dir = image_dir / "cam0"
    cam0_dir.mkdir(parents=True, exist_ok=True)

    host_work = str(work_dir.resolve())
    host_images = str(image_dir.resolve())
    host_target = str(target_yaml.parent.resolve())
    target_name = target_yaml.name

    bag_name = "calib.bag"
    docker_cmd = (
        "set -e; "
        "source /catkin_ws/devel/setup.bash; "
        "cd /data/work; "
        f"/catkin_ws/devel/lib/kalibr/kalibr_bagcreater "
        f"--folder /data/images --output-bag /data/work/{bag_name}; "
        "/catkin_ws/devel/lib/kalibr/kalibr_calibrate_cameras "
        f"--bag /data/work/{bag_name} "
        "--topics /cam0/image_raw "
        "--models pinhole-radtan "
        f"--target /data/target/{target_name} "
        "--mi-tol -1 "
        "--no-shuffle "
        "--no-final-filtering"
    )

    cmd = [
        "docker", "run", "--rm",
        "--entrypoint", "/ros_entrypoint.sh",
        "-v", f"{host_images}:/data/images:ro",
        "-v", f"{host_work}:/data/work:rw",
        "-v", f"{host_target}:/data/target:ro",
        docker_image,
        "bash", "-c", docker_cmd,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        print("    [kalibr] TIMEOUT", flush=True)
        return None

    if result.returncode != 0:
        print(f"    [kalibr] FAILED (rc={result.returncode})", flush=True)
        last_lines = result.stderr.strip().split("\n")[-10:]
        print("    stderr:", "\n    ".join(last_lines), flush=True)
        return None

    # Find output camchain YAML (Kalibr writes it as <bag_stem>-camchain.yaml)
    camchain_path = work_dir / (pathlib.Path(bag_name).stem + "-camchain.yaml")
    if not camchain_path.exists():
        # Sometimes written in cwd inside container → check alternatives
        candidates = list(work_dir.glob("*camchain*.yaml"))
        if not candidates:
            print("    [kalibr] camchain YAML not found", flush=True)
            return None
        camchain_path = candidates[0]

    return parse_camchain_yaml(camchain_path)


def parse_camchain_yaml(yaml_path: pathlib.Path) -> dict[str, Any] | None:
    """Parse Kalibr camchain YAML and return {fx,fy,cx,cy,k1,k2,p1,p2}."""
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text())
    except Exception:
        try:
            # Minimal parser if PyYAML unavailable
            text = yaml_path.read_text()
            data = _parse_camchain_minimal(text)
        except Exception as e:
            print(f"    [parse] YAML parse error: {e}", flush=True)
            return None

    cam0 = data.get("cam0", {})
    intr = cam0.get("intrinsics", [None]*4)
    dist = cam0.get("distortion_coeffs", [0.0]*4)
    if None in intr or len(intr) < 4:
        return None
    return {
        "fx": float(intr[0]), "fy": float(intr[1]),
        "cx": float(intr[2]), "cy": float(intr[3]),
        "k1": float(dist[0]) if len(dist) > 0 else 0.0,
        "k2": float(dist[1]) if len(dist) > 1 else 0.0,
        "p1": float(dist[2]) if len(dist) > 2 else 0.0,
        "p2": float(dist[3]) if len(dist) > 3 else 0.0,
    }


def _parse_camchain_minimal(text: str) -> dict:
    """Ultra-minimal YAML parser for the Kalibr camchain format."""
    import re
    intr_match = re.search(r"intrinsics:\s*\[([^\]]+)\]", text)
    dist_match = re.search(r"distortion_coeffs:\s*\[([^\]]+)\]", text)
    intr = [float(x) for x in intr_match.group(1).split(",")] if intr_match else []
    dist = [float(x) for x in dist_match.group(1).split(",")] if dist_match else []
    return {"cam0": {"intrinsics": intr, "distortion_coeffs": dist}}


# ---------------------------------------------------------------------------
# Held-out evaluation
# ---------------------------------------------------------------------------

def evaluate_heldout_kalibr(
    problem: fim.CalibrationProblem,
    heldout_indices: list[int],
    K_hat: dict[str, float],
) -> float:
    """solvePnP on held-out frames using Kalibr intrinsics; return mean RMS px."""
    K_mat = np.array([
        [K_hat["fx"], 0.0, K_hat["cx"]],
        [0.0, K_hat["fy"], K_hat["cy"]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    dist_mat = np.array(
        [[K_hat["k1"], K_hat["k2"], K_hat["p1"], K_hat["p2"], 0.0]], dtype=np.float64
    )
    errors: list[float] = []
    for idx in heldout_indices:
        meas = problem.measurements[idx]  # (N, 2) with NaN for invisible
        valid = np.isfinite(meas).all(axis=1)
        if valid.sum() < 6:
            continue
        obj = problem.target_points[valid].reshape(-1, 1, 3).astype(np.float64)
        img = meas[valid].reshape(-1, 1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K_mat, dist_mat)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K_mat, dist_mat)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2))**2, axis=1))))
        errors.append(rms)
    return float(np.mean(errors)) if errors else float("nan")


# ---------------------------------------------------------------------------
# Selection methods table
# ---------------------------------------------------------------------------

def _build_methods(problem: fim.CalibrationProblem, K: int, prior: np.ndarray) -> list:
    def _random_best(n: int = 10):
        rng = np.random.default_rng(0)
        best_idx, best_score = None, -np.inf
        for _ in range(n):
            idx = list(rng.choice(problem.num_candidates, K, replace=False))
            sel_vec = np.zeros(problem.num_candidates)
            sel_vec[idx] = 1.0
            from OASIS.FIM import construct_candidate_inf_blocks, compute_min_eig_score
            blocks = construct_candidate_inf_blocks(problem)
            score = compute_min_eig_score(problem, sel_vec, blocks, prior=prior)
            if score > best_score:
                best_score, best_idx = score, idx
        poses = [{"rotation_wc": problem.candidate_rotations[i],
                  "translation_wc": problem.candidate_translations[i],
                  "visible_points": 0} for i in best_idx]
        return poses, best_idx, best_score, np.zeros(problem.num_candidates)

    return [
        ("random",    lambda: _random_best()),
        ("coverage",  lambda: sel.coverage_selection(problem, K)),
        ("motion-div",lambda: sel.motion_diversity_selection(problem, K)),
        ("a-optimal", lambda: sel.greedy_selection_a_optimal(problem, K, prior=prior)),
        ("d-optimal", lambda: sel.greedy_selection_d_optimal(problem, K, prior=prior)),
        ("greedy-e",  lambda: sel.greedy_selection(problem, K, prior=prior)),
        ("fw-e",      lambda: sel.frank_wolfe_selection(problem, K, prior=prior)),
        ("fw-a",      lambda: sel.frank_wolfe_a_optimal_selection(problem, K, prior=prior)),
        ("fw-d",      lambda: sel.frank_wolfe_d_optimal_selection(problem, K, prior=prior)),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="results/kalibr_aprilgrid", type=pathlib.Path)
    parser.add_argument("--select-k", type=int, default=20)
    parser.add_argument("--held-out-count", type=int, default=20)
    parser.add_argument("--noise-sigma", type=float, default=0.5,
                        help="Pixel noise std-dev added to rendered images (default 0.5)")
    parser.add_argument("--docker-image", default="kalibr")
    parser.add_argument("--fim-backend", choices=("numeric", "gtsam"), default="numeric")
    parser.add_argument("--skip-render", action="store_true",
                        help="Reuse already-rendered images in output-dir/images_all/")
    args = parser.parse_args()

    fim.set_information_backend(args.fim_backend)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images_all"
    target_yaml = out_dir / "aprilgrid_target.yaml"

    # ------------------------------------------------------------------ #
    # 1. Generate hemisphere poses
    # ------------------------------------------------------------------ #
    print("Generating hemisphere poses...", flush=True)
    R_all, t_all = generate_hemisphere_poses()
    N = len(R_all)
    print(f"  N = {N} poses", flush=True)

    # ------------------------------------------------------------------ #
    # 2. Build target points and CalibrationProblem
    # ------------------------------------------------------------------ #
    target_pts = build_aprilgrid_world_points()
    print(f"  Target corners: {len(target_pts)}", flush=True)

    # Generate noisy corner measurements using fim.project_points
    rng = np.random.default_rng(42)
    measurements = np.full((N, len(target_pts), 2), np.nan, dtype=float)
    for i, (R_wc, t_wc) in enumerate(zip(R_all, t_all)):
        proj = fim.project_points(target_pts, R_wc, t_wc, _INTRINSICS)
        valid = (
            np.isfinite(proj).all(axis=1)
            & (proj[:, 0] >= 0) & (proj[:, 0] < _IMAGE_W)
            & (proj[:, 1] >= 0) & (proj[:, 1] < _IMAGE_H)
        )
        noisy = proj.copy()
        noisy[valid] += rng.normal(0, args.noise_sigma, (int(valid.sum()), 2))
        noisy[~valid] = np.nan
        measurements[i] = noisy

    intrinsics_init = _INTRINSICS + fim.default_intrinsics_init_offset(_INTRINSICS.size)
    problem = fim.CalibrationProblem(
        target_points=target_pts,
        candidate_rotations=R_all,
        candidate_translations=t_all,
        measurements=measurements,
        intrinsics_gt=_INTRINSICS,
        intrinsics_init=intrinsics_init,
        image_size=_IMAGE_SIZE,
        pixel_noise_sigma=args.noise_sigma,
    )

    # ------------------------------------------------------------------ #
    # 3. Held-out split (farthest-point sampling on translations)
    # ------------------------------------------------------------------ #
    heldout_indices = farthest_point_sample(t_all, args.held_out_count, seed=42)
    candidate_mask = np.ones(N, dtype=bool)
    candidate_mask[heldout_indices] = False
    candidate_indices = np.where(candidate_mask)[0].tolist()

    # Restrict problem to candidate pool only
    problem_cand = fim.CalibrationProblem(
        target_points=target_pts,
        candidate_rotations=R_all[candidate_mask],
        candidate_translations=t_all[candidate_mask],
        measurements=measurements[candidate_mask],
        intrinsics_gt=_INTRINSICS,
        intrinsics_init=intrinsics_init,
        image_size=_IMAGE_SIZE,
        pixel_noise_sigma=args.noise_sigma,
    )

    print(f"  Held-out: {len(heldout_indices)}, candidates: {len(candidate_indices)}", flush=True)

    # ------------------------------------------------------------------ #
    # 4. Render images
    # ------------------------------------------------------------------ #
    _write_target_yaml(target_yaml)
    print(f"  Wrote target YAML: {target_yaml}", flush=True)

    texture_path = out_dir / "aprilgrid_texture.png"
    if not texture_path.exists() or not args.skip_render:
        ensure_aprilgrid_texture(
            texture_path, rows=_ROWS, cols=_COLS,
            tag_size=_TAG_SIZE, tag_spacing=_TAG_SPACING,
            pixels_per_tag_pitch=_PPP,
        )
    texture_gray = cv2.imread(str(texture_path), cv2.IMREAD_GRAYSCALE)
    assert texture_gray is not None, f"Failed to load texture: {texture_path}"

    images_dir.mkdir(parents=True, exist_ok=True)
    render_rng = np.random.default_rng(7)

    rendered_paths: list[pathlib.Path | None] = []
    if not args.skip_render:
        print(f"Rendering {N} synthetic images...", flush=True)
        n_ok = 0
        for i, (R_wc, t_wc) in enumerate(zip(R_all, t_all)):
            img = render_image(texture_gray, R_wc, t_wc, target_pts, render_rng, args.noise_sigma)
            img_path = images_dir / f"{i:08d}.png"
            if img is not None:
                cv2.imwrite(str(img_path), img)
                rendered_paths.append(img_path)
                n_ok += 1
            else:
                rendered_paths.append(None)
        print(f"  Rendered {n_ok}/{N} images successfully", flush=True)
    else:
        print("Skipping render (--skip-render), checking existing images...", flush=True)
        for i in range(N):
            p = images_dir / f"{i:08d}.png"
            rendered_paths.append(p if p.exists() else None)
        n_ok = sum(1 for p in rendered_paths if p is not None)
        print(f"  Found {n_ok}/{N} existing images", flush=True)

    # ------------------------------------------------------------------ #
    # 5. Run selection methods and Kalibr for each
    # ------------------------------------------------------------------ #
    prior = fim.build_prior_blocks(problem_cand)
    methods = _build_methods(problem_cand, args.select_k, prior)

    results = []
    for method_name, method_fn in methods:
        print(f"\n[{method_name}]", flush=True)
        t0 = time.time()
        try:
            _, cand_sel_indices, score, *_ = method_fn()
        except Exception as e:
            print(f"  selection FAILED: {e}", flush=True)
            results.append({
                "method": method_name, "selected_global_indices": [],
                "time_selection_s": float("nan"), "kalibr_success": False,
                "heldout_rms": float("nan"),
            })
            continue
        t_sel = time.time() - t0

        # Map candidate-pool indices back to global indices
        cand_arr = np.array(candidate_indices)
        global_sel_indices = cand_arr[np.array(cand_sel_indices)].tolist()
        print(f"  Selected {len(global_sel_indices)} poses in {t_sel:.1f}s  score={score:.4e}", flush=True)

        # --- prepare image folder for Kalibr ---
        method_work = out_dir / method_name
        method_img_dir = method_work / "cam0"
        method_img_dir.mkdir(parents=True, exist_ok=True)

        # Clear old images
        for f in method_img_dir.glob("*.png"):
            f.unlink()

        n_copied = 0
        for seq, gidx in enumerate(global_sel_indices):
            src = rendered_paths[gidx]
            if src is None or not src.exists():
                print(f"  [warn] image {gidx} not rendered, skipping", flush=True)
                continue
            # Timestamp-named file (Kalibr bagcreater sorts by filename)
            dst = method_img_dir / f"{seq:010d}.png"
            shutil.copy(src, dst)
            n_copied += 1

        print(f"  Copied {n_copied} images for Kalibr", flush=True)
        if n_copied < 5:
            print("  Too few images, skipping Kalibr", flush=True)
            results.append({
                "method": method_name,
                "selected_global_indices": global_sel_indices,
                "time_selection_s": t_sel,
                "kalibr_success": False,
                "heldout_rms": float("nan"),
            })
            continue

        # --- run Kalibr ---
        print("  Running Kalibr...", flush=True)
        t_kal0 = time.time()
        K_hat = run_kalibr(
            image_dir=method_work,
            work_dir=method_work,
            target_yaml=target_yaml,
            docker_image=args.docker_image,
        )
        t_kal = time.time() - t_kal0

        if K_hat is None:
            print(f"  Kalibr FAILED ({t_kal:.0f}s)", flush=True)
            results.append({
                "method": method_name,
                "selected_global_indices": global_sel_indices,
                "time_selection_s": t_sel,
                "time_kalibr_s": t_kal,
                "kalibr_success": False,
                "heldout_rms": float("nan"),
            })
            continue

        print(f"  Kalibr OK ({t_kal:.0f}s)  fx={K_hat['fx']:.1f}  k1={K_hat['k1']:.4f}", flush=True)

        # --- evaluate held-out ---
        heldout_rms = evaluate_heldout_kalibr(problem, heldout_indices, K_hat)
        print(f"  Held-out RMS = {heldout_rms:.4f} px", flush=True)

        # --- intrinsic errors relative to GT ---
        fx_err = abs(K_hat["fx"] - _FX)
        fy_err = abs(K_hat["fy"] - _FY)
        cx_err = abs(K_hat["cx"] - _CX)
        cy_err = abs(K_hat["cy"] - _CY)
        focal_err = (fx_err + fy_err) / 2.0
        pp_err = math.sqrt(cx_err**2 + cy_err**2)

        results.append({
            "method": method_name,
            "selected_global_indices": global_sel_indices,
            "time_selection_s": round(t_sel, 3),
            "time_kalibr_s": round(t_kal, 1),
            "kalibr_success": True,
            "kalibr_intrinsics": K_hat,
            "focal_err_px": round(focal_err, 3),
            "pp_err_px": round(pp_err, 3),
            "heldout_rms": round(heldout_rms, 4),
        })

    # ------------------------------------------------------------------ #
    # 6. Save results
    # ------------------------------------------------------------------ #
    summary = {
        "select_k": args.select_k,
        "held_out_count": args.held_out_count,
        "num_candidates": len(candidate_indices),
        "noise_sigma": args.noise_sigma,
        "board": {
            "rows": _ROWS, "cols": _COLS,
            "tag_size_m": _TAG_SIZE, "tag_spacing": _TAG_SPACING,
        },
        "gt_intrinsics": {
            "fx": _FX, "fy": _FY, "cx": _CX, "cy": _CY,
        },
        "results": results,
    }

    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {json_path}", flush=True)

    # CSV summary
    csv_path = out_dir / "benchmark_results.csv"
    fieldnames = ["method", "kalibr_success", "focal_err_px", "pp_err_px",
                  "heldout_rms", "time_selection_s", "time_kalibr_s"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {csv_path}", flush=True)

    # Print table
    print("\n" + "="*72)
    print(f"{'Method':<14}  {'HO-RMS':>8}  {'focal-err':>10}  {'pp-err':>8}  {'Kalibr':>7}")
    print("-"*72)
    for r in results:
        ok = "OK" if r.get("kalibr_success") else "FAIL"
        print(
            f"{r['method']:<14}  "
            f"{r.get('heldout_rms', float('nan')):8.4f}  "
            f"{r.get('focal_err_px', float('nan')):10.3f}  "
            f"{r.get('pp_err_px', float('nan')):8.3f}  "
            f"{ok:>7}"
        )
    print("="*72)


if __name__ == "__main__":
    main()
