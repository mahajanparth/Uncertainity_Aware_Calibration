"""Compare view-selection methods using Kalibr as the downstream calibration backend.

Uses the existing Isaac Sim AprilGrid capture session:
  - 600 candidate hemisphere poses (all_candidate_problem.npz)
  - 100 captured images with confirmed Kalibr detection
  - Kalibr-generated AprilGrid target (april_6x6.yaml)

Pipeline
--------
1. Load 600-candidate CalibrationProblem + image paths for the 100 captured views.
2. Reserve 20 held-out views (farthest-point sampling on translations).
3. For each selection method → K=20 indices from the remaining 80 captured candidates:
   a. Write images into cam0/ subfolder for kalibr_bagcreater.
   b. docker run kalibr  (bagcreater + kalibr_calibrate_cameras --mi-tol -1).
   c. Parse <bag>-camchain.yaml for intrinsics and distortion.
   d. Evaluate held-out RMS with solvePnP(K_hat, dist_hat) on held-out images.
4. Save benchmark_results.json and summary CSV.

Usage
-----
    python3 -m Experiments.main_kalibr_isaacsim \\
        --session-dir /path/to/isaac_outputs/interactive_run_aprilgrid_.../ \\
        --output-dir results/kalibr_isaacsim \\
        [--select-k 20] [--held-out-count 20] [--docker-image kalibr]
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
from OASIS import methods as sel

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_session(session_dir: pathlib.Path) -> tuple:
    """Return (problem_all, captured_list, target_yaml).

    captured_list: list of (candidate_idx_in_all_problem, image_path)
    problem_all: 600-candidate CalibrationProblem

    Image resolution strategy (in order):
    1. image_path as stored in summary (may be absolute from original machine)
    2. session_dir/captures/<basename>
    3. session_dir/extracted_frames/frame_NNNN.png (extracted from bag by order)
    """
    summary_path = session_dir / "interactive_summary.json"
    summary = json.loads(summary_path.read_text())

    problem_path = session_dir / "all_candidate_problem.npz"
    problem_all = fim.load_calibration_problem_npz(str(problem_path))

    # selected_indices[N] = global candidate_idx for bag frame N
    selected_indices = summary.get("selected_indices", [])

    # Check if we have pre-extracted frames from the bag
    extracted_dir = session_dir / "extracted_frames"
    extracted_frames: list[pathlib.Path | None] = []
    if extracted_dir.exists():
        for i in range(len(selected_indices)):
            p = extracted_dir / f"frame_{i:04d}.png"
            extracted_frames.append(p if p.exists() else None)

    def _resolve(cidx: int, history_path: str, bag_order: int) -> pathlib.Path | None:
        # Prefer extracted frames (bag-extracted, guaranteed grayscale) over captures/
        if bag_order < len(extracted_frames) and extracted_frames[bag_order] is not None:
            return extracted_frames[bag_order]
        p = pathlib.Path(history_path)
        if p.exists():
            return p
        alt = session_dir / "captures" / p.name
        if alt.exists():
            return alt
        return None

    # Build ordered capture list from selected_indices + history
    # history order matches bag order
    bag_order = 0
    history_entries: list[tuple[int, str]] = []
    for entry in summary.get("history", []):
        if entry.get("step_type") == "seed":
            history_entries.append((entry["candidate_index"], entry.get("image_path", "")))
        elif entry.get("step_type") == "recommended_batch":
            for cap in entry.get("captures", []):
                history_entries.append((cap["candidate_index"], cap.get("image_path", "")))

    captured: list[tuple[int, pathlib.Path]] = []
    for bag_ord, (cidx, ipath_str) in enumerate(history_entries):
        resolved = _resolve(cidx, ipath_str, bag_ord)
        if resolved is not None:
            captured.append((cidx, resolved))
        else:
            print(f"  [warn] image not found for candidate {cidx} (bag frame {bag_ord})", flush=True)

    target_yaml = session_dir / "april_6x6.yaml"
    if not target_yaml.exists():
        cands = list(session_dir.glob("*.yaml"))
        target_yaml = cands[0] if cands else None

    return problem_all, captured, target_yaml


def extract_bag_frames(session_dir: pathlib.Path, docker_image: str = "kalibr") -> bool:
    """Extract all images from session bag to session_dir/extracted_frames/.

    Returns True on success.
    """
    bag_path = session_dir / "cam_april.bag"
    if not bag_path.exists():
        bags = list(session_dir.glob("*.bag"))
        if not bags:
            return False
        bag_path = bags[0]

    out_dir = session_dir / "extracted_frames"
    if out_dir.exists() and len(list(out_dir.glob("*.png"))) > 0:
        return True  # already extracted

    out_dir.mkdir(exist_ok=True)
    print(f"  Extracting bag frames to {out_dir} ...", flush=True)

    cmd = [
        "docker", "run", "--rm",
        "--entrypoint", "/ros_entrypoint.sh",
        "-v", f"{session_dir.resolve()}:/data/session:ro",
        "-v", f"{out_dir.resolve()}:/data/out:rw",
        docker_image, "bash", "-c",
        "source /catkin_ws/devel/setup.bash && "
        "python3 -c \""
        "import rosbag, cv2, numpy as np, os\n"
        "os.makedirs('/data/out', exist_ok=True)\n"
        f"bag_name = '{bag_path.name}'\n"
        "with rosbag.Bag('/data/session/' + bag_name) as bag:\n"
        "    for seq,(t,msg,ts) in enumerate(bag.read_messages(topics=['/cam0/image_raw'])):\n"
        "        arr = np.frombuffer(msg.data,dtype=np.uint8).reshape(msg.height,msg.width)\n"
        "        cv2.imwrite(f'/data/out/frame_{seq:04d}.png', arr)\n"
        "print('Extracted', seq+1, 'frames')\n"
        "\"",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    if result.returncode != 0:
        print(f"  [warn] bag extraction failed: {result.stderr[-200:]}", flush=True)
        return False
    print(f"  Extracted frames: {len(list(out_dir.glob('*.png')))}", flush=True)
    return True


# ---------------------------------------------------------------------------
# Farthest-point sampling for held-out split
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

def run_kalibr(
    images_root: pathlib.Path,   # host dir containing cam0/ with selected images
    work_dir: pathlib.Path,
    target_yaml: pathlib.Path,
    docker_image: str = "kalibr",
    timeout: int = 600,
    init_focal_px: float | None = None,
) -> dict[str, Any] | None:
    """Run Kalibr in Docker. Returns parsed intrinsics dict or None on failure."""
    work_dir.mkdir(parents=True, exist_ok=True)

    host_images = str(images_root.resolve())
    host_work = str(work_dir.resolve())
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
        "--no-outliers-removal "
        "--no-final-filtering"
    )

    # KALIBR_MANUAL_FOCAL_LENGTH_INIT enables stdin-based focal length input.
    # When Kalibr's DLT init fails it reads f from std::cin; we pipe it in.
    stdin_input: str | None = None
    env_flags: list[str] = []
    if init_focal_px is not None:
        env_flags = ["-e", "KALIBR_MANUAL_FOCAL_LENGTH_INIT=1", "-i"]
        # Repeat enough times to cover multiple cameras / retry attempts
        stdin_input = "\n".join([f"{init_focal_px:.1f}"] * 10) + "\n"

    cmd = [
        "docker", "run", "--rm",
        "--entrypoint", "/ros_entrypoint.sh",
        *env_flags,
        "-v", f"{host_images}:/data/images:ro",
        "-v", f"{host_work}:/data/work:rw",
        "-v", f"{host_target}:/data/target:ro",
        docker_image,
        "bash", "-c", docker_cmd,
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            input=stdin_input,
        )
    except subprocess.TimeoutExpired:
        print("    [kalibr] TIMEOUT", flush=True)
        return None

    # Kalibr exits non-zero only on the PDF report step (matplotlib crash) —
    # the camchain.yaml is written before that, so check for it regardless.
    camchain_path = work_dir / (pathlib.Path(bag_name).stem + "-camchain.yaml")
    if not camchain_path.exists():
        candidates = sorted(work_dir.glob("*camchain*.yaml"))
        if candidates:
            camchain_path = candidates[0]
        else:
            # No output at all — real failure
            print(f"    [kalibr] FAILED (rc={result.returncode}, no output)", flush=True)
            last = result.stderr.strip().split("\n")[-8:]
            print("    stderr:", "\n    ".join(last), flush=True)
            return None

    parsed = _parse_camchain(camchain_path)
    if parsed is None or any(not np.isfinite(v) for v in parsed.values()):
        # Calibration produced NaN intrinsics — report the last stderr lines
        last = result.stderr.strip().split("\n")[-5:]
        print(f"    [kalibr] NaN intrinsics (rc={result.returncode})", flush=True)
        print("    stderr:", "\n    ".join(last), flush=True)
        return None

    return parsed


def _parse_camchain(yaml_path: pathlib.Path) -> dict[str, Any] | None:
    try:
        import yaml
        data = yaml.safe_load(yaml_path.read_text())
    except Exception:
        try:
            import re
            text = yaml_path.read_text()
            intr = re.search(r"intrinsics:\s*\[([^\]]+)\]", text)
            dist = re.search(r"distortion_coeffs:\s*\[([^\]]+)\]", text)
            intr_v = [float(x) for x in intr.group(1).split(",")] if intr else []
            dist_v = [float(x) for x in dist.group(1).split(",")] if dist else []
            data = {"cam0": {"intrinsics": intr_v, "distortion_coeffs": dist_v}}
        except Exception as e:
            print(f"    [parse] failed: {e}", flush=True)
            return None

    cam0 = data.get("cam0", {})
    intr = cam0.get("intrinsics", [])
    dist = cam0.get("distortion_coeffs", [0.0] * 4)
    if len(intr) < 4:
        return None
    return {
        "fx": float(intr[0]), "fy": float(intr[1]),
        "cx": float(intr[2]), "cy": float(intr[3]),
        "k1": float(dist[0]) if len(dist) > 0 else 0.0,
        "k2": float(dist[1]) if len(dist) > 1 else 0.0,
        "p1": float(dist[2]) if len(dist) > 2 else 0.0,
        "p2": float(dist[3]) if len(dist) > 3 else 0.0,
    }


# ---------------------------------------------------------------------------
# Held-out evaluation using Kalibr intrinsics
# ---------------------------------------------------------------------------

def evaluate_heldout(
    problem: fim.CalibrationProblem,
    heldout_global_indices: list[int],
    K_hat: dict[str, float],
) -> float:
    """solvePnP on held-out frames with Kalibr intrinsics; return mean RMS (px)."""
    K_mat = np.array([
        [K_hat["fx"], 0.0, K_hat["cx"]],
        [0.0,  K_hat["fy"], K_hat["cy"]],
        [0.0,  0.0,          1.0],
    ], dtype=np.float64)
    dist_mat = np.array(
        [[K_hat["k1"], K_hat["k2"], K_hat["p1"], K_hat["p2"], 0.0]],
        dtype=np.float64
    )
    errors: list[float] = []
    for idx in heldout_global_indices:
        meas = problem.measurements[idx]
        valid = np.isfinite(meas).all(axis=1)
        if valid.sum() < 6:
            continue
        obj = problem.target_points[valid].reshape(-1, 1, 3).astype(np.float64)
        img = meas[valid].reshape(-1, 1, 2).astype(np.float64)
        ok, rvec, tvec = cv2.solvePnP(obj, img, K_mat, dist_mat)
        if not ok:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, K_mat, dist_mat)
        rms = float(np.sqrt(np.mean(
            np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1)
        )))
        errors.append(rms)
    return float(np.mean(errors)) if errors else float("nan")


# ---------------------------------------------------------------------------
# Selection method wrappers
# ---------------------------------------------------------------------------

def _build_methods(problem: fim.CalibrationProblem, K: int, prior: np.ndarray) -> list:
    def _random_best(n_trials: int = 25):
        rng = np.random.default_rng(0)
        best_idx, best_score = None, -np.inf
        blocks = fim.construct_candidate_inf_blocks(problem)
        for _ in range(n_trials):
            idx = list(rng.choice(problem.num_candidates, K, replace=False))
            sel_vec = np.zeros(problem.num_candidates)
            sel_vec[idx] = 1.0
            score = fim.compute_min_eig_score(problem, sel_vec, blocks, prior=prior)
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
        ("fw-e",      lambda: sel.frank_wolfe_selection(problem, K, prior=prior)
                              if K < problem.num_candidates else
                              sel.greedy_selection(problem, K, prior=prior)),
        ("fw-a",      lambda: sel.frank_wolfe_a_optimal_selection(problem, K, prior=prior)
                              if K < problem.num_candidates else
                              sel.greedy_selection_a_optimal(problem, K, prior=prior)),
        ("fw-d",      lambda: sel.frank_wolfe_d_optimal_selection(problem, K, prior=prior)
                              if K < problem.num_candidates else
                              sel.greedy_selection_d_optimal(problem, K, prior=prior)),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-dir",
        default="isaac_outputs/interactive_run_aprilgrid_kalibr_pdf_docker_100_noiseless",
        type=pathlib.Path,
    )
    parser.add_argument("--output-dir", default="results/kalibr_isaacsim", type=pathlib.Path)
    parser.add_argument("--select-k", type=int, default=20)
    parser.add_argument("--held-out-count", type=int, default=20)
    parser.add_argument("--docker-image", default="kalibr")
    parser.add_argument("--fim-backend", choices=("numeric", "gtsam"), default="numeric")
    parser.add_argument(
        "--init-focal-px", type=float, default=None,
        help="Initial focal length (px) passed via KALIBR_MANUAL_FOCAL_LENGTH_INIT. "
             "If omitted, derived from GT intrinsics when available.",
    )
    parser.add_argument(
        "--noise-sigma", type=float, default=0.0,
        help="Gaussian pixel noise std-dev added to images before Kalibr (0 = noiseless). "
             "Also sets pixel_noise_sigma in the FIM problem.",
    )
    args = parser.parse_args()

    fim.set_information_backend(args.fim_backend)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve session dir
    session_dir = args.session_dir
    if not session_dir.exists():
        session_dir = REPO_ROOT / args.session_dir
    if not session_dir.exists():
        # Try absolute Desktop path
        session_dir = pathlib.Path("/home/parth/Desktop/isaac_outputs") / args.session_dir.name
    if not session_dir.exists():
        raise FileNotFoundError(f"Session dir not found: {args.session_dir}")

    print(f"Session: {session_dir}", flush=True)

    # ------------------------------------------------------------------ #
    # 1. Load data  (extract bag frames first if needed)
    # ------------------------------------------------------------------ #
    extract_bag_frames(session_dir, docker_image=args.docker_image)
    problem_all, captured_list, target_yaml = load_session(session_dir)
    N_all = problem_all.num_candidates
    N_cap = len(captured_list)
    noise_sigma = args.noise_sigma
    print(f"Problem: {N_all} candidates total, {N_cap} captured with images", flush=True)
    print(f"GT intrinsics: fx={problem_all.intrinsics_gt[0]:.1f}", flush=True)
    print(f"Pixel noise sigma: {noise_sigma}", flush=True)

    # Map captured views: local pool index → (global_idx, image_path)
    cap_global_indices = [c[0] for c in captured_list]  # global idx in problem_all
    cap_image_paths = [c[1] for c in captured_list]

    # Verify image files exist
    missing = [str(p) for p in cap_image_paths if not p.exists()]
    if missing:
        print(f"  [warn] {len(missing)} images not found, e.g. {missing[0]}")
        # Filter to only those with images
        captured_list = [(g, p) for g, p in captured_list if p.exists()]
        cap_global_indices = [c[0] for c in captured_list]
        cap_image_paths = [c[1] for c in captured_list]
        N_cap = len(captured_list)
        print(f"  Proceeding with {N_cap} images", flush=True)

    # Sub-problem restricted to captured views only
    cap_arr = np.array(cap_global_indices)
    _sigma = noise_sigma if noise_sigma > 0.0 else problem_all.pixel_noise_sigma
    problem_cap = fim.CalibrationProblem(
        target_points=problem_all.target_points,
        candidate_rotations=problem_all.candidate_rotations[cap_arr],
        candidate_translations=problem_all.candidate_translations[cap_arr],
        measurements=problem_all.measurements[cap_arr],
        intrinsics_gt=problem_all.intrinsics_gt,
        intrinsics_init=problem_all.intrinsics_init,
        image_size=problem_all.image_size,
        pixel_noise_sigma=_sigma,
    )

    # ------------------------------------------------------------------ #
    # 2. Held-out split (farthest-point sampling on translations)
    # ------------------------------------------------------------------ #
    translations_cap = problem_all.candidate_translations[cap_arr]
    heldout_local = farthest_point_sample(translations_cap, args.held_out_count, seed=42)
    heldout_global = [cap_global_indices[i] for i in heldout_local]

    cand_local_mask = np.ones(N_cap, dtype=bool)
    cand_local_mask[heldout_local] = False
    cand_local_indices = np.where(cand_local_mask)[0].tolist()

    print(f"Held-out: {len(heldout_local)}, candidates: {len(cand_local_indices)}", flush=True)

    # Sub-problem restricted to candidate (non-held-out) views
    problem_cand = fim.CalibrationProblem(
        target_points=problem_all.target_points,
        candidate_rotations=problem_all.candidate_rotations[cap_arr[cand_local_mask]],
        candidate_translations=problem_all.candidate_translations[cap_arr[cand_local_mask]],
        measurements=problem_all.measurements[cap_arr[cand_local_mask]],
        intrinsics_gt=problem_all.intrinsics_gt,
        intrinsics_init=problem_all.intrinsics_init,
        image_size=problem_all.image_size,
        pixel_noise_sigma=_sigma,
    )

    # ------------------------------------------------------------------ #
    # 3. Run selection methods and Kalibr for each
    # ------------------------------------------------------------------ #
    prior = fim.build_prior_blocks(problem_cand)
    methods = _build_methods(problem_cand, args.select_k, prior)
    GT = problem_all.intrinsics_gt

    # Initial focal length for Kalibr: use CLI override, else mean of GT fx/fy
    init_focal_px = args.init_focal_px
    if init_focal_px is None and GT is not None and len(GT) >= 2:
        init_focal_px = float((GT[0] + GT[1]) / 2.0)
    print(f"Kalibr focal-length init: {init_focal_px:.1f} px", flush=True)

    results = []
    for method_name, method_fn in methods:
        print(f"\n[{method_name}]", flush=True)
        t0 = time.time()
        try:
            _, cand_pool_sel_indices, score, *_ = method_fn()
        except Exception as e:
            print(f"  selection FAILED: {e}", flush=True)
            results.append({"method": method_name, "kalibr_success": False,
                            "heldout_rms": float("nan"), "time_selection_s": float("nan")})
            continue
        t_sel = time.time() - t0

        # Map: cand_pool indices → local indices in cap list → global indices
        # cand_pool_sel_indices are indices into problem_cand (which is cand_local_indices)
        selected_local_in_cand_pool = list(cand_pool_sel_indices)  # idx into problem_cand
        selected_local_indices = [cand_local_indices[i] for i in selected_local_in_cand_pool]
        selected_global_indices = [cap_global_indices[i] for i in selected_local_indices]

        print(f"  Selected {len(selected_local_indices)} poses in {t_sel:.1f}s  "
              f"score={score:.4e}", flush=True)

        # --- prepare images for Kalibr ---
        method_dir = out_dir / method_name
        cam0_dir = method_dir / "cam0"
        cam0_dir.mkdir(parents=True, exist_ok=True)
        for f in cam0_dir.glob("*.png"):
            f.unlink()

        rng_noise = np.random.default_rng(42)
        n_copied = 0
        for seq, loc_idx in enumerate(selected_local_indices):
            src = cap_image_paths[loc_idx]
            if not src.exists():
                print(f"  [warn] missing image: {src}", flush=True)
                continue
            # kalibr_bagcreater reads filename as nanosecond timestamp.
            # Use 100ms spacing so frames aren't merged under the 20ms sync window.
            ts_ns = seq * 100_000_000  # 100 ms per frame
            dst = cam0_dir / f"{ts_ns:019d}.png"
            # Always write as single-channel 8-bit grayscale; Kalibr's AprilTag
            # detector requires mono images — RGB PNGs cause detection failure.
            img = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"  [warn] could not read: {src}", flush=True)
                continue
            if noise_sigma > 0.0:
                noisy = img.astype(np.float32) + rng_noise.normal(
                    0.0, noise_sigma, img.shape).astype(np.float32)
                img = np.clip(noisy, 0, 255).astype(np.uint8)
            cv2.imwrite(str(dst), img)
            n_copied += 1

        print(f"  Copied {n_copied} images", flush=True)
        if n_copied < 5:
            results.append({"method": method_name, "kalibr_success": False,
                            "heldout_rms": float("nan"),
                            "time_selection_s": round(t_sel, 3),
                            "selected_global_indices": selected_global_indices})
            continue

        # --- run Kalibr ---
        print("  Running Kalibr...", flush=True)
        work_dir = out_dir / f"{method_name}_work"
        t_kal0 = time.time()
        K_hat = run_kalibr(method_dir, work_dir, target_yaml,
                           docker_image=args.docker_image,
                           init_focal_px=init_focal_px)
        t_kal = time.time() - t_kal0

        if K_hat is None:
            print(f"  Kalibr FAILED ({t_kal:.0f}s)", flush=True)
            results.append({
                "method": method_name,
                "selected_global_indices": selected_global_indices,
                "time_selection_s": round(t_sel, 3),
                "time_kalibr_s": round(t_kal, 1),
                "kalibr_success": False,
                "heldout_rms": float("nan"),
            })
            continue

        print(f"  Kalibr OK ({t_kal:.0f}s)  "
              f"fx={K_hat['fx']:.2f}  fy={K_hat['fy']:.2f}  "
              f"k1={K_hat['k1']:.4f}", flush=True)

        # --- evaluate held-out RMS ---
        heldout_rms = evaluate_heldout(problem_all, heldout_global, K_hat)
        print(f"  Held-out RMS = {heldout_rms:.4f} px", flush=True)

        # --- intrinsic errors vs GT ---
        fx_err = abs(K_hat["fx"] - GT[0])
        fy_err = abs(K_hat["fy"] - GT[1])
        cx_err = abs(K_hat["cx"] - GT[2])
        cy_err = abs(K_hat["cy"] - GT[3])
        focal_err = (fx_err + fy_err) / 2.0
        pp_err = math.sqrt(cx_err**2 + cy_err**2)

        results.append({
            "method": method_name,
            "selected_global_indices": selected_global_indices,
            "time_selection_s": round(t_sel, 3),
            "time_kalibr_s": round(t_kal, 1),
            "kalibr_success": True,
            "kalibr_intrinsics": K_hat,
            "focal_err_px": round(focal_err, 3),
            "pp_err_px": round(pp_err, 3),
            "heldout_rms": round(heldout_rms, 4),
        })

    # ------------------------------------------------------------------ #
    # 4. Save results
    # ------------------------------------------------------------------ #
    summary = {
        "session_dir": str(session_dir),
        "select_k": args.select_k,
        "held_out_count": args.held_out_count,
        "noise_sigma": noise_sigma,
        "num_captured": N_cap,
        "num_candidates": len(cand_local_indices),
        "gt_intrinsics": {k: float(v) for k, v in zip(
            ["fx","fy","cx","cy","k1","k2","p1","p2","k3"], GT)},
        "results": results,
    }

    json_path = out_dir / "benchmark_results.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {json_path}", flush=True)

    csv_path = out_dir / "benchmark_results.csv"
    fieldnames = ["method", "kalibr_success", "focal_err_px", "pp_err_px",
                  "heldout_rms", "time_selection_s", "time_kalibr_s"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"Saved: {csv_path}", flush=True)

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
