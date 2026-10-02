"""Add FW-A and FW-D results to every existing benchmark JSON.

Reads each benchmark_results.json, skips methods already present,
runs the missing FW variants, appends their results, and saves in-place.

Usage:
    python3 -m Experiments.patch_fw_variants [--results-dir results/k20]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal
from OASIS import methods as sel


# ---------------------------------------------------------------------------
# Helpers (mirror run_all_k20.py)
# ---------------------------------------------------------------------------

def _fim_metrics(problem, indices, prior):
    try:
        return cal.evaluate_selection(problem, indices, prior=prior)
    except Exception as e:
        print(f"    [warn] FIM metrics failed: {e}")
        return {"min_eig": float("nan"), "trace_cov": float("nan"),
                "logdet": float("nan"), "cond": float("nan"), "visible_points": 0.0}


def _calib_metrics(problem, indices):
    result = cal.calibrate_opencv_full(problem, indices)
    if not result["success"]:
        nan = float("nan")
        return {"focal_err_px": nan, "pp_err_px": nan, "dist_err": nan,
                "rot_err_deg": nan, "trans_err_cm": nan,
                "train_rms": nan, "heldout_rms": nan}
    grouped = cal.compute_groupwise_errors(
        problem, result["intrinsics"], result["valid_indices"],
        result["rvecs"], result["tvecs"],
    )
    dist_mat = result["dist"][:5].reshape(1, 5)
    heldout_rms = cal.heldout_reprojection_error(
        problem, indices, result["K_mat"], dist_mat
    )
    return {**grouped, "train_rms": result["train_rms"], "heldout_rms": heldout_rms}


def _serialise(v):
    if isinstance(v, np.integer): return int(v)
    if isinstance(v, np.floating): return float(v)
    if isinstance(v, np.ndarray): return v.tolist()
    if isinstance(v, list) and v and isinstance(v[0], (np.integer, np.floating)):
        return [int(x) if isinstance(x, np.integer) else float(x) for x in v]
    return v


NEW_METHODS = [
    ("fw-a", lambda prob, K, prior: sel.frank_wolfe_a_optimal_selection(prob, K, prior=prior)),
    ("fw-d", lambda prob, K, prior: sel.frank_wolfe_d_optimal_selection(prob, K, prior=prior)),
]


def patch_file(json_path: pathlib.Path, problem_cache: dict) -> None:
    raw = json.loads(json_path.read_text())
    existing_methods = {r["method"] for r in raw["results"]}
    missing = [(name, fn) for name, fn in NEW_METHODS if name not in existing_methods]
    if not missing:
        print(f"  [skip] all methods present: {json_path.parent.name}", flush=True)
        return

    prob_path = raw.get("problem", "")
    if prob_path not in problem_cache:
        p = pathlib.Path(prob_path)
        if not p.exists():
            print(f"  [error] problem not found: {prob_path}", flush=True)
            return
        problem_cache[prob_path] = fim.load_calibration_problem_npz(str(p))
    problem = problem_cache[prob_path]
    prior = fim.build_prior_blocks(problem)
    K = raw["select_k"]

    for name, fn in missing:
        print(f"  [{name}] running...", flush=True)
        t0 = time.time()
        try:
            selected_poses, selected_indices, best_score, selection, *_ = fn(problem, K, prior)
        except Exception as e:
            print(f"    FAILED: {e}", flush=True)
            raw["results"].append({"method": name, "selected_indices": [],
                                   "time_s": float("nan"), "min_eig": float("nan"),
                                   "logdet": float("nan"), "focal_err_px": float("nan"),
                                   "pp_err_px": float("nan"), "dist_err": float("nan"),
                                   "train_rms": float("nan"), "heldout_rms": float("nan")})
            continue
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.1f}s  score={best_score:.4e}", flush=True)

        r = {"method": name, "selected_indices": list(selected_indices), "time_s": elapsed}
        r.update(_fim_metrics(problem, selected_indices, prior))
        r.update(_calib_metrics(problem, selected_indices))
        raw["results"].append({k: _serialise(v) for k, v in r.items()})

    json_path.write_text(json.dumps(raw, indent=2))
    print(f"  → saved {json_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/k20", type=pathlib.Path)
    args = parser.parse_args()

    fim.set_information_backend("numeric")

    pattern = str(args.results_dir / "*" / "benchmark_*" / "benchmark_results.json")
    json_files = sorted(glob.glob(pattern))
    print(f"Found {len(json_files)} benchmark JSON files to patch.\n")

    problem_cache: dict = {}
    for jf in json_files:
        config_name = pathlib.Path(jf).parent.parent.name
        print(f"\n[{config_name}]", flush=True)
        patch_file(pathlib.Path(jf), problem_cache)

    print("\nDone.")


if __name__ == "__main__":
    main()
