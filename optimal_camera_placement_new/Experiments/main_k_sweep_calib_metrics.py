"""K-budget sweep with full calibration metrics. Held-out = all N-K unselected frames.

For each K in {5,10,15,20,30,40,50} and each of the 9 selection methods,
this script:
  1. Selects K frames from all N candidates.
  2. Runs cv2.calibrateCamera on the K selected frames.
  3. Evaluates reprojection RMS on the remaining N-K frames (held-out).
  4. Reports: min_eig, logdet, ef, epp, ed, train_rms, heldout_rms

Usage:
  python3 -m Experiments.main_k_sweep_calib_metrics \\
      --problem results/k20/datasets/aprilgrid_zero/all_candidate_problem.npz \\
      --tag aprilgrid_zero \\
      --select-ks 5 10 15 20 30 40 50 \\
      --output-dir results/k_sweep_calib_nk
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime
from typing import Any, Dict, List

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal
from OASIS import methods as sel


# ---------------------------------------------------------------------------
# Single (K, method) evaluation
# ---------------------------------------------------------------------------

def _evaluate(
    problem: fim.CalibrationProblem,
    local_indices: List[int],
    prior,
) -> Dict[str, float]:
    fim_m = cal.evaluate_selection(problem, local_indices, prior=prior)
    result = cal.calibrate_opencv_full(problem, local_indices)
    if not result["success"]:
        nan = float("nan")
        return {
            **fim_m,
            "focal_err_px": nan, "pp_err_px": nan, "dist_err": nan,
            "train_rms": nan, "heldout_rms": nan,
        }
    grouped = cal.compute_groupwise_errors(
        problem, result["intrinsics"], result["valid_indices"],
        result["rvecs"], result["tvecs"],
    )
    dist_mat = result["dist"][:5].reshape(1, 5)
    heldout = cal.heldout_reprojection_error(problem, local_indices, result["K_mat"], dist_mat)
    return {
        **fim_m,
        "focal_err_px": grouped["focal_err_px"],
        "pp_err_px": grouped["pp_err_px"],
        "dist_err": grouped["dist_err"],
        "train_rms": result["train_rms"],
        "heldout_rms": heldout,
    }


# ---------------------------------------------------------------------------
# Run all methods for one K
# ---------------------------------------------------------------------------

def run_one_k(
    K: int,
    problem: fim.CalibrationProblem,
    prior,
    num_random_trials: int,
) -> Dict[str, Dict[str, float]]:
    N = problem.num_candidates
    results: Dict[str, Dict[str, float]] = {}

    # ---- Random (mean over trials) ----
    print(f"  [random] {num_random_trials} trials...", flush=True)
    rng = np.random.default_rng(0)
    rand_trials: List[Dict[str, float]] = []
    for _ in range(num_random_trials):
        idx = sorted(rng.choice(N, size=K, replace=False).tolist())
        rand_trials.append(_evaluate(problem, idx, prior))
    metric_keys = [k for k, v in rand_trials[0].items() if isinstance(v, float)]
    rand_mean: Dict[str, float] = {}
    for mk in metric_keys:
        vals = [t[mk] for t in rand_trials if np.isfinite(t[mk])]
        rand_mean[mk] = float(np.mean(vals)) if vals else float("nan")
    results["random"] = rand_mean

    # ---- Deterministic methods ----
    methods_def = [
        ("coverage",         lambda: sel.coverage_selection(problem, K)[1]),
        ("a-optimal",        lambda: sel.greedy_selection_a_optimal(problem, K, prior=prior)[1]),
        ("d-optimal",        lambda: sel.greedy_selection_d_optimal(problem, K, prior=prior)[1]),
        ("greedy-e",         lambda: sel.greedy_selection(problem, K, prior=prior)[1]),
        ("fw-e",             lambda: sel.frank_wolfe_selection(problem, K, prior=prior)[1]),
        ("fw-a",             lambda: sel.frank_wolfe_a_optimal_selection(problem, K, prior=prior)[1]),
        ("fw-d",             lambda: sel.frank_wolfe_d_optimal_selection(problem, K, prior=prior)[1]),
    ]
    for name, fn in methods_def:
        t0 = time.time()
        try:
            local_idx = fn()
            elapsed = time.time() - t0
            m = _evaluate(problem, local_idx, prior)
            m["time_s"] = elapsed
            results[name] = m
            print(f"  [{name}] min_eig={m.get('min_eig', float('nan')):.4e}  "
                  f"heldout={m.get('heldout_rms', float('nan')):.4f}  ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  [{name}] FAILED: {e}", flush=True)
            nan = float("nan")
            results[name] = {"min_eig": nan, "logdet": nan, "focal_err_px": nan,
                             "pp_err_px": nan, "dist_err": nan,
                             "train_rms": nan, "heldout_rms": nan, "time_s": elapsed}

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="K-sweep: full calibration metrics, held-out = N-K unselected frames")
    parser.add_argument("--problem",
                        default="results/k20/datasets/aprilgrid_zero/all_candidate_problem.npz")
    parser.add_argument("--select-ks", type=int, nargs="+",
                        default=[5, 10, 15, 20, 30, 40, 50])
    parser.add_argument("--num-random-trials", type=int, default=25)
    parser.add_argument("--output-dir", default="results/k_sweep_calib_nk")
    parser.add_argument("--tag", default="aprilgrid_zero")
    args = parser.parse_args()

    fim.set_information_backend("numeric")
    problem = fim.load_calibration_problem_npz(args.problem)
    N = problem.num_candidates
    select_ks = sorted(set(args.select_ks))
    prior = fim.build_prior_blocks(problem)

    print(f"\n=== K-Sweep: Full Calibration Metrics (held-out = N-K) ===")
    print(f"  problem        : {args.problem}")
    print(f"  total N        : {N}")
    print(f"  K values       : {select_ks}")
    print(f"  random trials  : {args.num_random_trials}")
    print(f"  held-out       : all N-K unselected frames per method")

    # ---- Run sweep ----
    all_results: Dict[int, Dict[str, Dict[str, float]]] = {}
    for K in select_ks:
        if K >= N:
            print(f"\n[K={K}] skipped (K >= N={N})")
            continue
        print(f"\n{'='*60}")
        print(f"K = {K}  (held-out = {N - K} frames)", flush=True)
        all_results[K] = run_one_k(K, problem, prior, args.num_random_trials)

    # ---- Print summary table ----
    METRIC_COLS = ["min_eig", "logdet", "focal_err_px", "pp_err_px", "dist_err",
                   "train_rms", "heldout_rms"]
    METHODS = ["random", "coverage", "a-optimal", "d-optimal",
               "greedy-e", "fw-e", "fw-a", "fw-d"]
    for K in select_ks:
        if K not in all_results:
            continue
        print(f"\n--- K={K} (held-out={N-K}) ---")
        hdr = f"{'Method':<18}" + "".join(f"{m:>14}" for m in METRIC_COLS)
        print(hdr)
        print("-" * len(hdr))
        for meth in METHODS:
            if meth not in all_results[K]:
                continue
            r = all_results[K][meth]
            row = f"{meth:<18}"
            for mc in METRIC_COLS:
                v = r.get(mc, float("nan"))
                row += f"  {v:>12.4e}" if np.isfinite(v) else f"  {'—':>12}"
            print(row)

    # ---- Save JSON ----
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.output_dir) / f"{args.tag}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _ser(v):
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v) if np.isfinite(v) else None
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    json_data = {
        "problem": args.problem,
        "tag": args.tag,
        "n_full": N,
        "heldout_strategy": "N-K (all unselected frames)",
        "select_ks": select_ks,
        "results": {
            str(K): {
                meth: {mk: _ser(mv) for mk, mv in mdata.items()}
                for meth, mdata in kdata.items()
            }
            for K, kdata in all_results.items()
        },
    }
    out_path = out_dir / "k_sweep_calib_metrics.json"
    out_path.write_text(json.dumps(json_data, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
