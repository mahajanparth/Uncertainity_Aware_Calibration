"""Main subset-selection benchmark.

Compares 7 pose-selection methods on the same candidate pool:
  1. random          — uniformly random (25 trials)
  2. coverage        — greedy farthest-point on translations
  3. motion-diversity— greedy farthest-point on SO(3) rotations
  4. a-optimal       — greedy minimise trace(H_cal^{-1})
  5. d-optimal       — greedy maximise logdet(H_cal)
  6. greedy-e        — greedy maximise lambda_min(H_cal)
  7. fw-e            — Frank-Wolfe continuous relaxation + rounding

Metrics:
  FIM proxies : min_eig, trace_cov, logdet  (from Fisher information)
  Calibration : param_error, train_rms, heldout_rms  (from cv2.calibrateCamera)
  Runtime     : time_s

Usage:
  python3 -m Experiments.main_benchmark \\
      --problem isaac_outputs/.../all_candidate_problem.npz \\
      --select-k 20 --fim-backend numeric
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal
from OASIS import calibration_visualize as cal_viz
from OASIS import methods as sel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _out_dir(base: pathlib.Path) -> pathlib.Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"benchmark_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fim_metrics(problem, indices, prior) -> Dict[str, float]:
    try:
        return cal.evaluate_selection(problem, indices, prior=prior)
    except Exception as e:
        print(f"    [warn] FIM metrics failed: {e}")
        return {"min_eig": float("nan"), "trace_cov": float("nan"),
                "logdet": float("nan"), "cond": float("nan"), "visible_points": 0.0}


def _calib_metrics(problem, indices) -> Dict[str, float]:
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


def _run_method(name: str, fn, problem, prior, skip_calibration: bool) -> Dict[str, Any]:
    print(f"\n[{name}] running...", flush=True)
    t0 = time.time()
    try:
        selected_poses, selected_indices, best_score, selection, *_ = fn()
    except Exception as e:
        print(f"    [error] {e}")
        return {"method": name, "selected_indices": [], "time_s": float("nan"),
                "min_eig": float("nan"), "trace_cov": float("nan"), "logdet": float("nan"),
                "param_error": float("nan"), "train_rms": float("nan"), "heldout_rms": float("nan")}
    elapsed = time.time() - t0
    print(f"    selected {len(selected_indices)} poses in {elapsed:.1f}s  score={best_score:.4e}", flush=True)

    result: Dict[str, Any] = {"method": name, "selected_indices": selected_indices, "time_s": elapsed}
    result.update(_fim_metrics(problem, selected_indices, prior))
    if not skip_calibration:
        result.update(_calib_metrics(problem, selected_indices))
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pose-selection benchmark: 7 methods head-to-head")
    parser.add_argument(
        "--problem",
        default="isaac_outputs/interactive_run_aprilgrid_100_noiseless/all_candidate_problem.npz",
        help="Path to calibration_problem.npz",
    )
    parser.add_argument("--select-k", type=int, default=20)
    parser.add_argument("--num-random-trials", type=int, default=25)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--fim-backend", choices=("numeric", "gtsam"), default="numeric")
    parser.add_argument("--skip-calibration", action="store_true",
                        help="Skip cv2.calibrateCamera step (FIM metrics only, faster)")
    args = parser.parse_args()

    fim.set_information_backend(args.fim_backend)
    problem = fim.load_calibration_problem_npz(args.problem)
    prior = fim.build_prior_blocks(problem)
    K = args.select_k
    N = problem.num_candidates

    print(f"\n=== Benchmark ===")
    print(f"  problem     : {args.problem}")
    print(f"  candidates  : {N}")
    print(f"  select_k    : {K}")
    print(f"  FIM backend : {args.fim_backend}")
    print(f"  calibration : {'skipped' if args.skip_calibration else 'cv2.calibrateCamera'}")

    # ------------------------------------------------------------------
    # 1. Random baseline — run multiple trials, report mean ± std
    # ------------------------------------------------------------------
    print(f"\n[random] running {args.num_random_trials} trials...", flush=True)
    t0 = time.time()
    rng = np.random.default_rng(0)
    rand_trials: List[Dict[str, float]] = []
    rand_selections: List[List[int]] = []
    for seed in range(args.num_random_trials):
        idx_list = sorted(rng.choice(N, size=K, replace=False).tolist())
        rand_selections.append(idx_list)
        m = _fim_metrics(problem, idx_list, prior)
        if not args.skip_calibration:
            m.update(_calib_metrics(problem, idx_list))
        rand_trials.append(m)
    rand_time = time.time() - t0

    rand_result: Dict[str, Any] = {"method": "random", "time_s": rand_time / args.num_random_trials}
    metric_keys = [k for k in rand_trials[0] if isinstance(rand_trials[0][k], float)]
    for k in metric_keys:
        vals = [t[k] for t in rand_trials if np.isfinite(t[k])]
        rand_result[k] = float(np.mean(vals)) if vals else float("nan")
        rand_result[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    # Best random trial by min_eig
    best_rand_idx = int(np.argmax([t.get("min_eig", float("-inf")) for t in rand_trials]))
    rand_result["selected_indices"] = rand_selections[best_rand_idx]
    print(f"    mean min_eig={rand_result.get('min_eig', float('nan')):.4e}  "
          f"±{rand_result.get('min_eig_std', 0):.2e}", flush=True)

    # ------------------------------------------------------------------
    # 2–7. Deterministic methods
    # ------------------------------------------------------------------
    deterministic_methods = [
        ("coverage",         lambda: sel.coverage_selection(problem, K) + (None,)),
        ("a-optimal",        lambda: sel.greedy_selection_a_optimal(problem, K, prior=prior) + (None,)),
        ("d-optimal",        lambda: sel.greedy_selection_d_optimal(problem, K, prior=prior) + (None,)),
        ("greedy-e",         lambda: sel.greedy_selection(problem, K, prior=prior) + (None,)),
        ("fw-e",             lambda: sel.frank_wolfe_selection(problem, K, prior=prior)),
        ("fw-a",             lambda: sel.frank_wolfe_a_optimal_selection(problem, K, prior=prior)),
        ("fw-d",             lambda: sel.frank_wolfe_d_optimal_selection(problem, K, prior=prior)),
    ]

    all_results: List[Dict[str, Any]] = [rand_result]
    for name, fn in deterministic_methods:
        result = _run_method(name, fn, problem, prior, args.skip_calibration)
        all_results.append(result)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    table_metrics = ["min_eig", "logdet", "trace_cov", "focal_err_px", "pp_err_px",
                     "dist_err", "rot_err_deg", "trans_err_cm",
                     "train_rms", "heldout_rms", "time_s"]
    col_w = 14
    header = f"{'Method':<18}" + "".join(f"{m:>{col_w}}" for m in table_metrics)
    print(f"\n{'='*len(header)}")
    print("RESULTS")
    print(header)
    print("-" * len(header))
    # Sort by min_eig descending
    for r in sorted(all_results, key=lambda x: x.get("min_eig", float("-inf")), reverse=True):
        row = f"{r['method']:<18}"
        for m in table_metrics:
            v = r.get(m, float("nan"))
            std = r.get(f"{m}_std", 0.0)
            if not np.isfinite(v):
                row += f"{'—':>{col_w}}"
            elif std > 0:
                row += f"{v:>{col_w-4}.3e}±{std:.1e}"
            else:
                row += f"{v:>{col_w}.4e}"
        print(row)
    print("=" * len(header))

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    out_dir = _out_dir(pathlib.Path(args.output_dir))

    # JSON
    def _serialisable(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, list) and v and isinstance(v[0], (np.integer, np.floating)):
            return [int(x) if isinstance(x, np.integer) else float(x) for x in v]
        return v

    json_data = {
        "problem": args.problem, "select_k": K, "num_candidates": N,
        "fim_backend": args.fim_backend,
        "results": [
            {k: _serialisable(v) for k, v in r.items()}
            for r in all_results
        ],
    }
    (out_dir / "benchmark_results.json").write_text(json.dumps(json_data, indent=2))

    # CSV (LaTeX-ready) — mean columns + std columns for random row
    std_metrics = [f"{m}_std" for m in table_metrics]
    csv_fields = ["method"] + table_metrics + std_metrics
    csv_path = out_dir / "benchmark_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        for r in all_results:
            writer.writerow({
                k: (r["method"] if k == "method"
                    else f"{r.get(k, float('nan')):.4e}")
                for k in csv_fields
            })

    # Plot
    plot_data = {r["method"]: r for r in all_results}
    cal_viz.plot_benchmark_comparison(plot_data, save_path=str(out_dir / "benchmark_comparison.png"))

    print(f"\nOutputs saved to: {out_dir}")
    print(f"  benchmark_results.json")
    print(f"  benchmark_table.csv")
    print(f"  benchmark_comparison.png")


if __name__ == "__main__":
    main()
