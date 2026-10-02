"""K-budget sweep: lambda_min vs K for all selection methods.

Fixed problem, fixed N, vary K in {5, 10, 15, 20, 30, 50, 100, 200}.
Shows diminishing returns in FIM observability as budget grows.

Usage:
  python3 -m Experiments.main_k_budget_sweep \\
      --problem isaac_outputs/.../all_candidate_problem_sigma05.npz \\
      --select-ks 5 10 15 20 30 50 100 200 \\
      --num-random-trials 5
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal
from OASIS import methods as sel


def _out_dir(base: pathlib.Path, tag: str) -> pathlib.Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = base / f"k_budget_sweep_{tag}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fim_only(problem, indices, prior) -> float:
    try:
        m = cal.evaluate_selection(problem, indices, prior=prior)
        return float(m.get("min_eig", float("nan")))
    except Exception:
        return float("nan")


def run_sweep(problem, prior, select_ks: List[int], num_random_trials: int) -> Dict[str, Any]:
    """Run all methods for each K value. Returns results dict."""
    method_names = ["random", "coverage", "d-optimal", "greedy-e", "fw-e"]
    results: Dict[str, Dict[int, Dict[str, float]]] = {m: {} for m in method_names}

    for K in select_ks:
        print(f"\n--- K={K} ---", flush=True)
        N = problem.num_candidates
        if K > N:
            print(f"  K={K} > N={N}, skipping")
            continue

        # Random
        t0 = time.time()
        rng = np.random.default_rng(42)
        rand_eigs = []
        for _ in range(num_random_trials):
            idx_list = sorted(rng.choice(N, size=K, replace=False).tolist())
            rand_eigs.append(_fim_only(problem, idx_list, prior))
        rand_time = time.time() - t0
        valid_eigs = [v for v in rand_eigs if np.isfinite(v)]
        results["random"][K] = {
            "min_eig": float(np.mean(valid_eigs)) if valid_eigs else float("nan"),
            "min_eig_std": float(np.std(valid_eigs)) if valid_eigs else float("nan"),
            "time_s": rand_time / num_random_trials,
        }
        print(f"  random       min_eig={results['random'][K]['min_eig']:.4e}", flush=True)

        # Coverage
        t0 = time.time()
        _, idx, _, _ = sel.coverage_selection(problem, K)
        elapsed = time.time() - t0
        results["coverage"][K] = {"min_eig": _fim_only(problem, idx, prior), "time_s": elapsed}
        print(f"  coverage     min_eig={results['coverage'][K]['min_eig']:.4e}", flush=True)

        # D-optimal
        t0 = time.time()
        _, idx, _, _ = sel.greedy_selection_d_optimal(problem, K, prior=prior)
        elapsed = time.time() - t0
        results["d-optimal"][K] = {"min_eig": _fim_only(problem, idx, prior), "time_s": elapsed}
        print(f"  d-optimal    min_eig={results['d-optimal'][K]['min_eig']:.4e}", flush=True)

        # Greedy-E
        t0 = time.time()
        _, idx, score, _ = sel.greedy_selection(problem, K, prior=prior)
        elapsed = time.time() - t0
        results["greedy-e"][K] = {"min_eig": _fim_only(problem, idx, prior), "time_s": elapsed}
        print(f"  greedy-e     min_eig={results['greedy-e'][K]['min_eig']:.4e}", flush=True)

        # FW-E
        t0 = time.time()
        try:
            _, idx, score, _, _ = sel.frank_wolfe_selection(problem, K, prior=prior)
            elapsed = time.time() - t0
            results["fw-e"][K] = {"min_eig": _fim_only(problem, idx, prior), "time_s": elapsed}
            print(f"  fw-e         min_eig={results['fw-e'][K]['min_eig']:.4e}", flush=True)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  fw-e         FAILED: {e}", flush=True)
            results["fw-e"][K] = {"min_eig": float("nan"), "time_s": elapsed}

    return results


def plot_sweep(results: Dict[str, Dict[int, Dict[str, float]]], select_ks: List[int],
               title: str, save_path: str) -> None:
    colours = {
        "random":          "#888888",
        "coverage":        "#4daf4a",

        "d-optimal":       "#ff7f00",
        "greedy-e":        "#377eb8",
        "fw-e":            "#e41a1c",
    }
    markers = {
        "random": "o", "coverage": "s",
        "d-optimal": "D", "greedy-e": "v", "fw-e": "*",
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left: lambda_min vs K ---
    ax = axes[0]
    for method, kdata in results.items():
        ks = sorted(k for k in kdata if k in select_ks)
        ys = [kdata[k].get("min_eig", float("nan")) for k in ks]
        ys_std = [kdata[k].get("min_eig_std", 0.0) for k in ks]
        valid = [i for i, y in enumerate(ys) if np.isfinite(y)]
        if not valid:
            continue
        ks_v = [ks[i] for i in valid]
        ys_v = [ys[i] for i in valid]
        ys_std_v = [ys_std[i] for i in valid]
        ax.plot(ks_v, ys_v, marker=markers.get(method, "o"),
                color=colours.get(method, "black"), label=method, linewidth=2, markersize=6)
        if any(s > 0 for s in ys_std_v):
            ax.fill_between(ks_v,
                            [y - s for y, s in zip(ys_v, ys_std_v)],
                            [y + s for y, s in zip(ys_v, ys_std_v)],
                            alpha=0.15, color=colours.get(method, "black"))
    ax.set_xlabel("Budget K (number of selected poses)")
    ax.set_ylabel(r"$\lambda_{\min}(H_{\mathrm{cal}})$")
    ax.set_title("Observability vs Budget")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Right: normalised lambda_min (relative to random mean) ---
    ax = axes[1]
    rand_vals = {k: results["random"].get(k, {}).get("min_eig", float("nan")) for k in select_ks}
    for method, kdata in results.items():
        ks = sorted(k for k in kdata if k in select_ks)
        ys_norm = []
        ks_v = []
        for k in ks:
            raw = kdata[k].get("min_eig", float("nan"))
            base = rand_vals.get(k, float("nan"))
            if np.isfinite(raw) and np.isfinite(base) and base > 0:
                ys_norm.append(raw / base)
                ks_v.append(k)
        if not ks_v:
            continue
        ax.plot(ks_v, ys_norm, marker=markers.get(method, "o"),
                color=colours.get(method, "black"), label=method, linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="grey", linestyle="--", linewidth=1, label="random baseline")
    ax.set_xlabel("Budget K (number of selected poses)")
    ax.set_ylabel(r"$\lambda_{\min}$ / $\lambda_{\min}^{\mathrm{random}}$")
    ax.set_title("Relative gain over random")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {save_path}")


def print_table(results: Dict[str, Dict[int, Dict[str, float]]], select_ks: List[int]) -> None:
    col_w = 12
    header = f"{'K':>5}  " + "  ".join(f"{m:>{col_w}}" for m in results)
    sep = "-" * len(header)
    print(f"\n{'='*len(header)}")
    print("lambda_min vs K")
    print(header)
    print(sep)
    for K in select_ks:
        row = f"{K:>5}  "
        for method, kdata in results.items():
            v = kdata.get(K, {}).get("min_eig", float("nan"))
            row += f"  {v:>{col_w}.3e}"
        print(row)
    print("=" * len(header))


def main():
    parser = argparse.ArgumentParser(description="K-budget sweep: lambda_min vs K")
    parser.add_argument(
        "--problem",
        default="isaac_outputs/interactive_run_charuco_100_noiseless/all_candidate_problem_sigma05.npz",
    )
    parser.add_argument(
        "--select-ks", type=int, nargs="+",
        default=[5, 10, 15, 20, 30, 50, 100, 200],
    )
    parser.add_argument("--num-random-trials", type=int, default=5)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--tag", default="charuco_sigma05",
                        help="Short tag appended to output directory name")
    args = parser.parse_args()

    fim.set_information_backend("numeric")
    problem = fim.load_calibration_problem_npz(args.problem)
    prior = fim.build_prior_blocks(problem)
    N = problem.num_candidates
    select_ks = sorted(set(args.select_ks))

    print(f"\n=== K-Budget Sweep ===")
    print(f"  problem    : {args.problem}")
    print(f"  candidates : {N}")
    print(f"  K values   : {select_ks}")
    print(f"  tag        : {args.tag}")

    results = run_sweep(problem, prior, select_ks, args.num_random_trials)

    print_table(results, select_ks)

    out_dir = _out_dir(pathlib.Path(args.output_dir), args.tag)

    plot_sweep(
        results, select_ks,
        title=f"K-Budget Sweep  ({args.tag},  N={N})",
        save_path=str(out_dir / "k_budget_sweep.png"),
    )

    # Save JSON
    def _ser(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return None if (v != v) else float(v)
        return v

    json_out: Dict[str, Any] = {
        "problem": args.problem,
        "tag": args.tag,
        "num_candidates": N,
        "select_ks": select_ks,
        "results": {
            method: {
                str(k): {mk: _ser(mv) for mk, mv in kdata.items()}
                for k, kdata in kd.items()
            }
            for method, kd in results.items()
        },
    }
    (out_dir / "k_budget_sweep.json").write_text(json.dumps(json_out, indent=2))
    print(f"\nOutputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
