"""Full K=20 experiment runner for ICRA 2026 Tables 1-10.

Generates all synthetic datasets, runs all benchmarks at K=20,
runs the K-budget sweep (including A-optimal), and saves results
in results/k20/ for generate_k20_tables.py to consume.

Usage:
    python3 -m Experiments.run_all_k20 [--output-dir results/k20] [--num-random-trials 25]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from datetime import datetime

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from Experiments.generate_synthetic_dataset import generate as gen_dataset
from OASIS import FIM as fim
from OASIS import calibration_analysis as cal
from OASIS import methods as sel

# ---------------------------------------------------------------------------
# Board configs
# ---------------------------------------------------------------------------

APRILGRID_PARAMS = dict(
    board_type="aprilgrid",
    rows=6, cols=6,
    square_size=0.0844,   # pitch = 2*tag_size (tag_spacing=1 → uniform 4.22cm spacing)
    marker_size=0.0422,   # tag_size
)

CHARUCO_PARAMS = dict(
    board_type="charuco",
    rows=13, cols=13,
    square_size=0.0422,
    marker_size=0.0321,
)

DISTORTION_CONFIGS = {
    "zero":   dict(k1=0.0,   k2=0.0,  p1=0.0,   p2=0.0),
    "medium": dict(k1=-0.15, k2=0.05, p1=0.001, p2=0.001),
    "high":   dict(k1=-0.35, k2=0.15, p1=0.002, p2=0.002),
}

# dir_key → (board_params, distortion, sigma_file_suffix)
BENCHMARK_CONFIGS = {
    "aprilgrid_noiseless":          (APRILGRID_PARAMS, "zero",   "all_candidate_problem.npz"),
    "charuco_noiseless":            (CHARUCO_PARAMS,   "zero",   "all_candidate_problem.npz"),
    "aprilgrid_sigma05":            (APRILGRID_PARAMS, "zero",   "all_candidate_problem_sigma05.npz"),
    "charuco_sigma05":              (CHARUCO_PARAMS,   "zero",   "all_candidate_problem_sigma05.npz"),
    "aprilgrid_sigma10":            (APRILGRID_PARAMS, "zero",   "all_candidate_problem_sigma10.npz"),
    "charuco_sigma10":              (CHARUCO_PARAMS,   "zero",   "all_candidate_problem_sigma10.npz"),
    "aprilgrid_medium_noiseless":   (APRILGRID_PARAMS, "medium", "all_candidate_problem.npz"),
    "charuco_medium_noiseless":     (CHARUCO_PARAMS,   "medium", "all_candidate_problem.npz"),
    "aprilgrid_medium_sigma05":     (APRILGRID_PARAMS, "medium", "all_candidate_problem_sigma05.npz"),
    "charuco_medium_sigma05":       (CHARUCO_PARAMS,   "medium", "all_candidate_problem_sigma05.npz"),
    "aprilgrid_medium_sigma10":     (APRILGRID_PARAMS, "medium", "all_candidate_problem_sigma10.npz"),
    "charuco_medium_sigma10":       (CHARUCO_PARAMS,   "medium", "all_candidate_problem_sigma10.npz"),
    "aprilgrid_high_noiseless":     (APRILGRID_PARAMS, "high",   "all_candidate_problem.npz"),
    "charuco_high_noiseless":       (CHARUCO_PARAMS,   "high",   "all_candidate_problem.npz"),
    "aprilgrid_high_sigma05":       (APRILGRID_PARAMS, "high",   "all_candidate_problem_sigma05.npz"),
    "charuco_high_sigma05":         (CHARUCO_PARAMS,   "high",   "all_candidate_problem_sigma05.npz"),
    "aprilgrid_high_sigma10":       (APRILGRID_PARAMS, "high",   "all_candidate_problem_sigma10.npz"),
    "charuco_high_sigma10":         (CHARUCO_PARAMS,   "high",   "all_candidate_problem_sigma10.npz"),
}

# K-sweep configs: (dir_key, sigma_file) — 3 rows in Table 10
K_SWEEP_CONFIGS = [
    ("charuco_sigma05",    "all_candidate_problem_sigma05.npz"),
    ("aprilgrid_sigma05",  "all_candidate_problem_sigma05.npz"),
    ("aprilgrid_noiseless","all_candidate_problem.npz"),
]

K_SWEEP_VALUES = [5, 10, 15, 20, 30, 50, 100, 200]


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def _data_dir(base: pathlib.Path, board_type: str, dist_key: str) -> pathlib.Path:
    return base / "datasets" / f"{board_type}_{dist_key}"


def generate_all_datasets(base: pathlib.Path) -> dict:
    """Generate datasets for all board × distortion combos. Returns path map."""
    paths = {}
    seen = set()
    for dir_key, (bp, dist_key, _) in BENCHMARK_CONFIGS.items():
        board_type = bp["board_type"]
        key = (board_type, dist_key)
        if key in seen:
            continue
        seen.add(key)
        out = _data_dir(base, board_type, dist_key)
        print(f"\n[GEN] {board_type} dist={dist_key} → {out}", flush=True)
        gen_dataset(
            board_type=bp["board_type"],
            rows=bp["rows"],
            cols=bp["cols"],
            square_size=bp["square_size"],
            marker_size=bp["marker_size"],
            distortion=DISTORTION_CONFIGS[dist_key],
            output_dir=out,
        )
    print("\n[GEN] All datasets generated.", flush=True)


def _problem_path(base: pathlib.Path, dir_key: str) -> pathlib.Path:
    bp, dist_key, sigma_file = BENCHMARK_CONFIGS[dir_key]
    return _data_dir(base, bp["board_type"], dist_key) / sigma_file


# ---------------------------------------------------------------------------
# Benchmark helpers
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
    intrinsics = result["intrinsics"]
    train_rms = result["train_rms"]
    grouped = cal.compute_groupwise_errors(
        problem, intrinsics, result["valid_indices"], result["rvecs"], result["tvecs"]
    )
    dist_mat = result["dist"][:5].reshape(1, 5)
    heldout_rms = cal.heldout_reprojection_error(
        problem, indices, result["K_mat"], dist_mat
    )
    return {**grouped, "train_rms": train_rms, "heldout_rms": heldout_rms}


def _run_method(name, fn, problem, prior):
    print(f"  [{name}]", end=" ", flush=True)
    t0 = time.time()
    try:
        result_tuple = fn()
        selected_poses, selected_indices, best_score, selection = result_tuple[:4]
    except Exception as e:
        print(f"FAILED: {e}", flush=True)
        return {"method": name, "selected_indices": [], "time_s": float("nan"),
                "min_eig": float("nan"), "trace_cov": float("nan"), "logdet": float("nan"),
                "param_error": float("nan"), "train_rms": float("nan"), "heldout_rms": float("nan")}
    elapsed = time.time() - t0
    print(f"{elapsed:.1f}s", flush=True)
    result = {"method": name, "selected_indices": list(selected_indices), "time_s": elapsed}
    result.update(_fim_metrics(problem, selected_indices, prior))
    result.update(_calib_metrics(problem, selected_indices))
    return result


def run_benchmark(problem_path: pathlib.Path, K: int, num_random: int,
                  out_dir: pathlib.Path) -> dict:
    """Run 7-method benchmark, save JSON, return results."""
    problem = fim.load_calibration_problem_npz(str(problem_path))
    prior = fim.build_prior_blocks(problem)
    N = problem.num_candidates
    print(f"  N={N}, K={K}", flush=True)

    # Random
    t0 = time.time()
    rng = np.random.default_rng(0)
    rand_trials, rand_sels = [], []
    for _ in range(num_random):
        idx_list = sorted(rng.choice(N, size=K, replace=False).tolist())
        rand_sels.append(idx_list)
        m = _fim_metrics(problem, idx_list, prior)
        m.update(_calib_metrics(problem, idx_list))
        rand_trials.append(m)
    rand_time = time.time() - t0
    rand_result = {"method": "random", "time_s": rand_time / num_random}
    for k in [kk for kk in rand_trials[0] if isinstance(rand_trials[0][kk], float)]:
        vals = [t[k] for t in rand_trials if np.isfinite(t[k])]
        rand_result[k] = float(np.mean(vals)) if vals else float("nan")
        rand_result[f"{k}_std"] = float(np.std(vals)) if vals else float("nan")
    best_rand = int(np.argmax([t.get("min_eig", float("-inf")) for t in rand_trials]))
    rand_result["selected_indices"] = rand_sels[best_rand]
    print(f"  [random] {num_random} trials, mean min_eig={rand_result.get('min_eig', float('nan')):.3e}", flush=True)

    # Deterministic methods
    methods = [
        ("coverage",         lambda: sel.coverage_selection(problem, K) + (None,)),
        ("motion-diversity", lambda: sel.motion_diversity_selection(problem, K) + (None,)),
        ("a-optimal",        lambda: sel.greedy_selection_a_optimal(problem, K, prior=prior) + (None,)),
        ("d-optimal",        lambda: sel.greedy_selection_d_optimal(problem, K, prior=prior) + (None,)),
        ("greedy-e",         lambda: sel.greedy_selection(problem, K, prior=prior) + (None,)),
        ("fw-e",             lambda: sel.frank_wolfe_selection(problem, K, prior=prior)),
        ("fw-a",             lambda: sel.frank_wolfe_a_optimal_selection(problem, K, prior=prior)),
        ("fw-d",             lambda: sel.frank_wolfe_d_optimal_selection(problem, K, prior=prior)),
    ]
    all_results = [rand_result]
    for name, fn in methods:
        all_results.append(_run_method(name, fn, problem, prior))

    # Save
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    bench_dir = out_dir / f"benchmark_{stamp}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    def _ser(v):
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, np.floating): return float(v)
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, list) and v and isinstance(v[0], (np.integer, np.floating)):
            return [int(x) if isinstance(x, np.integer) else float(x) for x in v]
        return v

    json_data = {
        "problem": str(problem_path), "select_k": K,
        "num_candidates": N, "fim_backend": "numeric",
        "results": [{k: _ser(v) for k, v in r.items()} for r in all_results],
    }
    (bench_dir / "benchmark_results.json").write_text(json.dumps(json_data, indent=2))
    print(f"  → saved {bench_dir / 'benchmark_results.json'}", flush=True)
    return json_data


# ---------------------------------------------------------------------------
# K-budget sweep
# ---------------------------------------------------------------------------

def _fim_only(problem, indices, prior) -> float:
    try:
        m = cal.evaluate_selection(problem, indices, prior=prior)
        return float(m.get("min_eig", float("nan")))
    except Exception:
        return float("nan")


def run_k_sweep(problem_path: pathlib.Path, select_ks: list, num_random: int,
                out_dir: pathlib.Path, tag: str):
    problem = fim.load_calibration_problem_npz(str(problem_path))
    prior = fim.build_prior_blocks(problem)
    N = problem.num_candidates
    method_names = ["random", "d-optimal", "greedy-e", "fw-e", "a-optimal"]
    results = {m: {} for m in method_names}

    for K in select_ks:
        if K > N:
            print(f"  K={K} > N={N}, skip")
            continue
        print(f"  K={K}: ", end="", flush=True)
        rng = np.random.default_rng(42)
        rand_eigs = [_fim_only(problem, sorted(rng.choice(N, K, replace=False).tolist()), prior)
                     for _ in range(num_random)]
        valid = [v for v in rand_eigs if np.isfinite(v)]
        results["random"][K] = {
            "min_eig": float(np.mean(valid)) if valid else float("nan"),
            "min_eig_std": float(np.std(valid)) if valid else float("nan"),
        }

        for name, fn in [
            ("d-optimal",  lambda K=K: sel.greedy_selection_d_optimal(problem, K, prior=prior)[1]),
            ("greedy-e",   lambda K=K: sel.greedy_selection(problem, K, prior=prior)[1]),
            ("a-optimal",  lambda K=K: sel.greedy_selection_a_optimal(problem, K, prior=prior)[1]),
        ]:
            t0 = time.time()
            idx = fn()
            results[name][K] = {"min_eig": _fim_only(problem, idx, prior), "time_s": time.time()-t0}

        t0 = time.time()
        try:
            _, idx, _, _, _ = sel.frank_wolfe_selection(problem, K, prior=prior)
            results["fw-e"][K] = {"min_eig": _fim_only(problem, idx, prior), "time_s": time.time()-t0}
        except Exception as e:
            results["fw-e"][K] = {"min_eig": float("nan"), "time_s": time.time()-t0}
            print(f"(fw-e failed: {e}) ", end="", flush=True)

        print("  ".join(f"{m}={results[m][K].get('min_eig', float('nan')):.3e}"
                        for m in method_names), flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)

    def _ser(v):
        if isinstance(v, (np.integer,)): return int(v)
        if isinstance(v, (np.floating, float)): return None if (v != v) else float(v)
        return v

    json_out = {
        "problem": str(problem_path), "tag": tag,
        "num_candidates": N, "select_ks": select_ks,
        "results": {
            method: {str(k): {mk: _ser(mv) for mk, mv in kdata.items()}
                     for k, kdata in kd.items()}
            for method, kd in results.items()
        },
    }
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"k_sweep_{tag}_{stamp}.json"
    out_path.write_text(json.dumps(json_out, indent=2))
    print(f"  → saved {out_path}", flush=True)
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/k20", type=pathlib.Path)
    parser.add_argument("--num-random-trials", type=int, default=25)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip benchmark if result JSON already exists")
    args = parser.parse_args()

    base = args.output_dir
    base.mkdir(parents=True, exist_ok=True)
    fim.set_information_backend("numeric")

    # ---- Step 1: Generate datasets ----
    print("\n" + "="*60)
    print("STEP 1: Generate synthetic datasets")
    print("="*60)
    generate_all_datasets(base)

    # ---- Step 2: Run all 18 benchmarks ----
    print("\n" + "="*60)
    print("STEP 2: Run benchmarks (K=20, 18 configurations)")
    print("="*60)
    for dir_key in BENCHMARK_CONFIGS:
        print(f"\n--- {dir_key} ---", flush=True)
        bench_out = base / dir_key
        # Skip if already done
        if args.skip_existing:
            existing = list(bench_out.glob("benchmark_*/benchmark_results.json"))
            if existing:
                print(f"  (skipping — {existing[-1]} exists)", flush=True)
                continue
        prob_path = _problem_path(base, dir_key)
        if not prob_path.exists():
            print(f"  [error] problem file not found: {prob_path}", flush=True)
            continue
        run_benchmark(prob_path, K=args.k,
                      num_random=args.num_random_trials, out_dir=bench_out)

    # ---- Step 3: K-budget sweep ----
    print("\n" + "="*60)
    print("STEP 3: K-budget sweep (Table 10)")
    print("="*60)
    sweep_out = base / "k_sweep"
    for dir_key, sigma_file in K_SWEEP_CONFIGS:
        bp, dist_key, _ = BENCHMARK_CONFIGS[dir_key]
        prob_path = _data_dir(base, bp["board_type"], dist_key) / sigma_file
        print(f"\n--- K-sweep: {dir_key} ---", flush=True)
        if not prob_path.exists():
            print(f"  [error] not found: {prob_path}", flush=True)
            continue
        run_k_sweep(prob_path, K_SWEEP_VALUES, num_random=5,
                    out_dir=sweep_out, tag=dir_key)

    print("\n" + "="*60)
    print(f"ALL DONE. Results in: {base.resolve()}")
    print("="*60)


if __name__ == "__main__":
    main()
