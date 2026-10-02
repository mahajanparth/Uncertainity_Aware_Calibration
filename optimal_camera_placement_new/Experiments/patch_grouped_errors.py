"""Patch existing benchmark JSON results to replace param_error with grouped errors.

Reads each benchmark_results.json, recomputes groupwise metrics from the
stored selected_indices, and writes the updated JSON in-place.

Usage:
    python3 -m Experiments.patch_grouped_errors --results-dir results/k20
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal


def patch_file(json_path: pathlib.Path, datasets: dict) -> None:
    raw = json.load(open(json_path))
    problem_path = raw.get("problem", "")

    # Find the matching problem file
    problem = datasets.get(problem_path)
    if problem is None:
        # Try to load by path (may be absolute or relative)
        p = pathlib.Path(problem_path)
        if not p.exists():
            print(f"  [skip] problem file not found: {problem_path}", flush=True)
            return
        problem = fim.load_calibration_problem_npz(str(p))
        datasets[problem_path] = problem

    changed = False
    for r in raw["results"]:
        indices = r.get("selected_indices", [])
        if not indices:
            continue

        # If grouped metrics already present and non-nan, skip
        if (np.isfinite(r.get("focal_err_px", float("nan"))) and
                np.isfinite(r.get("rot_err_deg", float("nan")))):
            continue

        result_full = cal.calibrate_opencv_full(problem, indices)
        if not result_full["success"]:
            nan = float("nan")
            for k in ("focal_err_px", "pp_err_px", "dist_err",
                      "rot_err_deg", "trans_err_cm"):
                r[k] = nan
        else:
            grouped = cal.compute_groupwise_errors(
                problem, result_full["intrinsics"],
                result_full["valid_indices"], result_full["rvecs"], result_full["tvecs"],
            )
            r.update(grouped)
            # Also refresh heldout_rms with correct K_mat/dist
            dist_mat = result_full["dist"][:5].reshape(1, 5)
            r["heldout_rms"] = cal.heldout_reprojection_error(
                problem, indices, result_full["K_mat"], dist_mat
            )
            r["train_rms"] = result_full["train_rms"]

        # Remove stale single L2 metric
        r.pop("param_error", None)
        changed = True

    if changed:
        json_path.write_text(json.dumps(raw, indent=2))
        print(f"  patched {json_path}", flush=True)
    else:
        print(f"  already up-to-date {json_path.parent.name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/k20", type=pathlib.Path)
    args = parser.parse_args()

    fim.set_information_backend("numeric")
    pattern = str(args.results_dir / "*" / "benchmark_*" / "benchmark_results.json")
    json_files = sorted(glob.glob(pattern))
    print(f"Found {len(json_files)} benchmark JSON files to patch.")

    datasets: dict = {}
    for jf in json_files:
        print(f"\n[{pathlib.Path(jf).parent.parent.name}]", flush=True)
        patch_file(pathlib.Path(jf), datasets)

    print("\nDone.")


if __name__ == "__main__":
    main()
