"""Certificate validation via exhaustive search on small sub-problems.

For each (N_sub, K) setting we:
  1. Uniformly sub-sample N_sub candidates from the full 720-candidate pool.
  2. Run FW-E to obtain the relaxed objective f* (continuous), the rounded value f_rnd,
     and the FW duality gap δ at convergence (δ < 1e-4 certifies the relaxation is solved).
  3. Enumerate all C(N_sub, K) subsets exhaustively to find the true integer optimum λ*.
  4. Report: λ*, f_rnd, f*, δ, integer gap (λ*-f_rnd)/λ*, and verify δ < 1e-3.

The certificate claim: FW duality gap δ at convergence bounds how far f* is from the
true relaxation optimum. Exhaustive search independently confirms f_rnd ≈ λ*.

Usage:
    python3 -m Experiments.run_certificate_validation [--seed 0] [--output-dir results/certificate]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from itertools import combinations
from math import comb

import numpy as np

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from OASIS import FIM as fim
from OASIS import optimizations as optim
from OASIS.optimizations import csr_matrix


# ---------------------------------------------------------------------------
# Fast batched exhaustive search
# ---------------------------------------------------------------------------

def _precompute_contributions(
    info_blocks: fim.CandidateInfoBlocks,
    prior_blocks: fim.CalibrationPrior,
    n: int,
    intr_dim: int,
    pose_dim: int,
) -> np.ndarray:
    """Per-candidate Schur contribution: c_i = h_tt_i - h_tp_i @ inv(h_pp_prior + h_pp_i) @ h_tp_i.T"""
    reg = 1e-9 * np.eye(pose_dim)
    contribs = np.zeros((n, intr_dim, intr_dim), dtype=np.float64)
    for i in range(n):
        pose_block = prior_blocks.h_pp + info_blocks.h_pp[i]
        contribs[i] = (
            info_blocks.h_tt[i]
            - info_blocks.h_tp[i] @ np.linalg.pinv(pose_block + reg) @ info_blocks.h_tp[i].T
        )
    return contribs


def exhaustive_search(
    contribs: np.ndarray,
    prior_h_tt: np.ndarray,
    K: int,
    batch_size: int = 8192,
) -> tuple[tuple[int, ...], float]:
    """Find the K-subset maximising λ_min via batched exhaustive enumeration."""
    n = contribs.shape[0]
    total = comb(n, K)
    print(f"  Exhaustive: N={n}, K={K}, C(N,K)={total:,}", flush=True)

    H0 = prior_h_tt.copy()
    best_val = -np.inf
    best_subset: tuple[int, ...] = ()

    batch: list[tuple] = []

    def _process(batch: list[tuple]) -> None:
        nonlocal best_val, best_subset
        idx_arr = np.array(batch, dtype=np.intp)          # (B, K)
        H_batch = H0 + contribs[idx_arr].sum(axis=1)      # (B, d, d)
        H_batch = 0.5 * (H_batch + H_batch.transpose(0, 2, 1))
        eigs = np.linalg.eigvalsh(H_batch)[:, 0]          # (B,)
        argmax = int(np.argmax(eigs))
        if eigs[argmax] > best_val:
            best_val = float(eigs[argmax])
            best_subset = batch[argmax]

    for subset in combinations(range(n), K):
        batch.append(subset)
        if len(batch) == batch_size:
            _process(batch)
            batch = []
    if batch:
        _process(batch)

    return best_subset, best_val


# ---------------------------------------------------------------------------
# FW-E on the sub-problem — returns (relaxed_obj, rounded_obj, fw_gap_at_convergence)
# ---------------------------------------------------------------------------

def run_fw(
    sub_problem: fim.CalibrationProblem,
    K: int,
    prior_blocks: fim.CalibrationPrior,
) -> tuple[float, float, float]:
    prior_info = fim.build_prior_information(sub_problem)
    inf_mats, _ = fim.construct_candidate_inf_mats(sub_problem)

    n = sub_problem.num_candidates
    selection_init = np.full(n, float(K) / n, dtype=float)
    A = np.ones((1, n), dtype=float)
    b = np.array([float(K)], dtype=float)
    H0 = csr_matrix(prior_info)
    H0.eliminate_zeros()

    relaxed_sel, relaxed_obj, _, fw_gap = optim.frank_wolfe_optimization(
        inf_mats=inf_mats,
        H0=H0,
        selection_init=selection_init,
        num_poses=n,
        A=A,
        b=b,
    )

    rounded_sel = optim.roundsolution(relaxed_sel, K)
    info_blocks = fim.construct_candidate_inf_blocks(sub_problem)
    prior_b = fim._coerce_prior_blocks(sub_problem, prior_blocks)
    rounded_obj = fim.compute_min_eig_score(sub_problem, rounded_sel, info_blocks, prior=prior_b)

    return float(relaxed_obj), float(rounded_obj), float(fw_gap)


# ---------------------------------------------------------------------------
# Settings: all C(N,K) ≤ ~6M so exhaustive search completes in seconds
# ---------------------------------------------------------------------------

SETTINGS = [
    # (N_sub, K, label)
    (30,  5,  "N=30,  K=5"),
    (40,  5,  "N=40,  K=5"),
    (50,  5,  "N=50,  K=5"),
    (40,  6,  "N=40,  K=6"),
    (30,  8,  "N=30,  K=8"),
    (60,  5,  "N=60,  K=5"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem-npz",
                        default="results/k20/datasets/aprilgrid_zero/all_candidate_problem.npz")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="results/certificate_validation",
                        type=pathlib.Path)
    args = parser.parse_args()

    fim.set_information_backend("numeric")
    rng = np.random.default_rng(args.seed)

    print(f"Loading problem: {args.problem_npz}", flush=True)
    full_problem = fim.load_calibration_problem_npz(args.problem_npz)
    print(f"Full pool: N={full_problem.num_candidates}, "
          f"intrinsics_dim={full_problem.intrinsics_dim}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for N_sub, K, label in SETTINGS:
        n_total = comb(N_sub, K)
        print(f"\n{'='*60}", flush=True)
        print(f"[{label}]  C({N_sub},{K}) = {n_total:,}", flush=True)

        sub_indices = rng.choice(full_problem.num_candidates, size=N_sub, replace=False)
        sub_indices.sort()
        sub_problem = fim.subsample_calibration_problem(full_problem, sub_indices.tolist())
        prior_blocks = fim.build_prior_blocks(sub_problem)

        t0 = time.time()
        info_blocks = fim.construct_candidate_inf_blocks(sub_problem)
        contribs = _precompute_contributions(
            info_blocks, prior_blocks, N_sub,
            sub_problem.intrinsics_dim, sub_problem.pose_dim,
        )
        t_pre = time.time() - t0

        # --- Exhaustive ---
        t0 = time.time()
        _, true_opt = exhaustive_search(contribs, prior_blocks.h_tt, K)
        t_exh = time.time() - t0
        print(f"  Exhaustive done in {t_exh:.1f}s  λ*={true_opt:.6e}", flush=True)

        # --- FW-E ---
        t0 = time.time()
        relaxed_obj, rounded_obj, fw_gap = run_fw(sub_problem, K, prior_blocks)
        t_fw = time.time() - t0
        print(f"  FW-E done in {t_fw:.1f}s  f*={relaxed_obj:.6e}  "
              f"f_rnd={rounded_obj:.6e}  δ={fw_gap:.2e}", flush=True)

        int_gap = float(true_opt - rounded_obj)
        int_gap_pct = 100.0 * int_gap / true_opt if true_opt > 0 else float("nan")
        rounding_gap = float(relaxed_obj - rounded_obj)
        print(f"  integer gap={int_gap:.4e} ({int_gap_pct:.2f}%)  "
              f"rounding gap={rounding_gap:.4e}  δ<1e-3: {fw_gap < 1e-3}", flush=True)

        rows.append({
            "label": label.strip(),
            "N": N_sub,
            "K": K,
            "C_N_K": n_total,
            "true_opt": true_opt,
            "rounded_obj": rounded_obj,
            "relaxed_obj": relaxed_obj,
            "fw_gap_delta": fw_gap,
            "integer_gap": int_gap,
            "integer_gap_pct": int_gap_pct,
            "rounding_gap": rounding_gap,
            "time_exhaustive_s": t_exh,
            "time_fw_s": t_fw,
        })

    out = args.output_dir / "certificate_validation.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nSaved: {out}", flush=True)

    # Summary table
    print("\n" + "="*95)
    print(f"{'Setting':<14}  {'C(N,K)':>12}  {'λ* (exact)':>11}  {'f_rnd':>11}  "
          f"{'f* (relax)':>11}  {'δ (FW gap)':>11}  {'Int.gap%':>8}")
    print("-"*95)
    for r in rows:
        print(
            f"{r['label']:<14}  {r['C_N_K']:>12,}  {r['true_opt']:>11.4e}  "
            f"{r['rounded_obj']:>11.4e}  {r['relaxed_obj']:>11.4e}  "
            f"{r['fw_gap_delta']:>11.2e}  {r['integer_gap_pct']:>7.2f}%"
        )
    print("="*95)


if __name__ == "__main__":
    main()
