"""Exhaustive comparison: true discrete optimum vs Greedy vs FW-rounded.

For small subsampled pools (N∈{30,40,50}) and budgets K∈{5,6,8}, we:
  1. Enumerate all C(N,K) subsets and find the true discrete optimum
     under E-optimal (max λ_min), A-optimal (min trace H⁻¹), D-optimal (max log det).
  2. Run Greedy-E, Greedy-A, Greedy-D and FW-E, FW-A, FW-D.
  3. Report: true opt, greedy value, greedy gap%, FW-rnd value, FW gap%,
     FW continuous relaxation value (the upper/lower bound).

Usage:
  python3 -m Experiments.main_exhaustive_comparison \\
      --problem results/k20/datasets/aprilgrid_zero/all_candidate_problem.npz \\
      --output /tmp/exhaustive_comparison.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from itertools import combinations
from typing import Dict, List, Tuple

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from OASIS import FIM as fim
from OASIS import methods as sel


# ---------------------------------------------------------------------------
# Precompute per-candidate Schur contributions
# ---------------------------------------------------------------------------

def precompute_schur_contributions(
    info_blocks: fim.CandidateInfoBlocks,
    prior: fim.CalibrationPrior,
    N: int,
) -> np.ndarray:
    """Return shape (N, d, d) where s[i] = h_tt[i] - h_tp[i] @ inv(prior_pp + h_pp[i]) @ h_tp[i]^T.

    H_cal(S) = prior_tt + sum_{i in S} s[i].  Decomposition is valid because
    each candidate has its own independent pose block.
    """
    d = info_blocks.h_tt.shape[1]
    s = np.zeros((N, d, d), dtype=float)
    reg = 1e-9 * np.eye(info_blocks.h_pp.shape[1])
    for i in range(N):
        pose_block = prior.h_pp + info_blocks.h_pp[i] + reg
        h_tp = info_blocks.h_tp[i]
        s[i] = info_blocks.h_tt[i] - h_tp @ np.linalg.solve(pose_block, h_tp.T)
    return s


# ---------------------------------------------------------------------------
# Fast exhaustive search using batched numpy
# ---------------------------------------------------------------------------

BATCH = 200_000  # subsets per batch


def exhaustive_search(
    s_contrib: np.ndarray,    # (N, d, d)
    prior_tt: np.ndarray,     # (d, d)
    K: int,
) -> Tuple[float, float, float, List[int], List[int], List[int]]:
    """Return (best_min_eig, best_neg_trace, best_logdet, best_E_idx, best_A_idx, best_D_idx).

    best_neg_trace is stored negative so "higher is better" for all three.
    """
    N = s_contrib.shape[0]
    num_subsets = 0
    best_e = -np.inf;  best_e_idx = None
    best_a = -np.inf;  best_a_idx = None   # stored as -trace (higher = better)
    best_d = -np.inf;  best_d_idx = None

    batch_idxs = []
    all_combos = combinations(range(N), K)

    def _process_batch(batch):
        nonlocal best_e, best_e_idx, best_a, best_a_idx, best_d, best_d_idx
        arr = np.array(batch, dtype=np.int32)          # (B, K)
        B = arr.shape[0]
        # H_cal[b] = prior_tt + sum of s_contrib[arr[b, :]]
        H = prior_tt[None] + s_contrib[arr].sum(axis=1)  # (B, d, d)
        # symmetrize
        H = 0.5 * (H + H.transpose(0, 2, 1))

        # E-criterion: max min_eig
        eigs = np.linalg.eigvalsh(H)          # (B, d)
        me = eigs[:, 0]                        # min eigenvalue per subset
        bi = int(np.argmax(me))
        if me[bi] > best_e:
            best_e = float(me[bi])
            best_e_idx = batch[bi]

        # D-criterion: max log det
        signs, logdets = np.linalg.slogdet(H)
        logdets = np.where(signs > 0, logdets, -np.inf)
        bi = int(np.argmax(logdets))
        if logdets[bi] > best_d:
            best_d = float(logdets[bi])
            best_d_idx = batch[bi]

        # A-criterion: min trace(H^{-1}) → max -trace(H^{-1})
        try:
            H_inv = np.linalg.inv(H)           # (B, d, d)
            tr_inv = np.trace(H_inv, axis1=1, axis2=2)   # (B,)
            neg_tr = -tr_inv
            bi = int(np.argmax(neg_tr))
            if neg_tr[bi] > best_a:
                best_a = float(neg_tr[bi])
                best_a_idx = batch[bi]
        except np.linalg.LinAlgError:
            pass

    batch = []
    for combo in all_combos:
        batch.append(list(combo))
        num_subsets += 1
        if len(batch) >= BATCH:
            _process_batch(batch)
            batch = []
    if batch:
        _process_batch(batch)

    return best_e, -best_a, best_d, best_e_idx, best_a_idx, best_d_idx


# ---------------------------------------------------------------------------
# Evaluate a discrete selection under all three criteria
# ---------------------------------------------------------------------------

def eval_all(
    s_contrib: np.ndarray,
    prior_tt: np.ndarray,
    indices: List[int],
) -> Tuple[float, float, float]:
    H = prior_tt + s_contrib[list(indices)].sum(axis=0)
    H = 0.5 * (H + H.T)
    min_eig = float(np.min(np.linalg.eigvalsh(H)))
    sign, ld = np.linalg.slogdet(H)
    logdet = float(ld) if sign > 0 else float("-inf")
    try:
        trace_inv = float(np.trace(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        trace_inv = float("inf")
    return min_eig, trace_inv, logdet


def eval_relaxed(
    info_blocks: fim.CandidateInfoBlocks,
    prior: fim.CalibrationPrior,
    problem_sub: fim.CalibrationProblem,
    relaxed_sel: np.ndarray,
) -> Tuple[float, float, float]:
    """Compute criteria for the continuous relaxed selection weights."""
    H = fim.compute_calibration_schur_compact(problem_sub, relaxed_sel, info_blocks, prior=prior)
    H = 0.5 * (H + H.T)
    min_eig = float(np.min(np.linalg.eigvalsh(H)))
    sign, ld = np.linalg.slogdet(H)
    logdet = float(ld) if sign > 0 else float("-inf")
    try:
        trace_inv = float(np.trace(np.linalg.inv(H)))
    except np.linalg.LinAlgError:
        trace_inv = float("inf")
    return min_eig, trace_inv, logdet


# ---------------------------------------------------------------------------
# Gap helpers
# ---------------------------------------------------------------------------

def gap_pct(val: float, true_opt: float, maximize: bool) -> float:
    """Signed percentage gap relative to true optimum."""
    if not np.isfinite(val) or not np.isfinite(true_opt) or abs(true_opt) < 1e-30:
        return float("nan")
    if maximize:
        return 100.0 * (true_opt - val) / abs(true_opt)   # positive = worse
    else:
        return 100.0 * (val - true_opt) / abs(true_opt)   # positive = worse


# ---------------------------------------------------------------------------
# Run one (N, K) configuration
# ---------------------------------------------------------------------------

def run_one(
    problem_full: fim.CalibrationProblem,
    N: int,
    K: int,
    seed: int,
    prior_full: fim.CalibrationPrior,
) -> Dict:
    from math import comb
    rng = np.random.default_rng(seed)
    sub_idx = sorted(rng.choice(problem_full.num_candidates, size=N, replace=False).tolist())
    problem_sub = fim.subsample_calibration_problem(problem_full, sub_idx)
    prior = fim.build_prior_blocks(problem_sub)
    info_blocks = fim.construct_candidate_inf_blocks(problem_sub)
    s_contrib = precompute_schur_contributions(info_blocks, prior, N)
    prior_tt = prior.h_tt

    num_subsets = comb(N, K)
    print(f"  N={N}, K={K}: {num_subsets:,} subsets — exhaustive...", flush=True)
    t0 = time.time()
    best_e, best_a_trace, best_d, best_e_idx, best_a_idx, best_d_idx = exhaustive_search(
        s_contrib, prior_tt, K
    )
    t_exh = time.time() - t0
    print(f"    exhaustive done in {t_exh:.1f}s | λ*={best_e:.4e} | tr*={best_a_trace:.4e} | ld*={best_d:.4f}", flush=True)

    # --- Greedy methods ---
    def run_greedy(fn, label):
        t = time.time()
        try:
            _, idx, score, _ = fn(problem_sub, K, prior=prior)
            elapsed = time.time() - t
            me, tri, ld = eval_all(s_contrib, prior_tt, idx)
            print(f"    {label}: λ={me:.4e} tr={tri:.4e} ld={ld:.4f} ({elapsed:.1f}s)", flush=True)
            return {"indices": list(idx), "min_eig": me, "trace_inv": tri, "logdet": ld, "time_s": elapsed}
        except Exception as ex:
            print(f"    {label}: FAILED {ex}", flush=True)
            return {"indices": [], "min_eig": float("nan"), "trace_inv": float("nan"), "logdet": float("nan")}

    greedy_e = run_greedy(sel.greedy_selection, "Greedy-E")
    greedy_a = run_greedy(sel.greedy_selection_a_optimal, "Greedy-A")
    greedy_d = run_greedy(sel.greedy_selection_d_optimal, "Greedy-D")

    # --- FW methods ---
    def run_fw(fn, label):
        t = time.time()
        try:
            _, idx, score, binary_sel, relaxed_sel = fn(problem_sub, K, prior=prior)
            elapsed = time.time() - t
            me, tri, ld = eval_all(s_contrib, prior_tt, idx)
            r_me, r_tri, r_ld = eval_relaxed(info_blocks, prior, problem_sub, relaxed_sel)
            print(f"    {label}: λ={me:.4e} tr={tri:.4e} ld={ld:.4f} | relax λ={r_me:.4e} ({elapsed:.1f}s)", flush=True)
            return {
                "indices": list(idx),
                "min_eig": me, "trace_inv": tri, "logdet": ld,
                "relax_min_eig": r_me, "relax_trace_inv": r_tri, "relax_logdet": r_ld,
                "time_s": elapsed,
            }
        except Exception as ex:
            print(f"    {label}: FAILED {ex}", flush=True)
            nan = float("nan")
            return {"indices": [], "min_eig": nan, "trace_inv": nan, "logdet": nan,
                    "relax_min_eig": nan, "relax_trace_inv": nan, "relax_logdet": nan}

    fw_e = run_fw(sel.frank_wolfe_selection, "FW-E")
    fw_a = run_fw(sel.frank_wolfe_a_optimal_selection, "FW-A")
    fw_d = run_fw(sel.frank_wolfe_d_optimal_selection, "FW-D")

    return {
        "N": N, "K": K, "seed": seed, "num_subsets": num_subsets, "time_exhaustive_s": t_exh,
        "true_opt": {"min_eig": best_e, "trace_inv": best_a_trace, "logdet": best_d,
                     "E_indices": best_e_idx, "A_indices": best_a_idx, "D_indices": best_d_idx},
        "greedy_e": greedy_e, "greedy_a": greedy_a, "greedy_d": greedy_d,
        "fw_e": fw_e, "fw_a": fw_a, "fw_d": fw_d,
    }


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def make_latex_table(results: List[Dict]) -> str:
    """Compact multi-criterion comparison table."""

    def fmt(v, big=False):
        if v is None or not np.isfinite(v):
            return r"—"
        if big:
            return f"${v:.0f}$"
        av = abs(v)
        if av >= 100:
            return f"${v:.0f}$"
        if av >= 1:
            return f"${v:.3f}$"
        if av >= 0.01:
            return f"${v:.4f}$"
        exp = int(np.floor(np.log10(av))) if av > 0 else 0
        mant = v / 10**exp
        return r"$" + f"{mant:.2f}" + r"\!\times\!10^{" + str(exp) + r"}$"

    def fmt_gap(g):
        if g is None or not np.isfinite(g):
            return r"—"
        if abs(g) < 0.01:
            return r"$<0.01\%$"
        return f"${g:.2f}\\%$"

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3.5pt}")
    lines.append(r"\small")
    caption = (
        r"Exhaustive comparison: true discrete optimum vs.\ Greedy and FW rounded solutions "
        r"on sub-sampled AprilGrid pools. "
        r"For each pool size $N$ and budget $K$, all $\binom{N}{K}$ subsets are enumerated "
        r"under three criteria: E-optimal ($\lambda_{\min}$, $\uparrow$), "
        r"A-optimal ($\mathrm{tr}(H^{-1})$, $\downarrow$), "
        r"D-optimal ($\log\det H$, $\uparrow$). "
        r"Gap: percentage suboptimality relative to true optimum. "
        r"Relax: FW continuous relaxation bound."
    )
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{tab:exhaustive_comparison}")
    lines.append(r"\begin{tabular}{llr r r r r r}")
    lines.append(r"\toprule")
    lines.append(r"$N$ & $K$ & Crit & $\binom{N}{K}$ & True opt & "
                 r"Greedy & Gap & FW-rnd & Gap & Relax \\")
    lines.append(r"\midrule")

    # fix header to 9 cols
    lines[-1] = (r"$N$ & $K$ & Crit & $\binom{N}{K}$ & True opt & "
                 r"Greedy & Gap & FW-rnd & Gap & Relax \\")
    lines.append(r"\begin{tabular}{llrr r rr rr r}")
    # rebuild properly
    lines = lines[:-8]  # pop the tabular attempts
    lines.append(r"\begin{tabular}{clcrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"$N{,}K$ & Crit & $\binom{N}{K}$ & True opt & Greedy & Grdy gap & FW-rnd & FW gap & Relax \\")
    lines.append(r"\midrule")

    for r in results:
        N, K = r["N"], r["K"]
        nc = r["num_subsets"]
        to = r["true_opt"]
        prefix = f"$({N},{K})$"

        rows3 = [
            ("E", to["min_eig"], r["greedy_e"]["min_eig"], r["fw_e"]["min_eig"], r["fw_e"]["relax_min_eig"], True),
            ("A", to["trace_inv"], r["greedy_a"]["trace_inv"], r["fw_a"]["trace_inv"], r["fw_a"]["relax_trace_inv"], False),
            ("D", to["logdet"],   r["greedy_d"]["logdet"],   r["fw_d"]["logdet"],   r["fw_d"]["relax_logdet"],   True),
        ]
        for ci, (crit, true_v, g_v, fw_v, rel_v, maximize) in enumerate(rows3):
            nk_col = prefix if ci == 0 else ""
            nc_col = f"${nc:,}$".replace(",", r"{,}") if ci == 0 else ""
            g_gap = gap_pct(g_v, true_v, maximize)
            fw_gap = gap_pct(fw_v, true_v, maximize)
            lines.append(
                f"{nk_col} & {crit} & {nc_col} & {fmt(true_v)} & {fmt(g_v)} & {fmt_gap(g_gap)} "
                f"& {fmt(fw_v)} & {fmt_gap(fw_gap)} & {fmt(rel_v)} \\\\"
            )
        lines.append(r"\addlinespace[3pt]")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SETTINGS = [
    (30, 5, 0),
    (40, 5, 1),
    (50, 5, 2),
    (40, 6, 3),
    (30, 8, 4),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", default="results/k20/datasets/aprilgrid_zero/all_candidate_problem.npz")
    parser.add_argument("--output", default="/tmp/exhaustive_comparison.json")
    parser.add_argument("--settings", type=str, default="all",
                        help="comma-separated list of 'N,K' or 'all'")
    args = parser.parse_args()

    fim.set_information_backend("numeric")
    problem_full = fim.load_calibration_problem_npz(args.problem)
    prior_full = fim.build_prior_blocks(problem_full)

    if args.settings == "all":
        settings = SETTINGS
    else:
        settings = []
        for s in args.settings.split(";"):
            n, k = s.split(",")
            settings.append((int(n), int(k), len(settings)))

    print(f"\n=== Exhaustive Comparison ===")
    print(f"  Full problem: N={problem_full.num_candidates}")
    print(f"  Settings: {[(s[0], s[1]) for s in settings]}")

    all_results = []
    for N, K, seed in settings:
        print(f"\n{'='*55}")
        t0 = time.time()
        r = run_one(problem_full, N, K, seed, prior_full)
        print(f"  Total time: {time.time()-t0:.1f}s")
        all_results.append(r)

    # Print summary
    print("\n=== Summary ===")
    print(f"{'N,K':<8} {'Crit':<4} {'True opt':>12} {'Greedy gap%':>12} {'FW gap%':>10} {'Relax':>12}")
    for r in all_results:
        N, K = r["N"], r["K"]
        to = r["true_opt"]
        rows = [
            ("E", to["min_eig"], r["greedy_e"]["min_eig"], r["fw_e"]["min_eig"], r["fw_e"]["relax_min_eig"], True),
            ("A", to["trace_inv"], r["greedy_a"]["trace_inv"], r["fw_a"]["trace_inv"], r["fw_a"]["relax_trace_inv"], False),
            ("D", to["logdet"], r["greedy_d"]["logdet"], r["fw_d"]["logdet"], r["fw_d"]["relax_logdet"], True),
        ]
        for crit, tv, gv, fv, rv, mx in rows:
            gg = gap_pct(gv, tv, mx)
            fg = gap_pct(fv, tv, mx)
            print(f"({N},{K})   {crit:<4} {tv:>12.4e} {gg:>12.2f}% {fg:>10.2f}% {rv:>12.4e}")

    # Save JSON
    def _ser(v):
        if isinstance(v, np.integer): return int(v)
        if isinstance(v, (np.floating, float)): return float(v) if np.isfinite(v) else None
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, list): return [_ser(x) for x in v]
        if isinstance(v, dict): return {k2: _ser(v2) for k2, v2 in v.items()}
        return v

    pathlib.Path(args.output).write_text(json.dumps(_ser(all_results), indent=2))
    print(f"\nSaved: {args.output}")

    # Generate LaTeX
    tex = make_latex_table(all_results)
    tex_path = pathlib.Path(args.output).with_suffix(".tex")
    tex_path.write_text(tex)
    print(f"LaTeX: {tex_path}")


if __name__ == "__main__":
    main()
