"""Generate LaTeX tables (Tables 1-10) for K=20 benchmark results.

Usage:
    python3 -m Experiments.generate_k20_tables \
        --results-dir results/k20 \
        --output-dir results/k20_analysis
"""

from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from typing import Dict, Any, List

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METHOD_ORDER = ["fw-e", "greedy-e", "a-optimal", "d-optimal", "random",
                "coverage", "motion-diversity"]

METHOD_LABELS = {
    "fw-e":             r"\textbf{FW-E}",
    "greedy-e":         r"Greedy-E",
    "a-optimal":        r"A-optimal",
    "d-optimal":        r"D-optimal",
    "random":           r"Random (mean)",
    "coverage":         r"Coverage",
    "motion-diversity": r"Motion-div.",
}

# Tables 1-6 definitions
TABLE_DEFS = [
    ("aprilgrid_noiseless",
     "tab:aprilgrid_noiseless",
     r"AprilGrid, near-noiseless ($\sigma{=}0.01$\,px). $N{=}720$, $K{=}20$. "
     r"Param-error and held-out RMS in pixels. Random: mean over 25 trials."),

    ("charuco_noiseless",
     "tab:charuco_noiseless",
     r"Matched ChArUco board ($13{\times}13$, 144\,pts, 46\,cm), "
     r"near-noiseless ($\sigma{=}0.01$\,px). $N{=}720$, $K{=}20$. "
     r"Param-error and held-out RMS in pixels. Random: mean over 25 trials."),

    ("aprilgrid_sigma05",
     "tab:aprilgrid_sigma05",
     r"AprilGrid, realistic noise ($\sigma{=}0.5$~px). $N{=}720$, $K{=}20$."),

    ("charuco_sigma05",
     "tab:charuco_sigma05",
     r"Matched ChArUco board ($13{\times}13$, 144\,pts, 46\,cm), "
     r"realistic noise ($\sigma{=}0.5$~px). $N{=}720$, $K{=}20$."),

    ("aprilgrid_sigma10",
     "tab:aprilgrid_sigma10",
     r"AprilGrid, $\sigma{=}1.0$~px noise. $N{=}720$, $K{=}20$."),

    ("charuco_sigma10",
     "tab:charuco_sigma10",
     r"Matched ChArUco board ($13{\times}13$, 144\,pts, 46\,cm), "
     r"$\sigma{=}1.0$~px noise. $N{=}720$, $K{=}20$."),
]

# Distortion table column structure
DISTORTION_TABLES = {
    "medium": {
        "label": "tab:distortion_medium",
        "caption": (r"Medium distortion ($k_1{=}{-}0.15$, $k_2{=}0.05$, "
                    r"$p_1{=}p_2{=}0.001$). "
                    r"$\lambda_{\min}(H_{\rm cal})$, $N{=}720$, $K{=}20$. "
                    r"Bold: best per column within each target."),
        "aprilgrid_keys": ["aprilgrid_medium_noiseless", "aprilgrid_medium_sigma05", "aprilgrid_medium_sigma10"],
        "charuco_keys":   ["charuco_medium_noiseless",   "charuco_medium_sigma05",   "charuco_medium_sigma10"],
    },
    "high": {
        "label": "tab:distortion_high",
        "caption": (r"High distortion ($k_1{=}{-}0.35$, $k_2{=}0.15$, "
                    r"$p_1{=}p_2{=}0.002$). "
                    r"$\lambda_{\min}(H_{\rm cal})$, $N{=}720$, $K{=}20$. "
                    r"${}^*$FW-E rounding failure."),
        "aprilgrid_keys": ["aprilgrid_high_noiseless", "aprilgrid_high_sigma05", "aprilgrid_high_sigma10"],
        "charuco_keys":   ["charuco_high_noiseless",   "charuco_high_sigma05",   "charuco_high_sigma10"],
    },
}

# K-sweep display methods (Table 10)
K_SWEEP_METHODS = ["random", "d-optimal", "greedy-e", "fw-e", "a-optimal"]
K_SWEEP_SHORT_LABELS = {
    "random":          "Random",
    "d-optimal":       "D-opt",
    "greedy-e":        "Greedy-E",
    "fw-e":            "FW-E",
    "a-optimal":       "A-opt",
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_benchmark_results(results_dir: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    data = {}
    for dir_key in set(k for t in TABLE_DEFS for k in [t[0]]) | \
                   {k for cfg in DISTORTION_TABLES.values()
                      for ks in (cfg["aprilgrid_keys"] + cfg["charuco_keys"]) for k in [ks]}:
        pattern = str(results_dir / dir_key / "benchmark_*" / "benchmark_results.json")
        hits = sorted(glob.glob(pattern))
        if not hits:
            continue
        raw = json.load(open(hits[-1]))
        data[dir_key] = {r["method"]: r for r in raw["results"]}
    return data


def load_k_sweep_results(results_dir: pathlib.Path) -> Dict[str, Any]:
    sweep_dir = results_dir / "k_sweep"
    data = {}
    for tag in ["charuco_sigma05", "aprilgrid_sigma05", "aprilgrid_noiseless"]:
        hits = sorted(sweep_dir.glob(f"k_sweep_{tag}_*.json"))
        if not hits:
            continue
        data[tag] = json.load(open(hits[-1]))
    return data


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _sci(v: float) -> str:
    if not np.isfinite(v):
        return r"---"
    if abs(v) >= 1e3 or (abs(v) > 0 and abs(v) < 1e-2):
        exp = int(np.floor(np.log10(abs(v))))
        mantissa = v / 10**exp
        return rf"{mantissa:.2f}\!\times\!10^{{{exp}}}"
    return f"{v:.3g}"


def _fmt(v: float, bold: bool = False) -> str:
    if not np.isfinite(v):
        return r"---"
    inner = _sci(v)
    s = f"${inner}$" if (r"\times" in inner) else inner
    return (r"\textbf{" + s + "}") if bold else s


def _find_winners(vals: list, higher: bool, tol: float = 0.005) -> set:
    finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
    if not finite:
        return set()
    best = max(v for _, v in finite) if higher else min(v for _, v in finite)
    threshold = abs(best) * tol
    return {i for i, v in finite if abs(v - best) <= threshold}


# ---------------------------------------------------------------------------
# LaTeX generators
# ---------------------------------------------------------------------------

def make_benchmark_table(dir_key: str, by_method: Dict[str, Any],
                         label: str, caption: str) -> str:
    # Columns: min_eig, logdet, focal_err_px, pp_err_px, dist_err, rot_err_deg, trans_err_cm, train_rms, heldout_rms, time_s
    col_keys = ["min_eig", "logdet", "focal_err_px", "pp_err_px",
                "dist_err", "rot_err_deg", "trans_err_cm",
                "train_rms", "heldout_rms", "time_s"]
    # higher_better: True for FIM metrics, False for error metrics; time excluded from bolding
    higher_better = [True, True, False, False, False, False, False, False, False]

    rows_data = []
    for m in METHOD_ORDER:
        r = by_method.get(m, {})
        rows_data.append([m] + [r.get(k, float("nan")) for k in col_keys])

    winners = [_find_winners([row[i + 1] for row in rows_data], hb)
               for i, hb in enumerate(higher_better)]

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\caption{" + caption + r" Focal/PP errors in pixels, distortion error unitless, "
        r"rotation error in degrees, translation error in cm.}",
        r"\label{" + label + "}",
        r"\begin{tabular}{lrrrrrrrrr}",
        r"\toprule",
        (r"Method & $\lambda_{\min}$ & $\log\det$ & "
         r"$\epsilon_f$\,(px) & $\epsilon_{pp}$\,(px) & "
         r"$\epsilon_d$ & $\epsilon_R$\,($^\circ$) & $\epsilon_t$\,(cm) & "
         r"Train-RMS & Held-out \\"),
        r"\midrule",
    ]
    for row_i, row in enumerate(rows_data):
        m = row[0]
        label_str = METHOD_LABELS[m]
        values = row[1:]  # 10 values: 9 metrics + time
        formatted = []
        for ci, v in enumerate(values[:-1]):   # skip time (last col removed from display)
            bold = (row_i in winners[ci]) if ci < len(winners) else False
            formatted.append(_fmt(v, bold=bold))
        wins_any = any(row_i in winners[c] for c in range(len(higher_better) - 1))
        if wins_any and r"\textbf" not in label_str:
            label_str = r"\textbf{" + label_str + "}"
        lines.append(f"{label_str} & {' & '.join(formatted)} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def make_distortion_table(dist_key: str, all_data: dict) -> str:
    cfg = DISTORTION_TABLES[dist_key]
    sigma_labels = [r"\sigma{=}0/0.01", r"\sigma{=}0.5", r"\sigma{=}1.0"]
    methods_shown = ["fw-e", "greedy-e", "a-optimal", "d-optimal", "random"]
    method_label_short = {
        "fw-e": "FW-E", "greedy-e": "Greedy-E", "a-optimal": "A-opt",
        "d-optimal": "D-opt", "random": "Random",
    }

    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\small",
        r"\caption{" + cfg["caption"] + "}",
        r"\label{" + cfg["label"] + "}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r" & \multicolumn{3}{c}{AprilGrid} & \multicolumn{3}{c}{Matched ChArUco} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        r"Method & " + " & ".join(rf"${sl}$" for sl in sigma_labels) +
        r" & " + " & ".join(rf"${sl}$" for sl in sigma_labels) + r" \\",
        r"\midrule",
    ]

    for m in methods_shown:
        ag_vals = [all_data.get(k, {}).get(m, {}).get("min_eig", float("nan"))
                   for k in cfg["aprilgrid_keys"]]
        cu_vals = [all_data.get(k, {}).get(m, {}).get("min_eig", float("nan"))
                   for k in cfg["charuco_keys"]]
        all_vals = ag_vals + cu_vals
        ag_winners = [_find_winners(
            [all_data.get(cfg["aprilgrid_keys"][ci], {}).get(mm, {}).get("min_eig", float("nan"))
             for mm in methods_shown], True)
            for ci in range(3)]
        cu_winners = [_find_winners(
            [all_data.get(cfg["charuco_keys"][ci], {}).get(mm, {}).get("min_eig", float("nan"))
             for mm in methods_shown], True)
            for ci in range(3)]
        m_idx = methods_shown.index(m)
        ag_cells = [_fmt(v, bold=(m_idx in ag_winners[ci])) for ci, v in enumerate(ag_vals)]
        cu_cells = [_fmt(v, bold=(m_idx in cu_winners[ci])) for ci, v in enumerate(cu_vals)]
        label_str = method_label_short[m]
        row = f"{label_str} & {' & '.join(ag_cells)} & {' & '.join(cu_cells)} \\\\"
        lines.append(row)

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def make_fw_convergence_table(all_data: dict) -> str:
    """Table 7: FW-E convergence. Reads from benchmark JSON's fw-e entry (no iter count stored)."""
    rows = [
        # (label, dir_key, sigma_label, noiseless)
        (r"ChArUco (matched)", "charuco_noiseless",   r"0.01",  True),
        (r"ChArUco (matched)", "charuco_sigma05",     r"0.5",   False),
        (r"ChArUco (matched)", "charuco_sigma10",     r"1.0",   False),
        (r"AprilGrid",         "aprilgrid_noiseless", r"0",     True),
        (r"AprilGrid",         "aprilgrid_sigma05",   r"0.5",   False),
        (r"AprilGrid",         "aprilgrid_sigma10",   r"1.0",   False),
    ]
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{FW-E convergence summary ($N{=}720$, $K{=}20$). "
        r"At $\sigma{=}1.0$\,px both targets, the high-noise landscape is nearly flat "
        r"and the duality gap closes early; rounding degrades $\lambda_{\min}$ significantly.}",
        r"\label{tab:fw_convergence}",
        r"\setlength{\tabcolsep}{5pt}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Target & $\sigma$ & Time (s) & Rounded $\lambda_{\min}$ & Held-out RMS \\",
        r"\midrule",
    ]
    for target_label, dir_key, sigma_label, _ in rows:
        r = all_data.get(dir_key, {}).get("fw-e", {})
        ts = r.get("time_s", float("nan"))
        lam = r.get("min_eig", float("nan"))
        hout = r.get("heldout_rms", float("nan"))
        lines.append(
            rf"{target_label} & {sigma_label} & "
            f"{ts:.2f} & ${_sci(lam)}$ & ${_sci(hout)}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def make_k_sweep_table(sweep_data: dict) -> str:
    """Table 10: K-budget sweep with 3 sub-tables."""
    select_ks = [5, 10, 15, 20, 30, 50, 100, 200]

    def _get(tag, method, K):
        if tag not in sweep_data:
            return float("nan")
        r = sweep_data[tag].get("results", {}).get(method, {}).get(str(K), {})
        return float(r.get("min_eig", float("nan")) or float("nan"))

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\small",
        r"\caption{$\lambda_{\min}(H_{\mathrm{cal}})$ vs.\ budget $K$. "
        r"Top: ChArUco, $\sigma{=}0.5$\,px, $N{=}720$. "
        r"Middle: AprilGrid, $\sigma{=}0.5$\,px, $N{=}720$. "
        r"Bottom: AprilGrid, $\sigma{=}0.01$\,px (near-noiseless), $N{=}720$. "
        r"Bold: best per column. ${}^*$FW-E rounding failure.}",
        r"\label{tab:k_sweep}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{l" + "r" * len(select_ks) + "}",
        r"\toprule",
        "Method & " + " & ".join(f"$K{{{{{k}}}}}$" for k in select_ks) + r" \\",
        r"\midrule",
    ]

    configs = [
        ("charuco_sigma05",    r"\textit{ChArUco, $\sigma=0.5$\,px}"),
        ("aprilgrid_sigma05",  r"\textit{AprilGrid, $\sigma=0.5$\,px}"),
        ("aprilgrid_noiseless",r"\textit{AprilGrid, $\sigma\approx0$\,px}"),
    ]

    for tag, section_label in configs:
        lines.append(rf"\multicolumn{{{len(select_ks)+1}}}{{l}}{{{section_label}}} \\[2pt]")
        for m in K_SWEEP_METHODS:
            vals = [_get(tag, m, K) for K in select_ks]
            winners = _find_winners(vals, higher=True)
            row_cells = []
            for i, v in enumerate(vals):
                cell = _sci(v)
                if np.isfinite(v) and i in winners:
                    cell = r"\mathbf{" + cell + "}"
                row_cells.append(f"${cell}$" if np.isfinite(v) else "---")
            lines.append(f"{K_SWEEP_SHORT_LABELS[m]} & {' & '.join(row_cells)} \\\\")
        lines.append(r"\midrule")

    # Remove last midrule, add bottomrule
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/k20", type=pathlib.Path)
    parser.add_argument("--output-dir",  default="results/k20_analysis", type=pathlib.Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading benchmark results from {args.results_dir} ...")
    all_data = load_benchmark_results(args.results_dir)
    print(f"  loaded {len(all_data)} configurations")

    print("Loading K-sweep results ...")
    sweep_data = load_k_sweep_results(args.results_dir)
    print(f"  loaded {len(sweep_data)} sweep configs")

    tex_parts = [
        "% Auto-generated K=20 tables for ICRA 2026\n",
        r"% Method order: FW-E, Greedy-E, A-optimal, D-optimal, Random, Coverage, Motion-div.",
        "\n",
    ]

    # Tables 1-6: main benchmarks
    print("\nGenerating Tables 1-6 (main benchmarks) ...")
    for i, (dir_key, label, caption) in enumerate(TABLE_DEFS, 1):
        if dir_key not in all_data:
            print(f"  [skip] Table {i}: {dir_key}")
            continue
        print(f"  Table {i}: {dir_key}")
        tex_parts.append(f"% ---- Table {i}: {dir_key} ----\n")
        tex_parts.append(make_benchmark_table(dir_key, all_data[dir_key], label, caption))

    # Table 7: FW-E convergence
    print("Generating Table 7 (FW-E convergence) ...")
    tex_parts.append("% ---- Table 7: FW-E convergence ----\n")
    tex_parts.append(make_fw_convergence_table(all_data))

    # Tables 8-9: distortion
    print("Generating Tables 8-9 (distortion) ...")
    tex_parts.append("% ---- Table 8: Medium distortion ----\n")
    tex_parts.append(make_distortion_table("medium", all_data))
    tex_parts.append("% ---- Table 9: High distortion ----\n")
    tex_parts.append(make_distortion_table("high", all_data))

    # Table 10: K-sweep
    print("Generating Table 10 (K-budget sweep) ...")
    tex_parts.append("% ---- Table 10: K-budget sweep ----\n")
    tex_parts.append(make_k_sweep_table(sweep_data))

    tex_path = args.output_dir / "tables_k20.tex"
    tex_path.write_text("\n".join(tex_parts))
    print(f"\nLaTeX saved: {tex_path}")

    # Also print a plain-text summary of Tables 1-6
    print("\n" + "="*100)
    print("PLAIN-TEXT SUMMARY OF TABLES 1-6")
    print("="*100)
    cols = ["min_eig", "logdet", "focal_err_px", "pp_err_px",
            "dist_err", "rot_err_deg", "trans_err_cm", "train_rms", "heldout_rms"]
    hdr = f"  {'Method':<18} " + " ".join(f"{c:>13}" for c in cols)
    for dir_key, label, caption in TABLE_DEFS:
        if dir_key not in all_data:
            continue
        print(f"\n{dir_key}:")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        bm = all_data[dir_key]
        for m in METHOD_ORDER:
            r = bm.get(m, {})
            vals = [r.get(c, float("nan")) for c in cols]
            row = f"  {m:<18} " + " ".join(
                f"{v:>13.4e}" if np.isfinite(v) else f"{'---':>13}" for v in vals
            )
            print(row)

    print(f"\nAll outputs saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
