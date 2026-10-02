"""Generate LaTeX tables (Tables 1-9) and plots for K=25 benchmark results.

Usage:
    python3 -m Experiments.generate_k25_tables_and_plots \
        --results-dir results/k25 \
        --output-dir results/k25_analysis
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
from typing import Dict, Any, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fixed method display order for ALL tables
METHOD_ORDER = ["fw-e", "greedy-e", "a-optimal", "d-optimal", "random",
                "coverage", "motion-diversity"]

METHOD_LABELS = {
    "fw-e":             r"FW-E",
    "greedy-e":         r"Greedy-E",
    "a-optimal":        r"A-optimal",
    "d-optimal":        r"D-optimal",
    "random":           r"Random (mean)",
    "coverage":         r"Coverage",
    "motion-diversity": r"Motion-div.",
}

# Colors for plots (same order as METHOD_ORDER)
METHOD_COLORS = {
    "fw-e":             "#e41a1c",
    "greedy-e":         "#377eb8",
    "a-optimal":        "#4daf4a",
    "d-optimal":        "#984ea3",
    "random":           "#ff7f00",
    "coverage":         "#a65628",
    "motion-diversity": "#999999",
}

METHOD_MARKERS = {
    "fw-e": "o", "greedy-e": "s", "a-optimal": "^", "d-optimal": "D",
    "random": "x", "coverage": "+", "motion-diversity": "v",
}

# Directory-name → (label, k1 distortion level)
DIR_CONFIG = {
    "aprilgrid_noiseless":        ("AprilGrid",        r"\sigma{=}0",     0.0,  "zero_dist"),
    "charuco_noiseless":          ("Matched ChArUco",  r"\sigma{=}0.01",  0.0,  "zero_dist"),
    "aprilgrid_sigma05":          ("AprilGrid",        r"\sigma{=}0.5",   0.0,  "zero_dist"),
    "charuco_sigma05":            ("Matched ChArUco",  r"\sigma{=}0.5",   0.0,  "zero_dist"),
    "aprilgrid_sigma10":          ("AprilGrid",        r"\sigma{=}1.0",   0.0,  "zero_dist"),
    "charuco_sigma10":            ("Matched ChArUco",  r"\sigma{=}1.0",   0.0,  "zero_dist"),
    "aprilgrid_medium_noiseless": ("AprilGrid",        r"\sigma{=}0",     -0.15, "medium_dist"),
    "charuco_medium_noiseless":   ("Matched ChArUco",  r"\sigma{=}0.01",  -0.15, "medium_dist"),
    "aprilgrid_medium_sigma05":   ("AprilGrid",        r"\sigma{=}0.5",   -0.15, "medium_dist"),
    "charuco_medium_sigma05":     ("Matched ChArUco",  r"\sigma{=}0.5",   -0.15, "medium_dist"),
    "aprilgrid_medium_sigma10":   ("AprilGrid",        r"\sigma{=}1.0",   -0.15, "medium_dist"),
    "charuco_medium_sigma10":     ("Matched ChArUco",  r"\sigma{=}1.0",   -0.15, "medium_dist"),
    "aprilgrid_high_noiseless":   ("AprilGrid",        r"\sigma{=}0",     -0.35, "high_dist"),
    "charuco_high_noiseless":     ("Matched ChArUco",  r"\sigma{=}0.01",  -0.35, "high_dist"),
    "aprilgrid_high_sigma05":     ("AprilGrid",        r"\sigma{=}0.5",   -0.35, "high_dist"),
    "charuco_high_sigma05":       ("Matched ChArUco",  r"\sigma{=}0.5",   -0.35, "high_dist"),
    "aprilgrid_high_sigma10":     ("AprilGrid",        r"\sigma{=}1.0",   -0.35, "high_dist"),
    "charuco_high_sigma10":       ("Matched ChArUco",  r"\sigma{=}1.0",   -0.35, "high_dist"),
}

# Table 1-9 structure: (dir_key, table_label, caption_text)
TABLE_DEFS = [
    # Tables 1-6: main experiments
    ("aprilgrid_noiseless",
     "tab:aprilgrid_noiseless",
     r"AprilGrid, noiseless ($\sigma{=}0$). $N{=}720$, $K{=}25$. "
     r"Param-error and held-out RMS in pixels. Random: mean over 25 trials."),

    ("charuco_noiseless",
     "tab:charuco_noiseless",
     r"Matched ChArUco board ($13{\times}13$, 144\,pts, 46\,cm), "
     r"near-noiseless ($\sigma{=}0.01$\,px). $N{=}720$, $K{=}25$. "
     r"Param-error and held-out RMS in pixels. Random: mean over 25 trials."),

    ("aprilgrid_sigma05",
     "tab:aprilgrid_sigma05",
     r"AprilGrid, realistic noise ($\sigma{=}0.5$~px). $N{=}720$, $K{=}25$."),

    ("charuco_sigma05",
     "tab:charuco_sigma05",
     r"Matched ChArUco board ($13{\times}13$, 144\,pts, 46\,cm), "
     r"realistic noise ($\sigma{=}0.5$~px). $N{=}720$, $K{=}25$."),

    ("aprilgrid_sigma10",
     "tab:aprilgrid_sigma10",
     r"AprilGrid, $\sigma{=}1.0$~px noise. $N{=}720$, $K{=}25$."),

    ("charuco_sigma10",
     "tab:charuco_sigma10",
     r"Matched ChArUco board ($13{\times}13$, 144\,pts, 46\,cm), "
     r"$\sigma{=}1.0$~px noise. $N{=}720$, $K{=}25$."),

    # Tables 7-9 distortion (now 6 tables, labelled 7a/b, 8a/b, 9a/b)
    # Medium distortion — AprilGrid
    ("aprilgrid_medium_noiseless",
     "tab:aprilgrid_medium_noiseless",
     r"AprilGrid, medium distortion ($k_1{=}{-}0.15$, $k_2{=}0.05$, "
     r"$p_1{=}p_2{=}0.001$), $\sigma{=}0$. $N{=}720$, $K{=}25$."),

    ("aprilgrid_medium_sigma05",
     "tab:aprilgrid_medium_sigma05",
     r"AprilGrid, medium distortion, $\sigma{=}0.5$~px. $N{=}720$, $K{=}25$."),

    ("aprilgrid_medium_sigma10",
     "tab:aprilgrid_medium_sigma10",
     r"AprilGrid, medium distortion, $\sigma{=}1.0$~px. $N{=}720$, $K{=}25$."),

    # Medium distortion — ChArUco
    ("charuco_medium_noiseless",
     "tab:charuco_medium_noiseless",
     r"Matched ChArUco, medium distortion ($k_1{=}{-}0.15$, $k_2{=}0.05$, "
     r"$p_1{=}p_2{=}0.001$), $\sigma{=}0.01$. $N{=}720$, $K{=}25$."),

    ("charuco_medium_sigma05",
     "tab:charuco_medium_sigma05",
     r"Matched ChArUco, medium distortion, $\sigma{=}0.5$~px. $N{=}720$, $K{=}25$."),

    ("charuco_medium_sigma10",
     "tab:charuco_medium_sigma10",
     r"Matched ChArUco, medium distortion, $\sigma{=}1.0$~px. $N{=}720$, $K{=}25$."),

    # High distortion — AprilGrid
    ("aprilgrid_high_noiseless",
     "tab:aprilgrid_high_noiseless",
     r"AprilGrid, high distortion ($k_1{=}{-}0.35$, $k_2{=}0.15$, "
     r"$p_1{=}p_2{=}0.002$), $\sigma{=}0$. $N{=}720$, $K{=}25$."),

    ("aprilgrid_high_sigma05",
     "tab:aprilgrid_high_sigma05",
     r"AprilGrid, high distortion, $\sigma{=}0.5$~px. $N{=}720$, $K{=}25$."),

    ("aprilgrid_high_sigma10",
     "tab:aprilgrid_high_sigma10",
     r"AprilGrid, high distortion, $\sigma{=}1.0$~px. $N{=}720$, $K{=}25$."),

    # High distortion — ChArUco
    ("charuco_high_noiseless",
     "tab:charuco_high_noiseless",
     r"Matched ChArUco, high distortion ($k_1{=}{-}0.35$, $k_2{=}0.15$, "
     r"$p_1{=}p_2{=}0.002$), $\sigma{=}0.01$. $N{=}720$, $K{=}25$."),

    ("charuco_high_sigma05",
     "tab:charuco_high_sigma05",
     r"Matched ChArUco, high distortion, $\sigma{=}0.5$~px. $N{=}720$, $K{=}25$."),

    ("charuco_high_sigma10",
     "tab:charuco_high_sigma10",
     r"Matched ChArUco, high distortion, $\sigma{=}1.0$~px. $N{=}720$, $K{=}25$."),
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(results_dir: pathlib.Path) -> Dict[str, Dict[str, Any]]:
    """Return {dir_key: {method: result_dict}} for all found benchmarks."""
    data = {}
    for dir_key in DIR_CONFIG:
        pattern = str(results_dir / dir_key / "benchmark_*" / "benchmark_results.json")
        hits = sorted(glob.glob(pattern))
        if not hits:
            print(f"  [warn] no results for {dir_key}")
            continue
        raw = json.load(open(hits[-1]))  # most recent
        by_method = {r["method"]: r for r in raw["results"]}
        data[dir_key] = by_method
    return data


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _sci(v: float, prec: int = 2) -> str:
    if not np.isfinite(v):
        return r"---"
    if abs(v) >= 1e3 or (abs(v) > 0 and abs(v) < 1e-2):
        exp = int(np.floor(np.log10(abs(v))))
        mantissa = v / 10**exp
        if prec == 2:
            return rf"{mantissa:.2f}\!\times\!10^{{{exp}}}"
        else:
            return rf"{mantissa:.1f}\!\times\!10^{{{exp}}}"
    return f"{v:.3g}"


def _fmt(v: float, bold: bool = False) -> str:
    """Auto-format with optional bolding.
    Scientific notation values are emitted as math mode; plain decimals as text."""
    if not np.isfinite(v):
        return r"---"
    if abs(v) >= 1e3 or (abs(v) > 0 and abs(v) < 1e-2):
        inner = _sci(v)
        return r"$\mathbf{" + inner + r"}$" if bold else "$" + inner + "$"
    s = f"{v:.3g}"
    return r"\textbf{" + s + "}" if bold else s


def _find_winners(vals: list, higher_is_better: bool, tol_frac: float = 0.002) -> set:
    """Return set of indices into vals that tie for best value (within tol_frac relative)."""
    finite = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
    if not finite:
        return set()
    best = max(v for _, v in finite) if higher_is_better else min(v for _, v in finite)
    tol = abs(best) * tol_frac
    return {i for i, v in finite if abs(v - best) <= tol}


# ---------------------------------------------------------------------------
# LaTeX table generator
# ---------------------------------------------------------------------------

def make_latex_table(dir_key: str, by_method: Dict[str, Any],
                     label: str, caption: str) -> str:
    """Generate a full 7-method LaTeX table (like Tables 1-6 in the paper)."""
    # Collect rows in fixed order
    rows_data = []  # (method, min_eig, logdet, param_error, train_rms, heldout_rms, time_s)
    for m in METHOD_ORDER:
        r = by_method.get(m, {})
        rows_data.append((
            m,
            r.get("min_eig", float("nan")),
            r.get("logdet", float("nan")),
            r.get("param_error", float("nan")),
            r.get("train_rms", float("nan")),
            r.get("heldout_rms", float("nan")),
            r.get("time_s", float("nan")),
        ))

    # Identify winners (col indices 1-6; higher_better: T,T,F,F,F,-)
    higher_better = [True, True, False, False, False, False]
    winners = [_find_winners([r[i+1] for r in rows_data], hb)
               for i, hb in enumerate(higher_better)]
    # winners[i] is the set of row indices that win column i+1

    lines = []
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{5pt}")
    lines.append(r"\caption{" + caption + "}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular}{lrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Method & $\lambda_{\min}$ & $\log\det$ & Param-err & Train-RMS & Held-out & Time (s)\\")
    lines.append(r"\midrule")

    for row_i, (m, min_eig, logdet, pe, tr, ho, ts) in enumerate(rows_data):
        label_str = METHOD_LABELS[m]
        cells = [min_eig, logdet, pe, tr, ho, ts]
        formatted = []
        for col_i, v in enumerate(cells):
            if col_i == 5:  # time: no bolding
                formatted.append(f"{v:.2f}" if np.isfinite(v) else "---")
            else:
                bold = row_i in winners[col_i]
                formatted.append(_fmt(v, bold=bold))

        # Bold the method name if it wins any non-time metric
        wins_any = any(row_i in winners[c] for c in range(5))
        if wins_any:
            label_str = r"\textbf{" + label_str + "}"

        line = f"{label_str} & {' & '.join(formatted)} \\\\"
        lines.append(line)

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

def plot_param_error_by_noise(all_data: Dict[str, Dict[str, Any]],
                               out_dir: pathlib.Path) -> None:
    """Bar charts: param-error per method, grouped by noise level. One figure per target×distortion."""
    configs = [
        # (dir_keys_in_order, title, filename)
        (["aprilgrid_noiseless", "aprilgrid_sigma05", "aprilgrid_sigma10"],
         "AprilGrid — Zero Distortion", "param_error_aprilgrid_nodist"),
        (["charuco_noiseless", "charuco_sigma05", "charuco_sigma10"],
         "Matched ChArUco — Zero Distortion", "param_error_charuco_nodist"),
        (["aprilgrid_medium_noiseless", "aprilgrid_medium_sigma05", "aprilgrid_medium_sigma10"],
         "AprilGrid — Medium Distortion ($k_1=-0.15$)", "param_error_aprilgrid_medium"),
        (["charuco_medium_noiseless", "charuco_medium_sigma05", "charuco_medium_sigma10"],
         "Matched ChArUco — Medium Distortion", "param_error_charuco_medium"),
        (["aprilgrid_high_noiseless", "aprilgrid_high_sigma05", "aprilgrid_high_sigma10"],
         "AprilGrid — High Distortion ($k_1=-0.35$)", "param_error_aprilgrid_high"),
        (["charuco_high_noiseless", "charuco_high_sigma05", "charuco_high_sigma10"],
         "Matched ChArUco — High Distortion", "param_error_charuco_high"),
    ]
    noise_labels = [r"$\sigma=0/0.01$", r"$\sigma=0.5$~px", r"$\sigma=1.0$~px"]

    for dir_keys, title, fname in configs:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=False)
        fig.suptitle(title, fontsize=13, fontweight="bold")

        for ax, dkey, nlabel in zip(axes, dir_keys, noise_labels):
            if dkey not in all_data:
                ax.set_visible(False)
                continue
            bm = all_data[dkey]
            methods_present = [m for m in METHOD_ORDER if m in bm]
            vals = [bm[m].get("param_error", float("nan")) for m in methods_present]
            colors = [METHOD_COLORS[m] for m in methods_present]
            short_labels = [METHOD_LABELS[m].replace("(mean)", "").strip()
                            .replace(r"\textbf{", "").replace("}", "") for m in methods_present]

            x = np.arange(len(methods_present))
            bars = ax.bar(x, vals, color=colors, edgecolor="k", linewidth=0.5)
            # Highlight minimum
            finite_vals = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
            if finite_vals:
                best_i = min(finite_vals, key=lambda t: t[1])[0]
                bars[best_i].set_edgecolor("gold")
                bars[best_i].set_linewidth(2.5)

            ax.set_xticks(x)
            ax.set_xticklabels(short_labels, rotation=40, ha="right", fontsize=8)
            ax.set_title(f"{nlabel}", fontsize=10)
            ax.set_ylabel("Param-error (px)" if ax == axes[0] else "")
            ax.set_xlabel("")
            ax.grid(axis="y", linestyle="--", alpha=0.4)

        plt.tight_layout()
        path = out_dir / f"{fname}.pdf"
        fig.savefig(path, bbox_inches="tight")
        path_png = out_dir / f"{fname}.png"
        fig.savefig(path_png, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path_png.name}")


def plot_lambda_min_by_noise(all_data: Dict[str, Dict[str, Any]],
                              out_dir: pathlib.Path) -> None:
    """Line+marker plot: λ_min per method across noise levels."""
    configs = [
        (["aprilgrid_noiseless", "aprilgrid_sigma05", "aprilgrid_sigma10"],
         "AprilGrid — Zero Distortion", "lambda_aprilgrid_nodist"),
        (["charuco_noiseless", "charuco_sigma05", "charuco_sigma10"],
         "Matched ChArUco — Zero Distortion", "lambda_charuco_nodist"),
        (["aprilgrid_medium_noiseless", "aprilgrid_medium_sigma05", "aprilgrid_medium_sigma10"],
         "AprilGrid — Medium Distortion", "lambda_aprilgrid_medium"),
        (["charuco_medium_noiseless", "charuco_medium_sigma05", "charuco_medium_sigma10"],
         "Matched ChArUco — Medium Distortion", "lambda_charuco_medium"),
        (["aprilgrid_high_noiseless", "aprilgrid_high_sigma05", "aprilgrid_high_sigma10"],
         "AprilGrid — High Distortion", "lambda_aprilgrid_high"),
        (["charuco_high_noiseless", "charuco_high_sigma05", "charuco_high_sigma10"],
         "Matched ChArUco — High Distortion", "lambda_charuco_high"),
    ]
    noise_vals = [0.01, 0.5, 1.0]

    for dir_keys, title, fname in configs:
        fig, ax = plt.subplots(figsize=(7, 4))
        fig.suptitle(title, fontsize=12, fontweight="bold")
        for m in METHOD_ORDER:
            y = []
            for dkey in dir_keys:
                if dkey not in all_data or m not in all_data[dkey]:
                    y.append(float("nan"))
                else:
                    y.append(all_data[dkey][m].get("min_eig", float("nan")))
            ax.plot(noise_vals, y, color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                    label=METHOD_LABELS[m], linewidth=1.8, markersize=6)

        ax.set_xlabel(r"Pixel noise $\sigma$ (px)", fontsize=11)
        ax.set_ylabel(r"$\lambda_{\min}(H_{\rm cal})$", fontsize=11)
        ax.set_yscale("log")
        ax.set_xticks(noise_vals)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        plt.tight_layout()
        path_png = out_dir / f"{fname}.png"
        fig.savefig(path_png, dpi=150, bbox_inches="tight")
        fig.savefig(out_dir / f"{fname}.pdf", bbox_inches="tight")
        plt.close(fig)
        print(f"  saved {path_png.name}")


def plot_lambda_vs_param_error(all_data: Dict[str, Dict[str, Any]],
                                out_dir: pathlib.Path) -> None:
    """Scatter: λ_min vs param-error across all experiments. Each dot = one (method, config)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    dist_groups = [
        ("zero_dist",   "Zero Distortion",   axes[0]),
        ("medium_dist", "Medium Distortion",  axes[1]),
    ]
    # high distortion on same right axis but different marker
    all_targets = {
        (dkey, m): (cfg[3], m)   # (dist_group, method)
        for dkey, cfg in DIR_CONFIG.items()
        for m in METHOD_ORDER
        if dkey in all_data and m in all_data[dkey]
    }

    for dist_group, group_title, ax in dist_groups:
        for m in METHOD_ORDER:
            lam_vals, pe_vals = [], []
            for dkey, cfg in DIR_CONFIG.items():
                if cfg[3] != dist_group:
                    continue
                if dkey not in all_data or m not in all_data[dkey]:
                    continue
                r = all_data[dkey][m]
                lam = r.get("min_eig", float("nan"))
                pe = r.get("param_error", float("nan"))
                if np.isfinite(lam) and np.isfinite(pe):
                    lam_vals.append(lam)
                    pe_vals.append(pe)
            if lam_vals:
                ax.scatter(lam_vals, pe_vals, color=METHOD_COLORS[m],
                           marker=METHOD_MARKERS[m], label=METHOD_LABELS[m],
                           alpha=0.75, s=50, edgecolors="k", linewidths=0.4)

        ax.set_xlabel(r"$\lambda_{\min}(H_{\rm cal})$ (log scale)", fontsize=10)
        ax.set_ylabel("Param-error (px)" if ax == axes[0] else "")
        ax.set_title(group_title, fontsize=11)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(fontsize=7, loc="upper right")

    # High distortion as third panel
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.set_title("High Distortion", fontsize=11)
    for m in METHOD_ORDER:
        lam_vals, pe_vals = [], []
        for dkey, cfg in DIR_CONFIG.items():
            if cfg[3] != "high_dist":
                continue
            if dkey not in all_data or m not in all_data[dkey]:
                continue
            r = all_data[dkey][m]
            lam = r.get("min_eig", float("nan"))
            pe = r.get("param_error", float("nan"))
            if np.isfinite(lam) and np.isfinite(pe):
                lam_vals.append(lam)
                pe_vals.append(pe)
        if lam_vals:
            ax2.scatter(lam_vals, pe_vals, color=METHOD_COLORS[m],
                       marker=METHOD_MARKERS[m], label=METHOD_LABELS[m],
                       alpha=0.75, s=50, edgecolors="k", linewidths=0.4)
    ax2.set_xlabel(r"$\lambda_{\min}(H_{\rm cal})$", fontsize=10)
    ax2.set_ylabel("Param-error (px)", fontsize=10)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.35)
    ax2.legend(fontsize=7)
    fig2.tight_layout()
    fig2.savefig(out_dir / "scatter_lambda_vs_pe_high.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    fig.tight_layout()
    path = out_dir / "scatter_lambda_vs_pe.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "scatter_lambda_vs_pe.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_method_ranking_heatmap(all_data: Dict[str, Dict[str, Any]],
                                 out_dir: pathlib.Path) -> None:
    """Heatmap: rank of each method by param-error across all 18 configs."""
    # Collect for each config: rank of each method (1=best)
    dir_keys_ordered = [
        "aprilgrid_noiseless", "aprilgrid_sigma05", "aprilgrid_sigma10",
        "charuco_noiseless", "charuco_sigma05", "charuco_sigma10",
        "aprilgrid_medium_noiseless", "aprilgrid_medium_sigma05", "aprilgrid_medium_sigma10",
        "charuco_medium_noiseless", "charuco_medium_sigma05", "charuco_medium_sigma10",
        "aprilgrid_high_noiseless", "aprilgrid_high_sigma05", "aprilgrid_high_sigma10",
        "charuco_high_noiseless", "charuco_high_sigma05", "charuco_high_sigma10",
    ]
    col_labels = [
        "AG σ=0", "AG σ=.5", "AG σ=1",
        "CU σ=0", "CU σ=.5", "CU σ=1",
        "AG-M σ=0", "AG-M σ=.5", "AG-M σ=1",
        "CU-M σ=0", "CU-M σ=.5", "CU-M σ=1",
        "AG-H σ=0", "AG-H σ=.5", "AG-H σ=1",
        "CU-H σ=0", "CU-H σ=.5", "CU-H σ=1",
    ]

    rank_matrix = np.full((len(METHOD_ORDER), len(dir_keys_ordered)), float("nan"))
    for col_i, dkey in enumerate(dir_keys_ordered):
        if dkey not in all_data:
            continue
        bm = all_data[dkey]
        pe_vals = [(m, bm[m].get("param_error", float("nan"))) for m in METHOD_ORDER if m in bm]
        valid = [(m, v) for m, v in pe_vals if np.isfinite(v)]
        sorted_methods = [m for m, _ in sorted(valid, key=lambda t: t[1])]
        for row_i, m in enumerate(METHOD_ORDER):
            if m in sorted_methods:
                rank_matrix[row_i, col_i] = sorted_methods.index(m) + 1

    fig, ax = plt.subplots(figsize=(16, 4))
    cmap = plt.cm.RdYlGn_r
    im = ax.imshow(rank_matrix, cmap=cmap, vmin=1, vmax=len(METHOD_ORDER), aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=9)
    ax.set_title("Method ranking by param-error (1=best, green=good, red=bad)\n"
                 "AG=AprilGrid, CU=ChArUco, M=Medium dist, H=High dist", fontsize=10)

    for r in range(rank_matrix.shape[0]):
        for c in range(rank_matrix.shape[1]):
            v = rank_matrix[r, c]
            if np.isfinite(v):
                ax.text(c, r, f"{int(v)}", ha="center", va="center",
                        fontsize=8, color="white" if v >= 5 else "black", fontweight="bold")

    plt.colorbar(im, ax=ax, label="Rank (param-error)", shrink=0.8)
    plt.tight_layout()
    path = out_dir / "heatmap_method_ranks.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "heatmap_method_ranks.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


def plot_distortion_effect(all_data: Dict[str, Dict[str, Any]],
                            out_dir: pathlib.Path) -> None:
    """Compare λ_min and param-error across distortion levels for each method at σ=0.5."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(r"Effect of Distortion at $\sigma=0.5$~px, $K=25$", fontsize=13, fontweight="bold")

    sigma05_keys = {
        "AprilGrid":        ["aprilgrid_sigma05", "aprilgrid_medium_sigma05", "aprilgrid_high_sigma05"],
        "Matched ChArUco":  ["charuco_sigma05",  "charuco_medium_sigma05",  "charuco_high_sigma05"],
    }
    dist_labels = ["None", "Medium\n($k_1=-0.15$)", "High\n($k_1=-0.35$)"]
    dist_x = [0, 1, 2]

    for col_i, (target, keys) in enumerate(sigma05_keys.items()):
        ax_lam = axes[0, col_i]
        ax_pe  = axes[1, col_i]
        for m in METHOD_ORDER:
            lam_y = [all_data.get(k, {}).get(m, {}).get("min_eig", float("nan")) for k in keys]
            pe_y  = [all_data.get(k, {}).get(m, {}).get("param_error", float("nan")) for k in keys]
            kw = dict(color=METHOD_COLORS[m], marker=METHOD_MARKERS[m],
                      label=METHOD_LABELS[m], linewidth=1.5, markersize=6)
            ax_lam.plot(dist_x, lam_y, **kw)
            ax_pe.plot(dist_x, pe_y,  **kw)

        for ax, ylabel, title_suffix in [
            (ax_lam, r"$\lambda_{\min}$", " — E-optimality"),
            (ax_pe,  "Param-error (px)",   " — Param-error"),
        ]:
            ax.set_xticks(dist_x)
            ax.set_xticklabels(dist_labels, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(target + title_suffix, fontsize=10)
            ax.set_yscale("log")
            ax.grid(True, which="both", linestyle="--", alpha=0.4)
            if col_i == 1:
                ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    path = out_dir / "distortion_effect_sigma05.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    fig.savefig(out_dir / "distortion_effect_sigma05.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/k25", type=pathlib.Path)
    parser.add_argument("--output-dir",  default="results/k25_analysis", type=pathlib.Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading results from {args.results_dir} ...")
    all_data = load_results(args.results_dir)
    print(f"  loaded {len(all_data)} configurations")

    # ---- LaTeX tables ----
    print("\nGenerating LaTeX tables ...")
    tex_parts = [
        "% Auto-generated K=25 tables\n",
        r"% Method order: FW-E, Greedy-E, A-optimal, D-optimal, Random, Coverage, Motion-div.",
        "\n",
    ]
    for i, (dir_key, label, caption) in enumerate(TABLE_DEFS, 1):
        if dir_key not in all_data:
            print(f"  [skip] Table {i}: {dir_key} not found")
            continue
        print(f"  Table {i}: {dir_key}")
        tex_parts.append(f"% ---- Table {i}: {dir_key} ----\n")
        tex_parts.append(make_latex_table(dir_key, all_data[dir_key], label, caption))

    tex_path = args.output_dir / "tables_k25.tex"
    tex_path.write_text("\n".join(tex_parts))
    print(f"\n  LaTeX saved: {tex_path}")

    # ---- Plots ----
    print("\nGenerating plots ...")
    plot_param_error_by_noise(all_data, args.output_dir)
    plot_lambda_min_by_noise(all_data, args.output_dir)
    plot_lambda_vs_param_error(all_data, args.output_dir)
    plot_method_ranking_heatmap(all_data, args.output_dir)
    plot_distortion_effect(all_data, args.output_dir)

    print(f"\nAll outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
