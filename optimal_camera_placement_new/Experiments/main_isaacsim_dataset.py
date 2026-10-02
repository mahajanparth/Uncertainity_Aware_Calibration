import argparse
import json
import pathlib
import sys
from datetime import datetime

if __package__ is None or __package__ == "":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal_analysis
from OASIS import calibration_visualize as cal_viz
from OASIS import methods


def create_run_output_dir(base_dir: pathlib.Path, run_name: str) -> pathlib.Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = base_dir / f"{run_name}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=False)
    return out_dir


def write_json(path: pathlib.Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def build_report_interpretation() -> dict:
    return {
        "primary_objective": {
            "name": "min_eig",
            "description": "Minimum eigenvalue of the calibration information matrix H_cal.",
            "desired_direction": "higher",
            "why": "Higher values mean the weakest constrained calibration direction is better observed.",
        },
        "selection_goal": "Prefer pose sets with high min_eig, low trace_cov, and low cond.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pose selection on an Isaac Sim exported CalibrationProblem.")
    parser.add_argument("--problem", required=True, help="Path to calibration_problem.npz")
    parser.add_argument("--select-k", type=int, default=20, help="Number of poses to select")
    parser.add_argument(
        "--selection-method",
        choices=("greedy", "frank-wolfe"),
        default="greedy",
        help="Selection algorithm used on top of the current FIM backend.",
    )
    parser.add_argument(
        "--fim-backend",
        choices=("numeric", "gtsam"),
        default=fim.get_information_backend(),
        help="Backend used to build candidate Fisher information blocks.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(pathlib.Path(__file__).resolve().parents[1] / "results"),
        help="Directory for plots and summaries",
    )
    args = parser.parse_args()

    fim.set_information_backend(args.fim_backend)
    problem = fim.load_calibration_problem_npz(args.problem)
    prior = fim.build_prior_blocks(problem)
    selection_method = methods.SelectionMethod(args.selection_method)
    selected_poses, selected_indices, best_score, selection, relaxed_selection = methods.select_poses(
        problem,
        select_k=args.select_k,
        prior=prior,
        method=selection_method,
    )
    report = cal_analysis.compare_selected_vs_random(problem, selected_indices, prior=prior, num_random_trials=25, seed=7)
    candidate_scores = cal_analysis.candidate_min_eig_scores(problem, prior=prior)
    candidate_spectra = cal_analysis.candidate_eigenvalue_spectra(problem, prior=prior)
    before_after = cal_analysis.before_after_calibration_summary(problem, selected_indices, prior=prior)

    out_dir = create_run_output_dir(pathlib.Path(args.output_dir), "isaacsim_problem")
    pose_plot_path = out_dir / "isaacsim_pose_selection.png"
    report_plot_path = out_dir / "isaacsim_selection_report.png"
    eig_plot_path = out_dir / "isaacsim_candidate_eigenvalues.png"
    eig_plot_3d_path = out_dir / "isaacsim_candidate_eigenvalues_3d.png"
    eig_spectra_plot_path = out_dir / "isaacsim_candidate_eigenvalue_spectra.png"
    uncertainty_plot_path = out_dir / "isaacsim_parameter_uncertainty_before_after.png"
    min_eig_compare_plot_path = out_dir / "isaacsim_min_eig_before_after.png"
    eig_compare_plot_path = out_dir / "isaacsim_eigenvalues_before_after.png"
    summary_path = out_dir / "summary.json"

    cal_viz.plot_pose_selection(problem, selected_indices, report["random_best_indices"], save_path=str(pose_plot_path))
    cal_viz.plot_selection_report(report, save_path=str(report_plot_path))
    cal_viz.plot_candidate_eigenvalues(candidate_scores, selected_indices, report["random_best_indices"], save_path=str(eig_plot_path))
    cal_viz.plot_candidate_eigenvalues_3d(
        problem,
        candidate_scores,
        selected_indices,
        report["random_best_indices"],
        save_path=str(eig_plot_3d_path),
    )
    cal_viz.plot_candidate_eigenvalue_spectra(
        candidate_spectra,
        selected_indices,
        report["random_best_indices"],
        save_path=str(eig_spectra_plot_path),
    )
    cal_viz.plot_parameter_uncertainty_before_after(
        before_after["parameter_labels"],
        before_after["before"]["std_dev"],
        before_after["after"]["std_dev"],
        save_path=str(uncertainty_plot_path),
    )
    cal_viz.plot_min_eigenvalue_before_after(
        before_after["before"]["min_eig"],
        before_after["after"]["min_eig"],
        save_path=str(min_eig_compare_plot_path),
    )
    cal_viz.plot_eigenvalues_before_after(
        before_after["before"]["eigvals"],
        before_after["after"]["eigvals"],
        save_path=str(eig_compare_plot_path),
    )

    summary = {
        "problem_path": str(pathlib.Path(args.problem).expanduser().resolve()),
        "fim_backend": fim.get_information_backend(),
        "selection_method": args.selection_method,
        "num_candidate_poses": problem.num_candidates,
        "select_k": args.select_k,
        "selected_indices": selected_indices,
        "best_score": float(best_score),
        "selection_vector": selection.astype(int).tolist(),
        "relaxed_selection_vector": None if relaxed_selection is None else np.asarray(relaxed_selection, dtype=float).tolist(),
        "selected_report": report["selected"],
        "random_average_report": report["random_average"],
        "random_best_report": report["random_best"],
        "random_best_indices": report["random_best_indices"],
        "candidate_scores": candidate_scores.tolist(),
        "candidate_eigenvalue_spectra": candidate_spectra.tolist(),
        "before_after_calibration": before_after,
        "report_interpretation": build_report_interpretation(),
        "output_dir": str(out_dir),
    }
    write_json(summary_path, summary)

    print("Optimal pose selection on Isaac Sim calibration dataset")
    print(f"Problem file: {args.problem}")
    print(f"FIM backend: {fim.get_information_backend()}")
    print(f"Selection method: {args.selection_method}")
    print(f"Number of candidate poses: {problem.num_candidates}")
    print(f"Number of selected poses: {args.select_k}")
    print(f"Selected pose indices: {selected_indices}")
    print(f"Best score: {best_score:.6f}")
    for idx, pose in zip(selected_indices, selected_poses):
        print(f"Pose {idx}: t_wc = {pose['translation_wc']}, visible_points = {pose['visible_points']}")
    print(f"Summary saved to: {summary_path}")
