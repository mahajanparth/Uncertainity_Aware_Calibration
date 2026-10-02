from __future__ import annotations

import time
from enum import Enum
from typing import List, Sequence, Tuple

import numpy as np

from . import FIM as infmat


class Metric(Enum):
    MIN_EIG = 1


class SelectionMethod(Enum):
    GREEDY = "greedy"
    FRANK_WOLFE = "frank-wolfe"
    FRANK_WOLFE_A = "frank-wolfe-a"
    FRANK_WOLFE_D = "frank-wolfe-d"
    COVERAGE = "coverage"
    MOTION_DIVERSITY = "motion-diversity"
    A_OPTIMAL = "a-optimal"
    D_OPTIMAL = "d-optimal"


def greedy_selection(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
    metric: Metric = Metric.MIN_EIG,
):
    if metric is not Metric.MIN_EIG:
        raise ValueError("This calibration pipeline keeps only the minimum-eigenvalue objective.")
    if prior is None:
        prior = infmat.build_prior_blocks(problem)

    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    available = np.ones(problem.num_candidates, dtype=bool)
    selection = np.zeros(problem.num_candidates, dtype=float)
    selected_indices: List[int] = []
    best_score = float("-inf")

    for _ in range(select_k):
        candidate_best_score = float("-inf")
        candidate_best_idx = -1
        for idx in range(problem.num_candidates):
            if not available[idx]:
                continue
            trial_selection = selection.copy()
            trial_selection[idx] = 1.0
            score = infmat.compute_min_eig_score(problem, trial_selection, info_blocks, prior=prior)
            if score > candidate_best_score:
                candidate_best_score = score
                candidate_best_idx = idx

        if candidate_best_idx < 0:
            break

        selection[candidate_best_idx] = 1.0
        available[candidate_best_idx] = False
        selected_indices.append(candidate_best_idx)
        best_score = candidate_best_score

    selected_poses = [
        {
            "rotation_wc": problem.candidate_rotations[idx],
            "translation_wc": problem.candidate_translations[idx],
            "visible_points": int(info_blocks.visible_counts[idx]),
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, best_score, selection


def frank_wolfe_selection(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
):
    from . import optimizations as continuous_optim

    if prior is None:
        prior = infmat.build_prior_blocks(problem)

    prior_info = infmat.build_prior_information(problem)
    print(f"[FW] Building {problem.num_candidates} candidate information matrices...", flush=True)
    inf_mats, _debug_visible_counts = infmat.construct_candidate_inf_mats(problem)
    print(f"[FW] Done. Starting Frank-Wolfe optimization (select_k={select_k})...", flush=True)

    n = problem.num_candidates
    selection_init = np.full(n, float(select_k) / max(n, 1), dtype=float)
    A = np.ones((1, n), dtype=float)
    b = np.array([float(select_k)], dtype=float)

    H0 = continuous_optim.csr_matrix(prior_info)
    H0.eliminate_zeros()  # prior_info is dense numpy — drop explicit zeros so H0 is truly sparse

    relaxed_selection, _relaxed_obj, _iterations, _fw_gap = continuous_optim.frank_wolfe_optimization(
        inf_mats=inf_mats,
        H0=H0,
        selection_init=selection_init,
        num_poses=problem.num_candidates,
        A=A,
        b=b,
    )
    selection = continuous_optim.roundsolution(relaxed_selection, select_k)
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    best_score = infmat.compute_min_eig_score(problem, selection, info_blocks, prior=prior)
    selected_indices = np.flatnonzero(selection > 0.5).astype(int).tolist()
    selected_poses = [
        {
            "rotation_wc": problem.candidate_rotations[idx],
            "translation_wc": problem.candidate_translations[idx],
            "visible_points": int(info_blocks.visible_counts[idx]),
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, best_score, selection, relaxed_selection


def frank_wolfe_a_optimal_selection(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
):
    """Frank-Wolfe A-optimal selection.

    Continuous relaxation of min trace(H_cal^{-1}(u)) via Frank-Wolfe,
    rounded to the top-K binary solution. Returns the same 5-tuple as
    frank_wolfe_selection so it's a drop-in replacement.
    """
    from . import optimizations as continuous_optim

    if prior is None:
        prior = infmat.build_prior_blocks(problem)

    prior_info = infmat.build_prior_information(problem)
    print(f"[FW-A] Building {problem.num_candidates} candidate information matrices...", flush=True)
    inf_mats, _debug_visible_counts = infmat.construct_candidate_inf_mats(problem)
    print(f"[FW-A] Done. Starting Frank-Wolfe A-optimal (select_k={select_k})...", flush=True)

    n = problem.num_candidates
    selection_init = np.full(n, float(select_k) / max(n, 1), dtype=float)
    A = np.ones((1, n), dtype=float)
    b = np.array([float(select_k)], dtype=float)

    H0 = continuous_optim.csr_matrix(prior_info)
    H0.eliminate_zeros()

    relaxed_selection, _relaxed_obj, _iterations, _fw_gap = continuous_optim.frank_wolfe_optimization_a_optimal(
        inf_mats=inf_mats,
        H0=H0,
        selection_init=selection_init,
        num_poses=problem.num_candidates,
        A=A,
        b=b,
    )
    selection = continuous_optim.roundsolution(relaxed_selection, select_k)

    # Compute A-optimal score (negative trace of covariance) on the rounded solution
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    h_cal = infmat.compute_calibration_schur_compact(problem, selection, info_blocks, prior=prior)
    reg = 1e-12 * np.eye(h_cal.shape[0])
    best_score = -float(np.trace(np.linalg.pinv(h_cal + reg)))

    selected_indices = np.flatnonzero(selection > 0.5).astype(int).tolist()
    selected_poses = [
        {
            "rotation_wc": problem.candidate_rotations[idx],
            "translation_wc": problem.candidate_translations[idx],
            "visible_points": int(info_blocks.visible_counts[idx]),
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, best_score, selection, relaxed_selection


def frank_wolfe_d_optimal_selection(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
):
    """Frank-Wolfe D-optimal selection.

    Continuous relaxation of max log det(H_cal(u)) via Frank-Wolfe,
    rounded to the top-K binary solution. Returns the same 5-tuple as
    frank_wolfe_selection so it's a drop-in replacement.
    """
    from . import optimizations as continuous_optim

    if prior is None:
        prior = infmat.build_prior_blocks(problem)

    prior_info = infmat.build_prior_information(problem)
    print(f"[FW-D] Building {problem.num_candidates} candidate information matrices...", flush=True)
    inf_mats, _debug_visible_counts = infmat.construct_candidate_inf_mats(problem)
    print(f"[FW-D] Done. Starting Frank-Wolfe D-optimal (select_k={select_k})...", flush=True)

    n = problem.num_candidates
    selection_init = np.full(n, float(select_k) / max(n, 1), dtype=float)
    A = np.ones((1, n), dtype=float)
    b = np.array([float(select_k)], dtype=float)

    H0 = continuous_optim.csr_matrix(prior_info)
    H0.eliminate_zeros()

    relaxed_selection, _relaxed_obj, _iterations, _fw_gap = continuous_optim.frank_wolfe_optimization_d_optimal(
        inf_mats=inf_mats,
        H0=H0,
        selection_init=selection_init,
        num_poses=problem.num_candidates,
        A=A,
        b=b,
    )
    selection = continuous_optim.roundsolution(relaxed_selection, select_k)

    # Compute D-optimal score (log det) on the rounded solution
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    h_cal = infmat.compute_calibration_schur_compact(problem, selection, info_blocks, prior=prior)
    reg = 1e-12 * np.eye(h_cal.shape[0])
    sign, logabsdet = np.linalg.slogdet(h_cal + reg)
    best_score = float(logabsdet) if sign > 0 else float("-inf")

    selected_indices = np.flatnonzero(selection > 0.5).astype(int).tolist()
    selected_poses = [
        {
            "rotation_wc": problem.candidate_rotations[idx],
            "translation_wc": problem.candidate_translations[idx],
            "visible_points": int(info_blocks.visible_counts[idx]),
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, best_score, selection, relaxed_selection


def _greedy_generic(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: infmat.CalibrationPrior,
    objective_fn,
):
    """Generic greedy selection: maximises objective_fn(h_cal) at each step."""
    info_blocks = infmat.construct_candidate_inf_blocks(problem)
    available = np.ones(problem.num_candidates, dtype=bool)
    selection = np.zeros(problem.num_candidates, dtype=float)
    selected_indices: List[int] = []
    best_score = float("-inf")

    for _ in range(select_k):
        candidate_best_score = float("-inf")
        candidate_best_idx = -1
        for idx in range(problem.num_candidates):
            if not available[idx]:
                continue
            trial_selection = selection.copy()
            trial_selection[idx] = 1.0
            h_cal = infmat.compute_calibration_schur_compact(problem, trial_selection, info_blocks, prior=prior)
            score = objective_fn(h_cal)
            if score > candidate_best_score:
                candidate_best_score = score
                candidate_best_idx = idx

        if candidate_best_idx < 0:
            break

        selection[candidate_best_idx] = 1.0
        available[candidate_best_idx] = False
        selected_indices.append(candidate_best_idx)
        best_score = candidate_best_score

    selected_poses = [
        {
            "rotation_wc": problem.candidate_rotations[idx],
            "translation_wc": problem.candidate_translations[idx],
            "visible_points": int(info_blocks.visible_counts[idx]),
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, best_score, selection


def greedy_selection_a_optimal(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
):
    """Greedy A-optimal: minimises trace of calibration covariance (= trace(H_cal^{-1}))."""
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    reg = 1e-12 * np.eye(problem.intrinsics_dim, dtype=float)
    objective_fn = lambda h: -float(np.trace(np.linalg.pinv(h + reg)))
    return _greedy_generic(problem, select_k, prior, objective_fn)


def greedy_selection_d_optimal(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
):
    """Greedy D-optimal: maximises log-determinant of H_cal."""
    if prior is None:
        prior = infmat.build_prior_blocks(problem)
    reg = 1e-12 * np.eye(problem.intrinsics_dim, dtype=float)
    objective_fn = lambda h: float(np.linalg.slogdet(h + reg)[1])
    return _greedy_generic(problem, select_k, prior, objective_fn)


def coverage_selection(
    problem: infmat.CalibrationProblem,
    select_k: int,
):
    """Coverage heuristic: greedy farthest-point sampling on camera translations."""
    translations = problem.candidate_translations  # (N, 3)
    n = problem.num_candidates
    available = np.ones(n, dtype=bool)
    selected_indices: List[int] = []

    # Seed: candidate closest to the mean translation
    mean_t = translations.mean(axis=0)
    seed = int(np.argmin(np.linalg.norm(translations - mean_t, axis=1)))
    selected_indices.append(seed)
    available[seed] = False

    # Greedy farthest-point: pick the available candidate farthest from all selected
    min_dists = np.linalg.norm(translations - translations[seed], axis=1)
    for _ in range(select_k - 1):
        min_dists[~available] = -1.0
        best_idx = int(np.argmax(min_dists))
        if best_idx < 0 or not available[best_idx]:
            break
        selected_indices.append(best_idx)
        available[best_idx] = False
        dists_to_new = np.linalg.norm(translations - translations[best_idx], axis=1)
        min_dists = np.minimum(min_dists, dists_to_new)

    selection = np.zeros(n, dtype=float)
    selection[selected_indices] = 1.0
    selected_poses = [
        {
            "rotation_wc": problem.candidate_rotations[idx],
            "translation_wc": problem.candidate_translations[idx],
            "visible_points": 0,
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, float("nan"), selection


def motion_diversity_selection(
    problem: infmat.CalibrationProblem,
    select_k: int,
):
    """Motion-diversity heuristic: greedy farthest-point sampling on SO(3) rotations."""
    rotations = problem.candidate_rotations  # (N, 3, 3)
    n = problem.num_candidates
    available = np.ones(n, dtype=bool)
    selected_indices: List[int] = []

    def _geodesic(R1, R2):
        cos_angle = (np.trace(R1.T @ R2) - 1.0) / 2.0
        return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))

    # Seed: rotation closest to identity
    seed = int(np.argmin([_geodesic(R, np.eye(3)) for R in rotations]))
    selected_indices.append(seed)
    available[seed] = False

    min_dists = np.array([_geodesic(R, rotations[seed]) for R in rotations])
    for _ in range(select_k - 1):
        min_dists[~available] = -1.0
        best_idx = int(np.argmax(min_dists))
        if best_idx < 0 or not available[best_idx]:
            break
        selected_indices.append(best_idx)
        available[best_idx] = False
        dists_to_new = np.array([_geodesic(R, rotations[best_idx]) for R in rotations])
        min_dists = np.minimum(min_dists, dists_to_new)

    selection = np.zeros(n, dtype=float)
    selection[selected_indices] = 1.0
    selected_poses = [
        {
            "rotation_wc": problem.candidate_rotations[idx],
            "translation_wc": problem.candidate_translations[idx],
            "visible_points": 0,
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, float("nan"), selection


def select_poses(
    problem: infmat.CalibrationProblem,
    select_k: int,
    prior: np.ndarray | infmat.CalibrationPrior | None = None,
    metric: Metric = Metric.MIN_EIG,
    method: SelectionMethod = SelectionMethod.GREEDY,
):
    if method is SelectionMethod.GREEDY:
        selected_poses, selected_indices, best_score, selection = greedy_selection(
            problem,
            select_k,
            prior=prior,
            metric=metric,
        )
        return selected_poses, selected_indices, best_score, selection, None
    if method is SelectionMethod.FRANK_WOLFE:
        if metric is not Metric.MIN_EIG:
            raise ValueError("Frank-Wolfe wrapper currently supports only the minimum-eigenvalue objective.")
        return frank_wolfe_selection(problem, select_k, prior=prior)
    if method is SelectionMethod.FRANK_WOLFE_A:
        return frank_wolfe_a_optimal_selection(problem, select_k, prior=prior)
    if method is SelectionMethod.FRANK_WOLFE_D:
        return frank_wolfe_d_optimal_selection(problem, select_k, prior=prior)
    if method is SelectionMethod.COVERAGE:
        selected_poses, selected_indices, best_score, selection = coverage_selection(problem, select_k)
        return selected_poses, selected_indices, best_score, selection, None
    if method is SelectionMethod.MOTION_DIVERSITY:
        selected_poses, selected_indices, best_score, selection = motion_diversity_selection(problem, select_k)
        return selected_poses, selected_indices, best_score, selection, None
    if method is SelectionMethod.A_OPTIMAL:
        selected_poses, selected_indices, best_score, selection = greedy_selection_a_optimal(problem, select_k, prior=prior)
        return selected_poses, selected_indices, best_score, selection, None
    if method is SelectionMethod.D_OPTIMAL:
        selected_poses, selected_indices, best_score, selection = greedy_selection_d_optimal(problem, select_k, prior=prior)
        return selected_poses, selected_indices, best_score, selection, None
    raise ValueError(f"Unsupported selection method: {method}")


def greedy_selection_exp(
    problems: Sequence[infmat.CalibrationProblem],
    select_k: int,
    priors: Sequence[np.ndarray | infmat.CalibrationPrior] | None = None,
    metric: Metric = Metric.MIN_EIG,
):
    if metric is not Metric.MIN_EIG:
        raise ValueError("This calibration pipeline keeps only the minimum-eigenvalue objective.")
    if not problems:
        raise ValueError("At least one calibration problem is required.")

    if priors is None:
        priors = [infmat.build_prior_blocks(problem) for problem in problems]

    info_blocks_per_problem = []
    for problem in problems:
        info_blocks_per_problem.append(infmat.construct_candidate_inf_blocks(problem))

    num_candidates = problems[0].num_candidates
    available = np.ones(num_candidates, dtype=bool)
    selection = np.zeros(num_candidates, dtype=float)
    selected_indices: List[int] = []
    best_score = float("-inf")

    for _ in range(select_k):
        candidate_best_score = float("-inf")
        candidate_best_idx = -1
        for idx in range(num_candidates):
            if not available[idx]:
                continue
            trial_selection = selection.copy()
            trial_selection[idx] = 1.0
            total_score = 0.0
            for problem, prior, info_blocks in zip(problems, priors, info_blocks_per_problem):
                total_score += infmat.compute_min_eig_score(problem, trial_selection, info_blocks, prior=prior)
            if total_score > candidate_best_score:
                candidate_best_score = total_score
                candidate_best_idx = idx

        if candidate_best_idx < 0:
            break

        selection[candidate_best_idx] = 1.0
        available[candidate_best_idx] = False
        selected_indices.append(candidate_best_idx)
        best_score = candidate_best_score

    selected_poses = [
        {
            "rotation_wc": problems[0].candidate_rotations[idx],
            "translation_wc": problems[0].candidate_translations[idx],
            "visible_points": int(info_blocks_per_problem[0].visible_counts[idx]),
        }
        for idx in selected_indices
    ]
    return selected_poses, selected_indices, best_score, selection


def run_single_experiment_exp(
    problems: Sequence[infmat.CalibrationProblem],
    select_k: int,
    priors: Sequence[np.ndarray] | None = None,
):
    start = time.time()
    selected_poses, selected_indices, best_score, selection = greedy_selection_exp(
        problems,
        select_k,
        priors=priors,
        metric=Metric.MIN_EIG,
    )
    elapsed = time.time() - start
    return best_score, selected_poses, selected_indices, elapsed, selection
