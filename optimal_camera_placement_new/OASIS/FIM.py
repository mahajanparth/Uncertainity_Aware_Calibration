from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from numpy import linalg as la
import scipy.sparse as sp
from joblib import Parallel, delayed
from tqdm import tqdm


@dataclass
class CalibrationProblem:
    """Synthetic camera calibration problem used for pose selection.

    Attributes:
        target_points: Known calibration target points in the world frame, shape (M, 3).
        candidate_rotations: Candidate camera rotations R_wc, shape (N, 3, 3).
        candidate_translations: Candidate camera translations t_wc, shape (N, 3).
        measurements: Noisy image observations for each candidate pose, shape (N, M, 2).
            Invalid observations are stored as NaN.
        intrinsics_gt: Ground-truth intrinsic parameter vector [fx, fy, cx, cy].
        intrinsics_init: Initial intrinsic estimate used as the linearization point.
        image_size: Image size as (width, height).
        pixel_noise_sigma: Standard deviation of image noise in pixels.
    """

    target_points: np.ndarray
    candidate_rotations: np.ndarray
    candidate_translations: np.ndarray
    measurements: np.ndarray
    intrinsics_gt: np.ndarray
    intrinsics_init: np.ndarray
    image_size: Tuple[int, int]
    pixel_noise_sigma: float = 1.0
    candidate_board_rotations: Optional[np.ndarray] = None
    candidate_board_translations: Optional[np.ndarray] = None
    camera_is_fixed: bool = False

    @property
    def num_candidates(self) -> int:
        return int(self.candidate_rotations.shape[0])

    @property
    def intrinsics_dim(self) -> int:
        return int(self.intrinsics_init.shape[0])

    @property
    def pose_dim(self) -> int:
        return 6

    @property
    def total_dim(self) -> int:
        return self.intrinsics_dim + self.pose_dim * self.num_candidates


@dataclass
class CandidateInfoBlocks:
    h_tt: np.ndarray
    h_tp: np.ndarray
    h_pp: np.ndarray
    visible_counts: np.ndarray


@dataclass
class CalibrationPrior:
    h_tt: np.ndarray
    h_pp: np.ndarray


MIN_PIXEL_NOISE_SIGMA = 1e-9
_DEFAULT_INFO_BACKEND = "numeric"


def get_information_backend() -> str:
    return _DEFAULT_INFO_BACKEND


def set_information_backend(name: str) -> None:
    normalized = str(name).strip().lower()
    if normalized not in {"numeric", "gtsam"}:
        raise ValueError(f"Unsupported information backend '{name}'. Expected one of: numeric, gtsam.")
    global _DEFAULT_INFO_BACKEND
    _DEFAULT_INFO_BACKEND = normalized


def has_gtsam_backend() -> bool:
    try:
        import gtsam  # noqa: F401
    except ImportError:
        return False
    return True


def effective_pixel_noise_sigma(pixel_noise_sigma: float) -> float:
    return max(float(pixel_noise_sigma), MIN_PIXEL_NOISE_SIGMA)


def default_intrinsics_vector(image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    return np.array([820.0, 815.0, width / 2.0, height / 2.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


def default_intrinsics_init_offset(intrinsics_dim: int) -> np.ndarray:
    offset = np.zeros(intrinsics_dim, dtype=float)
    base = np.array([25.0, -20.0, 8.0, -6.0], dtype=float)
    offset[: min(intrinsics_dim, base.size)] = base[: min(intrinsics_dim, base.size)]
    return offset


def default_intrinsics_prior_sigma(intrinsics_dim: int) -> np.ndarray:
    sigma = np.full(intrinsics_dim, 0.25, dtype=float)
    if intrinsics_dim >= 1:
        sigma[0] = 30.0
    if intrinsics_dim >= 2:
        sigma[1] = 30.0
    if intrinsics_dim >= 3:
        sigma[2] = 20.0
    if intrinsics_dim >= 4:
        sigma[3] = 20.0
    return sigma


def split_intrinsics(intrinsics: np.ndarray) -> Tuple[float, float, float, float, np.ndarray]:
    intrinsics = np.asarray(intrinsics, dtype=float).reshape(-1)
    if intrinsics.size < 4:
        raise ValueError("Intrinsics vector must contain at least [fx, fy, cx, cy].")
    fx, fy, cx, cy = [float(v) for v in intrinsics[:4]]
    dist = np.zeros(5, dtype=float)
    if intrinsics.size > 4:
        extra = intrinsics[4:]
        dist[: min(5, extra.size)] = extra[: min(5, extra.size)]
    return fx, fy, cx, cy, dist


def save_calibration_problem_npz(path: str, problem: "CalibrationProblem") -> None:
    payload = {
        "target_points": np.asarray(problem.target_points, dtype=float),
        "candidate_rotations": np.asarray(problem.candidate_rotations, dtype=float),
        "candidate_translations": np.asarray(problem.candidate_translations, dtype=float),
        "measurements": np.asarray(problem.measurements, dtype=float),
        "intrinsics_gt": np.asarray(problem.intrinsics_gt, dtype=float),
        "intrinsics_init": np.asarray(problem.intrinsics_init, dtype=float),
        "image_size": np.asarray(problem.image_size, dtype=int),
        "pixel_noise_sigma": np.asarray(problem.pixel_noise_sigma, dtype=float),
        "camera_is_fixed": np.asarray(problem.camera_is_fixed, dtype=bool),
    }
    if problem.candidate_board_rotations is not None:
        payload["candidate_board_rotations"] = np.asarray(problem.candidate_board_rotations, dtype=float)
    if problem.candidate_board_translations is not None:
        payload["candidate_board_translations"] = np.asarray(problem.candidate_board_translations, dtype=float)
    np.savez(path, **payload)


def load_calibration_problem_npz(path: str) -> "CalibrationProblem":
    with np.load(path, allow_pickle=False) as data:
        candidate_board_rotations = None
        candidate_board_translations = None
        if "candidate_board_rotations" in data.files:
            candidate_board_rotations = np.asarray(data["candidate_board_rotations"], dtype=float)
        if "candidate_board_translations" in data.files:
            candidate_board_translations = np.asarray(data["candidate_board_translations"], dtype=float)
        camera_is_fixed = False
        if "camera_is_fixed" in data.files:
            camera_is_fixed = bool(np.asarray(data["camera_is_fixed"]).item())
        pixel_noise_sigma = float(np.asarray(data["pixel_noise_sigma"]).item())
        image_size = tuple(int(v) for v in np.asarray(data["image_size"]).reshape(-1).tolist())
        return CalibrationProblem(
            target_points=np.asarray(data["target_points"], dtype=float),
            candidate_rotations=np.asarray(data["candidate_rotations"], dtype=float),
            candidate_translations=np.asarray(data["candidate_translations"], dtype=float),
            measurements=np.asarray(data["measurements"], dtype=float),
            intrinsics_gt=np.asarray(data["intrinsics_gt"], dtype=float),
            intrinsics_init=np.asarray(data["intrinsics_init"], dtype=float),
            image_size=image_size,
            pixel_noise_sigma=pixel_noise_sigma,
            candidate_board_rotations=candidate_board_rotations,
            candidate_board_translations=candidate_board_translations,
            camera_is_fixed=camera_is_fixed,
        )


def subsample_calibration_problem(
    problem: "CalibrationProblem",
    indices: list[int],
) -> "CalibrationProblem":
    """Return a new CalibrationProblem restricted to the given candidate indices."""
    idx = np.asarray(indices, dtype=int)
    cbr = problem.candidate_board_rotations[idx] if problem.candidate_board_rotations is not None else None
    cbt = problem.candidate_board_translations[idx] if problem.candidate_board_translations is not None else None
    return CalibrationProblem(
        target_points=problem.target_points,
        candidate_rotations=problem.candidate_rotations[idx],
        candidate_translations=problem.candidate_translations[idx],
        measurements=problem.measurements[idx],
        intrinsics_gt=problem.intrinsics_gt.copy(),
        intrinsics_init=problem.intrinsics_init.copy(),
        image_size=problem.image_size,
        pixel_noise_sigma=problem.pixel_noise_sigma,
        candidate_board_rotations=cbr,
        candidate_board_translations=cbt,
        camera_is_fixed=problem.camera_is_fixed,
    )


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def _exp_so3(w: np.ndarray) -> np.ndarray:
    theta = la.norm(w)
    if theta < 1e-12:
        return np.eye(3) + _skew(w)
    W = _skew(w)
    a = np.sin(theta) / theta
    b = (1.0 - np.cos(theta)) / (theta * theta)
    return np.eye(3) + a * W + b * (W @ W)


def project_points(
    points_w: np.ndarray,
    rotation_wc: np.ndarray,
    translation_wc: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    """Project known target points into the image using a pinhole camera model.

    Supported intrinsic layouts:
    - [fx, fy, cx, cy]
    - [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    Extra trailing entries beyond the first 9 are ignored by the projector.
    """
    fx, fy, cx, cy, dist = split_intrinsics(intrinsics)
    rotation_cw = rotation_wc.T
    points_c = (rotation_cw @ (points_w - translation_wc).T).T
    z = points_c[:, 2]
    proj = np.full((points_w.shape[0], 2), np.nan, dtype=float)
    valid = z > 1e-9
    if not np.any(valid):
        return proj

    x_norm = points_c[valid, 0] / z[valid]
    y_norm = points_c[valid, 1] / z[valid]
    k1, k2, p1, p2, k3 = dist
    r2 = x_norm * x_norm + y_norm * y_norm
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    x_dist = x_norm * radial + 2.0 * p1 * x_norm * y_norm + p2 * (r2 + 2.0 * x_norm * x_norm)
    y_dist = y_norm * radial + p1 * (r2 + 2.0 * y_norm * y_norm) + 2.0 * p2 * x_norm * y_norm
    proj[valid, 0] = fx * x_dist + cx
    proj[valid, 1] = fy * y_dist + cy
    return proj


def visible_measurement_mask(measurements: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    width, height = image_size
    finite = np.isfinite(measurements).all(axis=1)
    in_bounds = (
        (measurements[:, 0] >= 0.0)
        & (measurements[:, 0] < width)
        & (measurements[:, 1] >= 0.0)
        & (measurements[:, 1] < height)
    )
    return finite & in_bounds


def perturb_pose(rotation_wc: np.ndarray, translation_wc: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rot_delta = _exp_so3(delta[:3])
    new_rotation = rotation_wc @ rot_delta
    new_translation = translation_wc + delta[3:]
    return new_rotation, new_translation


def numerical_jacobian_intrinsics(
    points_w: np.ndarray,
    rotation_wc: np.ndarray,
    translation_wc: np.ndarray,
    intrinsics: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    base = project_points(points_w, rotation_wc, translation_wc, intrinsics)[valid_mask].reshape(-1)
    jac = np.zeros((base.size, intrinsics.size), dtype=float)
    for idx in range(intrinsics.size):
        step = np.zeros_like(intrinsics)
        step[idx] = eps
        plus = project_points(points_w, rotation_wc, translation_wc, intrinsics + step)[valid_mask].reshape(-1)
        minus = project_points(points_w, rotation_wc, translation_wc, intrinsics - step)[valid_mask].reshape(-1)
        jac[:, idx] = (plus - minus) / (2.0 * eps)
    return jac


def numerical_jacobian_pose(
    points_w: np.ndarray,
    rotation_wc: np.ndarray,
    translation_wc: np.ndarray,
    intrinsics: np.ndarray,
    valid_mask: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    base = project_points(points_w, rotation_wc, translation_wc, intrinsics)[valid_mask].reshape(-1)
    jac = np.zeros((base.size, 6), dtype=float)
    for idx in range(6):
        step = np.zeros(6, dtype=float)
        step[idx] = eps
        rot_plus, trans_plus = perturb_pose(rotation_wc, translation_wc, step)
        rot_minus, trans_minus = perturb_pose(rotation_wc, translation_wc, -step)
        plus = project_points(points_w, rot_plus, trans_plus, intrinsics)[valid_mask].reshape(-1)
        minus = project_points(points_w, rot_minus, trans_minus, intrinsics)[valid_mask].reshape(-1)
        jac[:, idx] = (plus - minus) / (2.0 * eps)
    return jac


def _candidate_information_matrix_numeric(problem: CalibrationProblem, candidate_index: int) -> np.ndarray:
    """Build the Fisher information contribution for one candidate pose numerically.

    The state is ordered as [intrinsics, pose_0, ..., pose_{N-1}]. Only the selected
    candidate's own 6-DoF nuisance pose block is populated.
    """
    measurements = problem.measurements[candidate_index]
    valid_mask = visible_measurement_mask(measurements, problem.image_size)
    if np.count_nonzero(valid_mask) < 4:
        return np.zeros((problem.total_dim, problem.total_dim), dtype=float)

    rotation_wc = problem.candidate_rotations[candidate_index]
    translation_wc = problem.candidate_translations[candidate_index]
    jac_intr = numerical_jacobian_intrinsics(
        problem.target_points,
        rotation_wc,
        translation_wc,
        problem.intrinsics_init,
        valid_mask,
    )
    jac_pose = numerical_jacobian_pose(
        problem.target_points,
        rotation_wc,
        translation_wc,
        problem.intrinsics_init,
        valid_mask,
    )

    jac = np.zeros((jac_intr.shape[0], problem.total_dim), dtype=float)
    jac[:, : problem.intrinsics_dim] = jac_intr
    pose_start = problem.intrinsics_dim + candidate_index * problem.pose_dim
    pose_end = pose_start + problem.pose_dim
    jac[:, pose_start:pose_end] = jac_pose

    sigma = effective_pixel_noise_sigma(problem.pixel_noise_sigma)
    weight = np.eye(jac.shape[0], dtype=float) / (sigma ** 2)
    return jac.T @ weight @ jac


def _candidate_information_blocks_numeric(problem: CalibrationProblem, candidate_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build the compact intrinsics/pose blocks for one candidate pose numerically."""
    measurements = problem.measurements[candidate_index]
    valid_mask = visible_measurement_mask(measurements, problem.image_size)
    visible = int(np.count_nonzero(valid_mask))
    if visible < 4:
        zeros_tt = np.zeros((problem.intrinsics_dim, problem.intrinsics_dim), dtype=float)
        zeros_tp = np.zeros((problem.intrinsics_dim, problem.pose_dim), dtype=float)
        zeros_pp = np.zeros((problem.pose_dim, problem.pose_dim), dtype=float)
        return zeros_tt, zeros_tp, zeros_pp, visible

    rotation_wc = problem.candidate_rotations[candidate_index]
    translation_wc = problem.candidate_translations[candidate_index]
    jac_intr = numerical_jacobian_intrinsics(
        problem.target_points,
        rotation_wc,
        translation_wc,
        problem.intrinsics_init,
        valid_mask,
    )
    jac_pose = numerical_jacobian_pose(
        problem.target_points,
        rotation_wc,
        translation_wc,
        problem.intrinsics_init,
        valid_mask,
    )

    sigma = effective_pixel_noise_sigma(problem.pixel_noise_sigma)
    weight = 1.0 / (sigma ** 2)
    h_tt = weight * (jac_intr.T @ jac_intr)
    h_tp = weight * (jac_intr.T @ jac_pose)
    h_pp = weight * (jac_pose.T @ jac_pose)
    return h_tt, h_tp, h_pp, visible


def _candidate_information_blocks_gtsam(problem: CalibrationProblem, candidate_index: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Build compact FIM blocks using a factor-graph backend.

    GTSAM's `Cal3DS2` provides analytic projection Jacobians for a distorted pinhole model
    with parameters `[fx, fy, s, u0, v0, k1, k2, p1, p2]`. OASIS currently stores intrinsics
    as `[fx, fy, cx, cy, k1, k2, p1, p2, k3]`, so the backend maps between these two layouts.
    The extra `k3` term in the OASIS vector is not represented by `Cal3DS2`; its Jacobian
    column is therefore zero in this backend.
    """
    if not has_gtsam_backend():
        raise RuntimeError(
            "The 'gtsam' information backend was requested, but the active Python "
            "environment does not provide the 'gtsam' module. Install/activate a "
            "GTSAM-enabled environment or switch back to fim.set_information_backend('numeric')."
        )
    import gtsam

    measurements = problem.measurements[candidate_index]
    valid_mask = visible_measurement_mask(measurements, problem.image_size)
    visible = int(np.count_nonzero(valid_mask))
    zeros_tt = np.zeros((problem.intrinsics_dim, problem.intrinsics_dim), dtype=float)
    zeros_tp = np.zeros((problem.intrinsics_dim, problem.pose_dim), dtype=float)
    zeros_pp = np.zeros((problem.pose_dim, problem.pose_dim), dtype=float)
    if visible < 4:
        return zeros_tt, zeros_tp, zeros_pp, visible

    intr = np.asarray(problem.intrinsics_init, dtype=float).reshape(-1)
    if intr.size < 4:
        raise ValueError("GTSAM backend requires at least [fx, fy, cx, cy] intrinsics.")
    k1 = float(intr[4]) if intr.size > 4 else 0.0
    k2 = float(intr[5]) if intr.size > 5 else 0.0
    p1 = float(intr[6]) if intr.size > 6 else 0.0
    p2 = float(intr[7]) if intr.size > 7 else 0.0
    calibration = gtsam.Cal3DS2(float(intr[0]), float(intr[1]), 0.0, float(intr[2]), float(intr[3]), k1, k2, p1, p2)

    rotation_wc = np.asarray(problem.candidate_rotations[candidate_index], dtype=float)
    translation_wc = np.asarray(problem.candidate_translations[candidate_index], dtype=float).reshape(3)
    pose_wc = gtsam.Pose3(gtsam.Rot3(rotation_wc), translation_wc)

    # OASIS intrinsics order: [fx, fy, cx, cy, k1, k2, p1, p2, k3]
    # GTSAM Cal3DS2 order:    [fx, fy, s,  u0, v0, k1, k2, p1, p2]
    cal_col_map = {
        0: 0,  # fx
        1: 1,  # fy
        3: 2,  # u0 -> cx
        4: 3,  # v0 -> cy
        5: 4,  # k1
        6: 5,  # k2
        7: 6,  # p1
        8: 7,  # p2
    }

    jac_intr_rows: list[np.ndarray] = []
    jac_pose_rows: list[np.ndarray] = []
    if problem.intrinsics_dim > 8:
        jac_intr_numeric = numerical_jacobian_intrinsics(
            problem.target_points,
            rotation_wc,
            translation_wc,
            intr,
            valid_mask,
        )
    else:
        jac_intr_numeric = None
    for point_w in np.asarray(problem.target_points[valid_mask], dtype=float):
        dpose = np.zeros((2, 6), dtype=float, order="F")
        dpoint = np.zeros((2, 3), dtype=float, order="F")
        dcal = np.zeros((2, 9), dtype=float, order="F")
        try:
            gtsam.PinholeCameraCal3DS2(pose_wc, calibration).project(
                np.asarray(point_w, dtype=float),
                dpose,
                dpoint,
                dcal,
            )
        except RuntimeError:
            # Projection can fail for degenerate/behind-camera points. Fall back to an empty block.
            return zeros_tt, zeros_tp, zeros_pp, visible

        jac_intr = np.zeros((2, problem.intrinsics_dim), dtype=float)
        for gtsam_col, oasis_col in cal_col_map.items():
            if oasis_col < problem.intrinsics_dim:
                jac_intr[:, oasis_col] = dcal[:, gtsam_col]
        if jac_intr_numeric is not None:
            row_start = len(jac_intr_rows) * 2
            row_end = row_start + 2
            jac_intr[:, 8:] = jac_intr_numeric[row_start:row_end, 8:]
        jac_intr_rows.append(jac_intr)
        jac_pose_rows.append(np.asarray(dpose, dtype=float))

    if not jac_intr_rows:
        return zeros_tt, zeros_tp, zeros_pp, visible

    jac_intr = np.vstack(jac_intr_rows)
    jac_pose = np.vstack(jac_pose_rows)
    sigma = effective_pixel_noise_sigma(problem.pixel_noise_sigma)
    weight = 1.0 / (sigma ** 2)
    h_tt = weight * (jac_intr.T @ jac_intr)
    h_tp = weight * (jac_intr.T @ jac_pose)
    h_pp = weight * (jac_pose.T @ jac_pose)
    return h_tt, h_tp, h_pp, visible


def candidate_information_matrix(
    problem: CalibrationProblem,
    candidate_index: int,
    backend: Optional[str] = None,
) -> np.ndarray:
    selected_backend = get_information_backend() if backend is None else str(backend).strip().lower()
    if selected_backend == "numeric":
        return _candidate_information_matrix_numeric(problem, candidate_index)
    if selected_backend == "gtsam":
        h_tt, h_tp, h_pp, visible = _candidate_information_blocks_gtsam(problem, candidate_index)
        if visible < 4:
            return np.zeros((problem.total_dim, problem.total_dim), dtype=float)
        fim = np.zeros((problem.total_dim, problem.total_dim), dtype=float)
        intr_end = problem.intrinsics_dim
        pose_start = intr_end + candidate_index * problem.pose_dim
        pose_end = pose_start + problem.pose_dim
        fim[:intr_end, :intr_end] = h_tt
        fim[:intr_end, pose_start:pose_end] = h_tp
        fim[pose_start:pose_end, :intr_end] = h_tp.T
        fim[pose_start:pose_end, pose_start:pose_end] = h_pp
        return fim
    raise ValueError(f"Unsupported information backend '{selected_backend}'.")


def candidate_information_blocks(
    problem: CalibrationProblem,
    candidate_index: int,
    backend: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    selected_backend = get_information_backend() if backend is None else str(backend).strip().lower()
    if selected_backend == "numeric":
        return _candidate_information_blocks_numeric(problem, candidate_index)
    if selected_backend == "gtsam":
        return _candidate_information_blocks_gtsam(problem, candidate_index)
    raise ValueError(f"Unsupported information backend '{selected_backend}'.")


def construct_candidate_inf_blocks(problem: CalibrationProblem, backend: Optional[str] = None) -> CandidateInfoBlocks:
    h_tt = np.zeros((problem.num_candidates, problem.intrinsics_dim, problem.intrinsics_dim), dtype=float)
    h_tp = np.zeros((problem.num_candidates, problem.intrinsics_dim, problem.pose_dim), dtype=float)
    h_pp = np.zeros((problem.num_candidates, problem.pose_dim, problem.pose_dim), dtype=float)
    visible_counts = np.zeros(problem.num_candidates, dtype=int)
    for idx in range(problem.num_candidates):
        h_tt[idx], h_tp[idx], h_pp[idx], visible_counts[idx] = candidate_information_blocks(problem, idx, backend=backend)
    return CandidateInfoBlocks(h_tt=h_tt, h_tp=h_tp, h_pp=h_pp, visible_counts=visible_counts)


def construct_candidate_inf_mats(problem: CalibrationProblem, backend: Optional[str] = None) -> Tuple[List, List[int]]:
    """Build one sparse FIM per candidate directly from compact blocks.

    Each candidate's FIM is sparse: only the intrinsics block, the cross
    intrinsics-pose block, and the candidate's own pose block are non-zero.
    Building sparse matrices here avoids allocating and then converting a
    huge dense (num_candidates × total_dim × total_dim) array.
    """
    n = problem.total_dim
    intr_dim = problem.intrinsics_dim
    pose_dim = problem.pose_dim

    def _compute_one(idx):
        h_tt, h_tp, h_pp, visible = candidate_information_blocks(problem, idx, backend=backend)
        pose_start = intr_dim + idx * pose_dim
        pose_end = pose_start + pose_dim
        mat = sp.lil_matrix((n, n), dtype=float)
        if visible >= 4:
            mat[:intr_dim, :intr_dim] = h_tt
            mat[:intr_dim, pose_start:pose_end] = h_tp
            mat[pose_start:pose_end, :intr_dim] = h_tp.T
            mat[pose_start:pose_end, pose_start:pose_end] = h_pp
        return idx, mat.tocsr(), visible

    results = Parallel(n_jobs=-1)(
        delayed(_compute_one)(idx)
        for idx in tqdm(range(problem.num_candidates), desc="Building FIM", unit="candidate")
    )

    mats: List = [None] * problem.num_candidates
    debug_visible_counts: List[int] = [0] * problem.num_candidates
    for idx, mat, visible in results:
        mats[idx] = mat
        debug_visible_counts[idx] = visible
    return mats, debug_visible_counts


def build_prior_information(
    problem: CalibrationProblem,
    intrinsics_prior_sigma: Sequence[float] | None = None,
    pose_prior_sigma: Sequence[float] = (2.0, 2.0, 2.0, 0.5, 0.5, 0.5),
) -> np.ndarray:
    """Build a weak prior used to stabilize the Schur complement numerically."""
    prior = np.zeros((problem.total_dim, problem.total_dim), dtype=float)
    if intrinsics_prior_sigma is None:
        intrinsics_prior_sigma = default_intrinsics_prior_sigma(problem.intrinsics_dim)
    intr_diag = 1.0 / np.square(np.asarray(intrinsics_prior_sigma, dtype=float))
    prior[: problem.intrinsics_dim, : problem.intrinsics_dim] = np.diag(intr_diag)

    pose_diag = 1.0 / np.square(np.asarray(pose_prior_sigma, dtype=float))
    for idx in range(problem.num_candidates):
        start = problem.intrinsics_dim + idx * problem.pose_dim
        end = start + problem.pose_dim
        prior[start:end, start:end] = np.diag(pose_diag)
    return prior


def build_prior_blocks(
    problem: CalibrationProblem,
    intrinsics_prior_sigma: Sequence[float] | None = None,
    pose_prior_sigma: Sequence[float] = (2.0, 2.0, 2.0, 0.5, 0.5, 0.5),
) -> CalibrationPrior:
    if intrinsics_prior_sigma is None:
        intrinsics_prior_sigma = default_intrinsics_prior_sigma(problem.intrinsics_dim)
    intr_diag = 1.0 / np.square(np.asarray(intrinsics_prior_sigma, dtype=float))
    pose_diag = 1.0 / np.square(np.asarray(pose_prior_sigma, dtype=float))
    return CalibrationPrior(h_tt=np.diag(intr_diag), h_pp=np.diag(pose_diag))


def _coerce_prior_blocks(problem: CalibrationProblem, prior: Optional[np.ndarray | CalibrationPrior]) -> CalibrationPrior:
    if prior is None:
        return build_prior_blocks(problem)
    if isinstance(prior, CalibrationPrior):
        return prior

    h_tt = prior[: problem.intrinsics_dim, : problem.intrinsics_dim]
    pose_start = problem.intrinsics_dim
    pose_end = pose_start + problem.pose_dim
    if prior.shape[0] >= pose_end:
        h_pp = prior[pose_start:pose_end, pose_start:pose_end]
    else:
        h_pp = np.zeros((problem.pose_dim, problem.pose_dim), dtype=float)
    return CalibrationPrior(h_tt=np.array(h_tt, copy=True), h_pp=np.array(h_pp, copy=True))


def compute_calibration_schur_compact(
    problem: CalibrationProblem,
    selection: Sequence[int] | np.ndarray,
    info_blocks: CandidateInfoBlocks,
    prior: Optional[np.ndarray | CalibrationPrior] = None,
) -> np.ndarray:
    prior_blocks = _coerce_prior_blocks(problem, prior)
    weights = np.asarray(selection, dtype=float)
    h_cal = np.array(prior_blocks.h_tt, copy=True)
    reg = 1e-9 * np.eye(problem.pose_dim, dtype=float)

    for idx, weight in enumerate(weights):
        if weight == 0.0:
            continue
        weighted_h_tp = weight * info_blocks.h_tp[idx]
        h_cal += weight * info_blocks.h_tt[idx]
        pose_block = prior_blocks.h_pp + weight * info_blocks.h_pp[idx]
        h_cal -= weighted_h_tp @ np.linalg.pinv(pose_block + reg) @ weighted_h_tp.T

    return 0.5 * (h_cal + h_cal.T)


def compute_min_eig_score(
    problem: CalibrationProblem,
    selection: Sequence[int] | np.ndarray,
    info_blocks: CandidateInfoBlocks,
    prior: Optional[np.ndarray | CalibrationPrior] = None,
) -> float:
    h_cal = compute_calibration_schur_compact(problem, selection, info_blocks, prior=prior)
    return float(np.min(np.linalg.eigvalsh(h_cal)))


def compute_combined_fim(selection: np.ndarray, inf_mats: np.ndarray, prior: np.ndarray) -> np.ndarray:
    combined = prior.copy()
    for weight, mat in zip(selection, inf_mats):
        combined += float(weight) * mat
    return combined


def compute_calibration_schur(fim: np.ndarray, intrinsics_dim: int) -> np.ndarray:
    """Keep the intrinsics block and marginalize nuisance pose variables."""
    h_tt = fim[:intrinsics_dim, :intrinsics_dim]
    h_tn = fim[:intrinsics_dim, intrinsics_dim:]
    h_nt = fim[intrinsics_dim:, :intrinsics_dim]
    h_nn = fim[intrinsics_dim:, intrinsics_dim:]

    if h_nn.size == 0:
        h_cal = h_tt
    else:
        reg = 1e-9 * np.eye(h_nn.shape[0], dtype=float)
        h_nn_inv = np.linalg.pinv(h_nn + reg)
        h_cal = h_tt - h_tn @ h_nn_inv @ h_nt
    return 0.5 * (h_cal + h_cal.T)


def compute_info_metric(problem: CalibrationProblem, selection: Sequence[int], prior: Optional[np.ndarray] = None) -> float:
    info_blocks = construct_candidate_inf_blocks(problem)
    weights = np.zeros(problem.num_candidates, dtype=float)
    weights[np.asarray(selection, dtype=int)] = 1.0
    return compute_min_eig_score(problem, weights, info_blocks, prior=prior)


def find_min_eig_pair(inf_mats: np.ndarray, selection: np.ndarray, prior: np.ndarray, intrinsics_dim: int):
    final_fim = compute_combined_fim(selection, inf_mats, prior)
    h_cal = compute_calibration_schur(final_fim, intrinsics_dim)
    eigvals, eigvecs = la.eigh(h_cal)
    return eigvals[0], eigvecs[:, 0], final_fim
