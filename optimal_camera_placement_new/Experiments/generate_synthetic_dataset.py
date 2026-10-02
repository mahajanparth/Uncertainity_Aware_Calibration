"""Standalone synthetic calibration-dataset generator.

Generates hemisphere camera-pose datasets without Isaac Sim.
Inlines pure-math helpers from isaacsim_interactive_calibration_loop.py
and imports board-point builders from calibration_board_utils.py.

Usage:
    python3 -m Experiments.generate_synthetic_dataset \
        --board-type charuco --board-rows 13 --board-cols 13 \
        --board-square-size 0.0422 --board-marker-size 0.0321 \
        [--k1 FLOAT] [--k2 FLOAT] [--k3 FLOAT] [--p1 FLOAT] [--p2 FLOAT] \
        --output-dir PATH
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from OASIS import FIM as fim
from isaac_sim_scripts.calibration_board_utils import (
    build_aprilgrid_corner_points_local,
    build_charuco_corner_points_local,
)

# ---------------------------------------------------------------------------
# Pure-math helpers (inlined from isaacsim_interactive_calibration_loop.py)
# ---------------------------------------------------------------------------

def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def _rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    trace = np.trace(rotation)
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rotation[2, 1] - rotation[1, 2]) / s
        y = (rotation[0, 2] - rotation[2, 0]) / s
        z = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        w = (rotation[2, 1] - rotation[1, 2]) / s
        x = 0.25 * s
        y = (rotation[0, 1] + rotation[1, 0]) / s
        z = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        w = (rotation[0, 2] - rotation[2, 0]) / s
        x = (rotation[0, 1] + rotation[1, 0]) / s
        y = 0.25 * s
        z = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        w = (rotation[1, 0] - rotation[0, 1]) / s
        x = (rotation[0, 2] + rotation[2, 0]) / s
        y = (rotation[1, 2] + rotation[2, 1]) / s
        z = 0.25 * s
    quat = np.array([w, x, y, z], dtype=np.float64)
    return quat / np.linalg.norm(quat)


def _quaternion_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _usd_rotation_to_calibration_rotation(rotation_usd: np.ndarray) -> np.ndarray:
    return rotation_usd @ np.diag([1.0, -1.0, -1.0])


def _axis_angle_to_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    omc = 1.0 - c
    return np.array(
        [
            [c + x * x * omc, x * y * omc - z * s, x * z * omc + y * s],
            [y * x * omc + z * s, c + y * y * omc, y * z * omc - x * s],
            [z * x * omc - y * s, z * y * omc + x * s, c + z * z * omc],
        ],
        dtype=np.float64,
    )


def _apply_camera_tilt(
    camera_orientation: np.ndarray, yaw_deg: float, pitch_deg: float
) -> np.ndarray:
    base_rotation = _quaternion_to_rotation_matrix(camera_orientation)
    yaw_rotation = _axis_angle_to_rotation_matrix(
        np.array([0.0, 1.0, 0.0], dtype=np.float64), math.radians(yaw_deg)
    )
    pitch_rotation = _axis_angle_to_rotation_matrix(
        np.array([1.0, 0.0, 0.0], dtype=np.float64), math.radians(pitch_deg)
    )
    tilted_rotation = base_rotation @ yaw_rotation @ pitch_rotation
    return _rotation_matrix_to_quaternion(tilted_rotation)


def _camera_pose_look_at(
    camera_position: np.ndarray, target_position: np.ndarray
) -> tuple:
    forward = _normalize(target_position - camera_position)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _normalize(np.cross(forward, world_up))
    up = _normalize(np.cross(right, forward))
    rotation = np.column_stack((right, up, -forward))
    return camera_position, _rotation_matrix_to_quaternion(rotation)


def _generate_hemisphere_camera_poses(
    target_center: np.ndarray,
    radii: list,
    elevations_deg: list,
    azimuth_count: int,
    yaw_offsets_deg: list,
    pitch_offsets_deg: list,
) -> list:
    poses = []
    for radius in sorted(radii):
        for elevation_idx, elevation_deg in enumerate(sorted(elevations_deg)):
            elevation_rad = math.radians(elevation_deg)
            xy_radius = radius * math.cos(elevation_rad)
            z_offset = radius * math.sin(elevation_rad)
            azimuth_indices = range(azimuth_count)
            if elevation_idx % 2 == 1:
                azimuth_indices = reversed(list(azimuth_indices))
            for azimuth_idx in azimuth_indices:
                azimuth = 2.0 * math.pi * azimuth_idx / azimuth_count
                camera_position = target_center + np.array(
                    [
                        xy_radius * math.cos(azimuth),
                        xy_radius * math.sin(azimuth),
                        z_offset,
                    ],
                    dtype=np.float64,
                )
                _, base_orientation = _camera_pose_look_at(camera_position, target_center)
                for yaw_deg in yaw_offsets_deg:
                    for pitch_deg in pitch_offsets_deg:
                        poses.append(
                            (camera_position, _apply_camera_tilt(base_orientation, yaw_deg, pitch_deg))
                        )
    return poses


# ---------------------------------------------------------------------------
# Board target point builder
# ---------------------------------------------------------------------------

def _build_target_points(
    board_type: str,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    target_center: np.ndarray,
) -> np.ndarray:
    if board_type == "charuco":
        local_pts = build_charuco_corner_points_local(rows, cols, square_size, marker_size)
    elif board_type == "aprilgrid":
        tag_spacing = max(square_size / marker_size - 1.0, 0.0)
        local_pts = build_aprilgrid_corner_points_local(
            rows, cols, tag_size=marker_size, tag_spacing=tag_spacing
        )
    else:
        raise ValueError(f"Unknown board_type '{board_type}'. Use 'charuco' or 'aprilgrid'.")

    width = cols * square_size
    height = rows * square_size
    pts_world = np.zeros((local_pts.shape[0], 3), dtype=float)
    pts_world[:, 0] = target_center[0] - width / 2.0 + local_pts[:, 0]
    pts_world[:, 1] = target_center[1] + height / 2.0 - local_pts[:, 1]
    pts_world[:, 2] = target_center[2] + 0.0038 + local_pts[:, 2]
    return pts_world


# ---------------------------------------------------------------------------
# Measurement generation
# ---------------------------------------------------------------------------

def _generate_measurements(
    target_points: np.ndarray,
    candidate_rotations: np.ndarray,
    candidate_translations: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple,
    pixel_noise_sigma: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width, height = image_size
    measurements = np.full(
        (candidate_rotations.shape[0], target_points.shape[0], 2), np.nan, dtype=float
    )
    for idx, (rotation_wc, translation_wc) in enumerate(
        zip(candidate_rotations, candidate_translations)
    ):
        projected = fim.project_points(target_points, rotation_wc, translation_wc, intrinsics)
        valid = (
            np.isfinite(projected).all(axis=1)
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] < width)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] < height)
        )
        noisy = projected.copy()
        if pixel_noise_sigma > 0.0:
            noisy[valid] += rng.normal(
                scale=pixel_noise_sigma, size=(int(np.count_nonzero(valid)), 2)
            )
        noisy[~valid] = np.nan
        measurements[idx] = noisy
    return measurements


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------

_SIGMA_VARIANTS = [
    (0.01, "all_candidate_problem.npz"),
    (0.5, "all_candidate_problem_sigma05.npz"),
    (1.0, "all_candidate_problem_sigma10.npz"),
]

_RADII = [0.28, 0.34, 0.40, 0.46, 0.52]
_ELEVATIONS_DEG = [12, 22, 32, 42, 52, 62]
_AZIMUTH_COUNT = 24
_TARGET_CENTER = np.array([0.4, 0.0, 0.0], dtype=np.float64)
_IMAGE_SIZE = (1920, 1080)
_FX = _FY = 1649.248
_CX, _CY = 960.0, 540.0


def generate(
    board_type: str,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    distortion: dict,
    output_dir: Path,
    seed: int = 0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    k1 = distortion.get("k1", 0.0)
    k2 = distortion.get("k2", 0.0)
    p1 = distortion.get("p1", 0.0)
    p2 = distortion.get("p2", 0.0)
    k3 = distortion.get("k3", 0.0)
    intrinsics_gt = np.array([_FX, _FY, _CX, _CY, k1, k2, p1, p2, k3], dtype=float)

    target_points = _build_target_points(
        board_type, rows, cols, square_size, marker_size, _TARGET_CENTER
    )
    pts_span_x = target_points[:, 0].max() - target_points[:, 0].min()
    pts_span_y = target_points[:, 1].max() - target_points[:, 1].min()
    print(
        f"Board: {board_type} {rows}x{cols}, pts={target_points.shape[0]}, "
        f"span=({pts_span_x*100:.1f}x{pts_span_y*100:.1f}) cm, "
        f"k1={k1:.3f}"
    )

    raw_poses = _generate_hemisphere_camera_poses(
        _TARGET_CENTER, _RADII, _ELEVATIONS_DEG, _AZIMUTH_COUNT, [0.0], [0.0]
    )
    candidate_rotations = np.array(
        [
            _usd_rotation_to_calibration_rotation(_quaternion_to_rotation_matrix(q))
            for _, q in raw_poses
        ],
        dtype=float,
    )
    candidate_translations = np.array([p for p, _ in raw_poses], dtype=float)
    n_poses = candidate_rotations.shape[0]
    print(f"  N={n_poses} hemisphere poses")

    intrinsics_init = intrinsics_gt + fim.default_intrinsics_init_offset(intrinsics_gt.size)

    for sigma, filename in _SIGMA_VARIANTS:
        measurements = _generate_measurements(
            target_points,
            candidate_rotations,
            candidate_translations,
            intrinsics_gt,
            _IMAGE_SIZE,
            sigma,
            seed=seed,
        )
        visible_views = int(
            np.sum(np.any(np.isfinite(measurements[:, :, 0]), axis=1))
        )
        problem = fim.CalibrationProblem(
            target_points=target_points,
            candidate_rotations=candidate_rotations,
            candidate_translations=candidate_translations,
            measurements=measurements,
            intrinsics_gt=intrinsics_gt,
            intrinsics_init=intrinsics_init,
            image_size=_IMAGE_SIZE,
            pixel_noise_sigma=sigma,
        )
        out_path = output_dir / filename
        fim.save_calibration_problem_npz(str(out_path), problem)
        print(f"  σ={sigma:.2f}: {visible_views}/{n_poses} views visible → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic calibration datasets (no Isaac Sim required)."
    )
    parser.add_argument("--board-type", required=True, choices=["charuco", "aprilgrid"])
    parser.add_argument("--board-rows", type=int, required=True)
    parser.add_argument("--board-cols", type=int, required=True)
    parser.add_argument("--board-square-size", type=float, required=True,
                        help="Square/tag pitch in metres")
    parser.add_argument("--board-marker-size", type=float, required=True,
                        help="Marker size in metres (ChArUco) or tag size (AprilGrid)")
    parser.add_argument("--k1", type=float, default=0.0)
    parser.add_argument("--k2", type=float, default=0.0)
    parser.add_argument("--k3", type=float, default=0.0)
    parser.add_argument("--p1", type=float, default=0.0)
    parser.add_argument("--p2", type=float, default=0.0)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    generate(
        board_type=args.board_type,
        rows=args.board_rows,
        cols=args.board_cols,
        square_size=args.board_square_size,
        marker_size=args.board_marker_size,
        distortion=dict(k1=args.k1, k2=args.k2, k3=args.k3, p1=args.p1, p2=args.p2),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
