#!/home/parth/anaconda3/envs/env_isaacsim/bin/python
"""Isaac Sim script that captures a checkerboard from many free-camera views.

This script:
1. Creates a ground plane.
2. Builds a checkerboard target.
3. Generates hemisphere camera viewpoints around the target.
4. Rotates the camera to look at the target from each viewpoint.
5. Saves RGB images and leaves Isaac Sim running until you close it.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def _bootstrap_headless_flag() -> bool:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--headless", action="store_true")
    args, _ = parser.parse_known_args()
    return args.headless


try:
    from isaacsim import SimulationApp
except ImportError:
    from omni.isaac.kit import SimulationApp


simulation_app = SimulationApp({"headless": _bootstrap_headless_flag()})

import omni.replicator.core as rep
from omni.isaac.core import World
from omni.isaac.core.objects import VisualCuboid, VisualSphere
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdLux

from calibration_board_utils import (
    build_charuco_corner_points_local,
    create_textured_board_mesh,
    ensure_charuco_texture,
)
from render_distortion_utils import distort_rgb_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="free_camera_outputs")
    parser.add_argument("--board-type", choices=["checkerboard", "charuco"], default="charuco")
    parser.add_argument("--board-rows", type=int, default=11)
    parser.add_argument("--board-cols", type=int, default=15)
    parser.add_argument("--board-square-size", type=float, default=0.021)
    parser.add_argument("--board-marker-size", type=float, default=0.016)
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--resolution-width", type=int, default=1280)
    parser.add_argument("--resolution-height", type=int, default=720)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.28, 0.32, 0.36, 0.40, 0.44, 0.48, 0.52])
    parser.add_argument("--elevations-deg", type=float, nargs="+", default=[10.0, 18.0, 26.0, 34.0, 42.0, 50.0, 58.0, 66.0])
    parser.add_argument("--azimuth-count", type=int, default=24)
    parser.add_argument("--yaw-offsets-deg", type=float, nargs="+", default=[0.0])
    parser.add_argument("--pitch-offsets-deg", type=float, nargs="+", default=[0.0])
    parser.add_argument("--startup-wait-frames", type=int, default=30)
    parser.add_argument("--pose-settle-frames", type=int, default=8)
    parser.add_argument("--capture-subframes", type=int, default=2)
    parser.add_argument("--plot-points", action="store_true")
    parser.add_argument("--dist-k1", type=float, default=0.0)
    parser.add_argument("--dist-k2", type=float, default=0.0)
    parser.add_argument("--dist-p1", type=float, default=0.0)
    parser.add_argument("--dist-p2", type=float, default=0.0)
    parser.add_argument("--dist-k3", type=float, default=0.0)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args()


def save_rgb_image(rgb_array: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if rgb_array.dtype != np.uint8:
        rgb_array = np.clip(rgb_array, 0, 255).astype(np.uint8)

    try:
        from PIL import Image

        Image.fromarray(rgb_array).save(output_path)
        return
    except ImportError:
        pass

    try:
        import imageio.v2 as imageio

        imageio.imwrite(output_path, rgb_array)
        return
    except ImportError:
        pass

    np.save(output_path.with_suffix(".npy"), rgb_array)
    print(
        f"Neither Pillow nor imageio is available. Saved NumPy array to "
        f"{output_path.with_suffix('.npy')}"
    )


def add_ground_plane(world: World) -> None:
    world.scene.add_default_ground_plane()


def add_dome_light() -> None:
    stage = get_current_stage()
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1200.0)
    dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    dome_light.CreateExposureAttr(0.0)


def build_checkerboard(
    world: World,
    board_path: str = "/World/CalibrationBoard",
    rows: int = 7,
    cols: int = 9,
    square_size: float = 0.035,
    board_center: Iterable[float] = (0.4, 0.0, 0.0),
) -> np.ndarray:
    board_center = np.array(tuple(board_center), dtype=float)
    create_prim(board_path, "Xform")

    backing_dims = np.array([cols * square_size + 0.03, rows * square_size + 0.03, 0.006])
    world.scene.add(
        VisualCuboid(
            prim_path=f"{board_path}/Backing",
            name="checkerboard_backing",
            position=board_center,
            scale=backing_dims,
            color=np.array([0.18, 0.18, 0.18]),
        )
    )

    x0 = -((cols - 1) * square_size) / 2.0
    y0 = ((rows - 1) * square_size) / 2.0
    for row in range(rows):
        for col in range(cols):
            color_value = 0.97 if (row + col) % 2 == 0 else 0.06
            tile_center = board_center + np.array([x0 + col * square_size, y0 - row * square_size, 0.0035])
            world.scene.add(
                VisualCuboid(
                    prim_path=f"{board_path}/Square_{row}_{col}",
                    name=f"checker_square_{row}_{col}",
                    position=tile_center,
                    scale=np.array([square_size, square_size, 0.001]),
                    color=np.array([color_value, color_value, color_value]),
                )
            )
    return board_center


def _add_board_tile(world: World, prim_path: str, name: str, position: np.ndarray, scale: np.ndarray, color: np.ndarray) -> None:
    world.scene.add(VisualCuboid(prim_path=prim_path, name=name, position=position, scale=scale, color=color))


def build_calibration_board(
    world: World,
    board_type: str,
    board_path: str = "/World/CalibrationBoard",
    rows: int = 7,
    cols: int = 9,
    square_size: float = 0.035,
    marker_size: float | None = None,
    board_center: Iterable[float] = (0.4, 0.0, 0.0),
) -> np.ndarray:
    board_center = np.array(tuple(board_center), dtype=float)
    if marker_size is None:
        marker_size = square_size * 0.78
    create_prim(board_path, "Xform")
    stage = get_current_stage()

    backing_dims = np.array([cols * square_size + 0.03, rows * square_size + 0.03, 0.006])
    _add_board_tile(
        world,
        f"{board_path}/Backing",
        "calibration_board_backing",
        board_center,
        backing_dims,
        np.array([0.18, 0.18, 0.18]),
    )

    if board_type == "charuco":
        texture_path = ensure_charuco_texture(
            Path(__file__).resolve().parent
            / "generated_assets"
            / f"charuco_{cols}x{rows}_sq{square_size:.4f}_mk{marker_size:.4f}_dict4x4_250.png",
            rows=rows,
            cols=cols,
            square_size=square_size,
            marker_size=marker_size,
            aruco_dict_name="DICT_4X4_250",
        )
        create_textured_board_mesh(
            stage,
            f"{board_path}/CharucoSurface",
            texture_path,
            width=cols * square_size,
            height=rows * square_size,
            center=board_center,
        )
        return board_center

    x0 = -((cols - 1) * square_size) / 2.0
    y0 = ((rows - 1) * square_size) / 2.0
    for row in range(rows):
        for col in range(cols):
            is_dark_square = (row + col) % 2 == 0
            square_center = board_center + np.array([x0 + col * square_size, y0 - row * square_size, 0.0035])
            square_color = np.array([0.97, 0.97, 0.97]) if is_dark_square else np.array([0.06, 0.06, 0.06])

            _add_board_tile(
                world,
                f"{board_path}/Square_{row}_{col}",
                f"calibration_square_{row}_{col}",
                square_center,
                np.array([square_size, square_size, 0.001]),
                square_color,
            )

    return board_center


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
    # USD cameras look along local -Z with +Y up. The calibration pipeline
    # expects +Z forward and +Y down.
    return rotation_usd @ np.diag([1.0, -1.0, -1.0])


def _axis_angle_to_rotation_matrix(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    x, y, z = axis
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s, x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c, y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s, c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def _apply_camera_tilt(camera_orientation: np.ndarray, yaw_deg: float, pitch_deg: float) -> np.ndarray:
    base_rotation = _quaternion_to_rotation_matrix(camera_orientation)
    yaw_rotation = _axis_angle_to_rotation_matrix(np.array([0.0, 1.0, 0.0], dtype=np.float64), math.radians(yaw_deg))
    pitch_rotation = _axis_angle_to_rotation_matrix(np.array([1.0, 0.0, 0.0], dtype=np.float64), math.radians(pitch_deg))
    tilted_rotation = base_rotation @ yaw_rotation @ pitch_rotation
    return _rotation_matrix_to_quaternion(tilted_rotation)


def _board_local_points_to_world(points_local: np.ndarray, board_center: np.ndarray, width: float, height: float, z_offset: float) -> np.ndarray:
    points_local = np.asarray(points_local, dtype=np.float64)
    points_world = np.zeros((points_local.shape[0], 3), dtype=np.float64)
    points_world[:, 0] = board_center[0] - width / 2.0 + points_local[:, 0]
    points_world[:, 1] = board_center[1] + height / 2.0 - points_local[:, 1]
    points_world[:, 2] = board_center[2] + z_offset + points_local[:, 2]
    return points_world


def build_board_corner_metadata(
    board_type: str,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    board_center: np.ndarray,
    aruco_dict_name: str = "DICT_4X4_250",
    z_offset: float = 0.0038,
) -> dict:
    width = cols * square_size
    height = rows * square_size
    metadata = {
        "board_type": board_type,
        "rows": int(rows),
        "cols": int(cols),
        "square_size": float(square_size),
        "marker_size": float(marker_size),
        "aruco_dict_name": aruco_dict_name,
        "board_center_world": board_center.astype(float).tolist(),
        "board_rotation_world": np.eye(3, dtype=float).tolist(),
        "board_width": float(width),
        "board_height": float(height),
        "board_z_offset": float(z_offset),
    }

    if board_type == "charuco":
        local_points = build_charuco_corner_points_local(
            rows=rows,
            cols=cols,
            square_size=square_size,
            marker_size=marker_size,
            aruco_dict_name=aruco_dict_name,
        )
        corner_ids = np.arange(local_points.shape[0], dtype=int)
        metadata["corner_ids"] = corner_ids.tolist()
        metadata["corner_points_local"] = local_points.tolist()
        metadata["corner_points_world"] = _board_local_points_to_world(local_points, board_center, width, height, z_offset).tolist()
        return metadata

    inner_cols = max(cols - 1, 0)
    inner_rows = max(rows - 1, 0)
    xs = np.arange(inner_cols, dtype=np.float64) + 1.0
    ys = np.arange(inner_rows, dtype=np.float64) + 1.0
    grid_x, grid_y = np.meshgrid(xs, ys)
    local_points = np.stack(
        [grid_x.reshape(-1) * square_size, grid_y.reshape(-1) * square_size, np.zeros(grid_x.size, dtype=np.float64)],
        axis=1,
    )
    corner_ids = np.arange(local_points.shape[0], dtype=int)
    metadata["corner_ids"] = corner_ids.tolist()
    metadata["corner_points_local"] = local_points.tolist()
    metadata["corner_points_world"] = _board_local_points_to_world(local_points, board_center, width, height, z_offset).tolist()
    return metadata


def camera_pose_look_at(camera_position: np.ndarray, target_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = _normalize(target_position - camera_position)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = _normalize(np.cross(forward, world_up))
    up = _normalize(np.cross(right, forward))
    rotation = np.column_stack((right, up, -forward))
    return camera_position, _rotation_matrix_to_quaternion(rotation)


def generate_hemisphere_camera_poses(
    target_center: np.ndarray,
    radii: list[float],
    elevations_deg: list[float],
    azimuth_count: int,
    yaw_offsets_deg: list[float],
    pitch_offsets_deg: list[float],
) -> list[tuple[np.ndarray, np.ndarray]]:
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
                    [xy_radius * math.cos(azimuth), xy_radius * math.sin(azimuth), z_offset],
                    dtype=np.float64,
                )
                _, base_orientation = camera_pose_look_at(camera_position, target_center)
                for yaw_deg in yaw_offsets_deg:
                    for pitch_deg in pitch_offsets_deg:
                        poses.append((camera_position, _apply_camera_tilt(base_orientation, yaw_deg, pitch_deg)))
    return poses


def add_pose_markers(world: World, target_center: np.ndarray, camera_poses: list[tuple[np.ndarray, np.ndarray]]) -> None:
    create_prim("/World/PoseMarkers", "Xform")
    world.scene.add(
        VisualSphere(
            prim_path="/World/PoseMarkers/TargetCenter",
            name="target_center_marker",
            position=target_center,
            radius=0.02,
            color=np.array([0.95, 0.2, 0.2]),
        )
    )
    for pose_idx, (camera_position, _) in enumerate(camera_poses):
        world.scene.add(
            VisualSphere(
                prim_path=f"/World/PoseMarkers/CameraPose_{pose_idx:03d}",
                name=f"camera_pose_marker_{pose_idx:03d}",
                position=camera_position,
                radius=0.01,
                color=np.array([0.15, 0.65, 0.95]),
            )
        )


def create_free_camera(camera_path: str, resolution: tuple[int, int]) -> tuple[UsdGeom.Camera, object]:
    stage = get_current_stage()
    camera_prim = UsdGeom.Camera.Define(stage, camera_path)
    camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera_prim.CreateFocalLengthAttr(18.0)
    camera_prim.CreateHorizontalApertureAttr(20.955)
    camera_prim.CreateVerticalApertureAttr(20.955 * float(resolution[1]) / float(resolution[0]))
    render_product = rep.create.render_product(camera_path, resolution)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])
    return camera_prim, rgb_annotator


def set_camera_pose(camera_prim: UsdGeom.Camera, position: np.ndarray, orientation: np.ndarray) -> None:
    xform = UsdGeom.Xformable(camera_prim.GetPrim())
    translate_op = None
    orient_op = None

    for op in xform.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate and translate_op is None:
            translate_op = op
        elif op.GetOpType() == UsdGeom.XformOp.TypeOrient and orient_op is None:
            orient_op = op

    if translate_op is None:
        translate_op = xform.AddTranslateOp()
    if orient_op is None:
        orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)

    translate_op.Set(Gf.Vec3d(*position.tolist()))
    orient_op.Set(Gf.Quatf(float(orientation[0]), Gf.Vec3f(*orientation[1:].astype(np.float32).tolist())))


def warm_up(world: World, frames: int = 60) -> None:
    for _ in range(frames):
        world.step(render=True)
    rep.orchestrator.step(rt_subframes=2, pause_timeline=False)


def run_until_closed(world: World) -> None:
    while simulation_app.is_running():
        world.step(render=True)


def export_dataset_metadata(
    output_dir: Path,
    resolution: tuple[int, int],
    camera_prim: UsdGeom.Camera,
    board_metadata: dict,
    frames: list[dict],
    distortion_coeffs: np.ndarray,
) -> Path:
    width_px, height_px = [int(v) for v in resolution]
    focal_length = float(camera_prim.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera_prim.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera_prim.GetVerticalApertureAttr().Get())
    fx = focal_length * width_px / horizontal_aperture
    fy = focal_length * height_px / vertical_aperture
    cx = width_px / 2.0
    cy = height_px / 2.0

    payload = {
        "dataset_type": "isaac_sim_calibration_capture",
        "image_size": [width_px, height_px],
        "camera": {
            "focal_length_mm": focal_length,
            "horizontal_aperture_mm": horizontal_aperture,
            "vertical_aperture_mm": vertical_aperture,
            "intrinsics_px": [fx, fy, cx, cy, *np.asarray(distortion_coeffs, dtype=float).reshape(-1)[:5].tolist()],
            "render_distortion_applied": True,
        },
        "board": board_metadata,
        "frames": frames,
    }
    output_path = output_dir / "dataset_metadata.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="ascii")
    return output_path


def capture_sequence(
    world: World,
    camera_prim: UsdGeom.Camera,
    rgb_annotator: object,
    camera_poses: list[tuple[np.ndarray, np.ndarray]],
    pose_settle_frames: int,
    capture_subframes: int,
    output_dir: Path,
    distortion_intrinsics: np.ndarray,
) -> list[dict]:
    rgb_dir = output_dir / "rgb"
    frames = []
    for image_idx, (camera_position, camera_orientation) in enumerate(camera_poses):
        set_camera_pose(camera_prim, camera_position, camera_orientation)
        for _ in range(pose_settle_frames):
            world.step(render=True)
        rep.orchestrator.step(rt_subframes=capture_subframes, pause_timeline=False)
        rgb_data = rgb_annotator.get_data()
        rgb_image = np.asarray(rgb_data, dtype=np.uint8)
        if rgb_image.ndim == 3 and rgb_image.shape[-1] == 4:
            rgb_image = rgb_image[..., :3]
        rgb_image = distort_rgb_image(rgb_image, distortion_intrinsics)
        image_name = f"frame_{image_idx:04d}.png"
        image_path = rgb_dir / image_name
        save_rgb_image(rgb_image, image_path)
        frames.append(
            {
                "frame_index": image_idx,
                "image_relative_path": str(Path("rgb") / image_name),
                "translation_wc": camera_position.astype(float).tolist(),
                "quaternion_wc": camera_orientation.astype(float).tolist(),
                "rotation_wc_usd": _quaternion_to_rotation_matrix(camera_orientation).astype(float).tolist(),
                "rotation_wc": _usd_rotation_to_calibration_rotation(_quaternion_to_rotation_matrix(camera_orientation)).astype(float).tolist(),
            }
        )
        print(f"Captured frame {image_idx:03d}")
    return frames


def main() -> None:
    args = parse_args()
    if args.headless:
        simulation_app.set_setting("/app/window/drawMouse", False)
        simulation_app.set_setting("/app/livestream/enabled", False)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    resolution = (args.resolution_width, args.resolution_height)

    world = World(stage_units_in_meters=1.0)
    add_ground_plane(world)
    add_dome_light()
    target_center = build_calibration_board(
        world,
        board_type=args.board_type,
        rows=args.board_rows,
        cols=args.board_cols,
        square_size=args.board_square_size,
        marker_size=args.board_marker_size,
    )
    camera_poses = generate_hemisphere_camera_poses(
        target_center,
        args.radii,
        args.elevations_deg,
        args.azimuth_count,
        args.yaw_offsets_deg,
        args.pitch_offsets_deg,
    )
    if args.num_images > 0:
        camera_poses = camera_poses[: args.num_images]
    print(f"Saving images to: {output_dir}")
    print(f"Full output path: {output_dir}")
    print(f"Prepared {len(camera_poses)} camera poses")
    if args.plot_points:
        add_pose_markers(world, target_center, camera_poses)
    camera_prim, rgb_annotator = create_free_camera("/World/FreeCaptureCamera", resolution)
    rep.orchestrator.set_capture_on_play(False)
    width_px, height_px = [float(v) for v in resolution]
    focal_length = float(camera_prim.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera_prim.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera_prim.GetVerticalApertureAttr().Get())
    distortion_intrinsics = np.array(
        [
            focal_length * width_px / horizontal_aperture,
            focal_length * height_px / vertical_aperture,
            width_px / 2.0,
            height_px / 2.0,
            args.dist_k1,
            args.dist_k2,
            args.dist_p1,
            args.dist_p2,
            args.dist_k3,
        ],
        dtype=float,
    )
    board_metadata = build_board_corner_metadata(
        board_type=args.board_type,
        rows=args.board_rows,
        cols=args.board_cols,
        square_size=args.board_square_size,
        marker_size=args.board_marker_size,
        board_center=np.asarray(target_center, dtype=np.float64),
    )

    print(f"Waiting {args.startup_wait_frames} frames before loading capture sequence")
    warm_up(world, frames=args.startup_wait_frames)
    world.reset()
    warm_up(world, frames=max(12, args.pose_settle_frames))
    frames = capture_sequence(
        world,
        camera_prim,
        rgb_annotator,
        camera_poses,
        pose_settle_frames=args.pose_settle_frames,
        capture_subframes=args.capture_subframes,
        output_dir=output_dir,
        distortion_intrinsics=distortion_intrinsics,
    )
    metadata_path = export_dataset_metadata(
        output_dir=output_dir,
        resolution=resolution,
        camera_prim=camera_prim,
        board_metadata=board_metadata,
        frames=frames,
        distortion_coeffs=distortion_intrinsics[4:9],
    )
    print(f"Saved dataset metadata to: {metadata_path}")
    print("Image capture complete. Leaving Isaac Sim running until you close it.")
    run_until_closed(world)


if __name__ == "__main__":
    main()
