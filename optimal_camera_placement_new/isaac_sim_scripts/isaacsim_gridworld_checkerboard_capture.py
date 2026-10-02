#!/home/parth/anaconda3/envs/env_isaacsim/bin/python
"""Standalone Isaac Sim script for checkerboard image capture.

This script:
1. Creates a simple tiled grid-world floor.
2. Loads a Franka manipulator.
3. Attaches a camera under the Franka hand.
4. Builds a checkerboard target in the scene.
5. Moves the arm through a few viewpoints and saves RGB images.

Run with the `env_isaacsim` Python, for example:
    /home/parth/anaconda3/envs/env_isaacsim/bin/python \
        isaacsim_gridworld_checkerboard_capture.py --num-images 8
"""

from __future__ import annotations

import argparse
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
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import add_reference_to_stage, get_current_stage
from omni.isaac.core.utils.types import ArticulationAction
from pxr import Gf, UsdGeom, UsdPhysics

from calibration_board_utils import create_textured_board_mesh, ensure_charuco_texture

try:
    from omni.isaac.franka import Franka
except ImportError:
    Franka = None

try:
    from isaacsim.robot.manipulators.examples.franka import KinematicsSolver as FrankaKinematicsSolver
except ImportError:
    FrankaKinematicsSolver = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="manipulator_checkerboard_outputs")
    parser.add_argument("--board-type", choices=["checkerboard", "charuco"], default="charuco")
    parser.add_argument("--board-rows", type=int, default=11)
    parser.add_argument("--board-cols", type=int, default=15)
    parser.add_argument("--board-square-size", type=float, default=0.021)
    parser.add_argument("--board-marker-size", type=float, default=0.016)
    parser.add_argument("--num-images", type=int, default=0)
    parser.add_argument("--resolution-width", type=int, default=1280)
    parser.add_argument("--resolution-height", type=int, default=720)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.32, 0.38, 0.44, 0.50])
    parser.add_argument("--elevations-deg", type=float, nargs="+", default=[20.0, 30.0, 40.0, 50.0])
    parser.add_argument("--azimuth-count", type=int, default=12)
    parser.add_argument("--yaw-offsets-deg", type=float, nargs="+", default=[0.0])
    parser.add_argument("--pitch-offsets-deg", type=float, nargs="+", default=[0.0])
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
            tile_center = board_center + np.array(
                [x0 + col * square_size, y0 - row * square_size, 0.0035]
            )
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


def _add_board_tile(
    world: World,
    prim_path: str,
    name: str,
    position: np.ndarray,
    scale: np.ndarray,
    color: np.ndarray,
) -> None:
    world.scene.add(
        VisualCuboid(
            prim_path=prim_path,
            name=name,
            position=position,
            scale=scale,
            color=color,
        )
    )


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


def _wait_for_stage_updates(frames: int = 10) -> None:
    for _ in range(frames):
        simulation_app.update()


def _find_articulation_root_path(root_path: str) -> str | None:
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return None

    stack = [root_prim]
    while stack:
        prim = stack.pop()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            return prim.GetPath().pathString
        stack.extend(reversed(list(prim.GetChildren())))
    return None


def _find_existing_hand_path(base_root_path: str) -> str:
    stage = get_current_stage()
    root_prim = stage.GetPrimAtPath(base_root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Invalid robot root for hand search: {base_root_path}")

    preferred_name_tokens = ("panda_hand", "hand", "gripper", "finger")
    rigid_body_candidates = []
    fallback_candidates = []

    stack = [root_prim]
    while stack:
        prim = stack.pop()
        prim_name = prim.GetName().lower()
        prim_path = prim.GetPath().pathString
        if any(token in prim_name for token in preferred_name_tokens):
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                rigid_body_candidates.append(prim_path)
            fallback_candidates.append(prim_path)
        stack.extend(reversed(list(prim.GetChildren())))

    for suffix in ("panda_hand", "right_gripper", "hand"):
        for candidate in rigid_body_candidates:
            if candidate.endswith(f"/{suffix}"):
                return candidate

    if rigid_body_candidates:
        return rigid_body_candidates[0]

    candidate_paths = [
        f"{base_root_path}/panda_hand",
        f"{base_root_path}/right_gripper",
        f"{base_root_path}/hand",
        "/World/Franka/panda_hand",
        "/World/Franka/panda/panda_hand",
    ]
    for candidate in candidate_paths:
        if stage.GetPrimAtPath(candidate).IsValid():
            return candidate

    if fallback_candidates:
        return fallback_candidates[0]

    raise RuntimeError(
        "Could not find a valid Franka hand prim for camera attachment. "
        f"Tried: {candidate_paths}"
    )


def add_franka_robot(world: World) -> tuple[Robot, str]:
    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError(
            "Could not locate the Isaac Sim assets root. Make sure Nucleus or "
            "local Isaac Sim assets are available."
        )

    franka_path = "/World/Franka"
    if Franka is not None:
        robot = world.scene.add(Franka(prim_path=franka_path, name="franka"))
        _wait_for_stage_updates()
        return robot, _find_existing_hand_path(franka_path)

    franka_usd = assets_root + "/Isaac/Robots/Franka/franka_alt_fingers.usd"
    add_reference_to_stage(franka_usd, franka_path)
    _wait_for_stage_updates()

    articulation_root_path = _find_articulation_root_path(franka_path)
    if articulation_root_path is None:
        raise RuntimeError(
            "The Franka USD was loaded, but no articulation root was found under "
            f"{franka_path}. Check the referenced asset at {franka_usd}."
        )

    robot = world.scene.add(Robot(prim_path=articulation_root_path, name="franka"))
    hand_path = _find_existing_hand_path(articulation_root_path)
    return robot, hand_path


def create_hand_camera(
    camera_path: str,
    resolution: tuple[int, int],
) -> tuple[str, object]:
    stage = get_current_stage()
    camera_prim = UsdGeom.Camera.Define(stage, camera_path)
    camera_xform = UsdGeom.Xformable(camera_prim.GetPrim())

    for op in camera_xform.GetOrderedXformOps():
        camera_xform.RemoveXformOp(op)

    camera_xform.AddTranslateOp().Set(Gf.Vec3d(0.08, 0.0, 0.045))
    camera_xform.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, 0.0))
    camera_prim.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera_prim.CreateFocalLengthAttr(18.0)
    camera_prim.CreateHorizontalApertureAttr(20.955)

    render_product = rep.create.render_product(camera_path, resolution)
    rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb_annotator.attach([render_product])
    return camera_path, rgb_annotator


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
    quaternion = np.array([w, x, y, z], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _camera_pose_look_at(camera_position: np.ndarray, target_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forward = _normalize(target_position - camera_position)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(forward, world_up)) > 0.98:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    right = _normalize(np.cross(forward, world_up))
    up = _normalize(np.cross(right, forward))

    # USD camera convention: local -Z is view direction, +Y is up.
    rotation = np.column_stack((right, up, -forward))
    return camera_position, _rotation_matrix_to_quaternion(rotation)


def _quat_to_rotation_matrix(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


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
    base_rotation = _quat_to_rotation_matrix(camera_orientation)
    yaw_rotation = _axis_angle_to_rotation_matrix(np.array([0.0, 1.0, 0.0], dtype=np.float64), math.radians(yaw_deg))
    pitch_rotation = _axis_angle_to_rotation_matrix(np.array([1.0, 0.0, 0.0], dtype=np.float64), math.radians(pitch_deg))
    tilted_rotation = base_rotation @ yaw_rotation @ pitch_rotation
    return _rotation_matrix_to_quaternion(tilted_rotation)


def _hand_target_from_camera_pose(
    camera_position: np.ndarray, camera_orientation: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    camera_in_hand_translation = np.array([0.08, 0.0, 0.045], dtype=np.float64)
    camera_in_hand_rotation = np.diag([1.0, -1.0, -1.0])

    world_from_camera = _quat_to_rotation_matrix(camera_orientation)
    world_from_hand = world_from_camera @ camera_in_hand_rotation.T
    hand_position = camera_position - world_from_hand @ camera_in_hand_translation
    hand_orientation = _rotation_matrix_to_quaternion(world_from_hand)
    return hand_position, hand_orientation


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
            # Alternate sweep direction by elevation ring to keep consecutive IK targets nearby.
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
                        poses.append((camera_position, _apply_camera_tilt(base_orientation, yaw_deg, pitch_deg)))
    return poses


def add_pose_markers(world: World, target_center: np.ndarray, camera_poses: list[tuple[np.ndarray, np.ndarray]]) -> list[str]:
    create_prim("/World/PoseMarkers", "Xform")
    marker_paths = []

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
        marker_path = f"/World/PoseMarkers/CameraPose_{pose_idx:03d}"
        world.scene.add(
            VisualSphere(
                prim_path=marker_path,
                name=f"camera_pose_marker_{pose_idx:03d}",
                position=camera_position,
                radius=0.012,
                color=np.array([0.15, 0.65, 0.95]),
            )
        )
        marker_paths.append(marker_path)

    return marker_paths


def set_marker_color(marker_path: str, color: np.ndarray) -> None:
    stage = get_current_stage()
    prim = stage.GetPrimAtPath(marker_path)
    if not prim.IsValid():
        return
    displayable = UsdGeom.Gprim(prim)
    displayable.GetDisplayColorAttr().Set([Gf.Vec3f(*color.tolist())])


def warm_up(world: World, frames: int = 60) -> None:
    for _ in range(frames):
        world.step(render=True)


def run_until_closed(world: World) -> None:
    while simulation_app.is_running():
        world.step(render=True)


def filter_feasible_camera_poses(
    world: World,
    robot: Robot,
    ik_solver: object,
    camera_poses: list[tuple[np.ndarray, np.ndarray]],
) -> list[tuple[tuple[np.ndarray, np.ndarray], object]]:
    feasible_poses = []
    controller = robot.get_articulation_controller()

    for pose_idx, (camera_position, camera_orientation) in enumerate(camera_poses):
        hand_position, hand_orientation = _hand_target_from_camera_pose(camera_position, camera_orientation)
        action, success = ik_solver.compute_inverse_kinematics(
            target_position=hand_position,
            target_orientation=hand_orientation,
            position_tolerance=0.02,
            orientation_tolerance=0.3,
        )
        if not success:
            continue

        feasible_poses.append(((camera_position, camera_orientation), action))
        for _ in range(20):
            controller.apply_action(action)
            world.step(render=True)

        if pose_idx % 25 == 0:
            print(f"Feasibility pre-pass: checked {pose_idx + 1}/{len(camera_poses)} poses")

    print(f"Feasibility pre-pass kept {len(feasible_poses)} / {len(camera_poses)} poses")
    return feasible_poses


def capture_sequence(
    world: World,
    robot: Robot,
    rgb_annotator: object,
    output_dir: Path,
    target_center: np.ndarray,
    feasible_camera_poses: list[tuple[tuple[np.ndarray, np.ndarray], object]],
) -> None:
    camera_poses = [pose for pose, _ in feasible_camera_poses]
    marker_paths = add_pose_markers(world, target_center, camera_poses)

    controller = robot.get_articulation_controller()
    for image_idx, ((_, _), action) in enumerate(feasible_camera_poses):
        controller = robot.get_articulation_controller()
        for _ in range(60):
            controller.apply_action(action)
            world.step(render=True)

        set_marker_color(marker_paths[image_idx], np.array([0.2, 0.9, 0.25]))

        rgb_data = rgb_annotator.get_data()
        if rgb_data is None:
            raise RuntimeError("RGB annotator did not return image data.")

        rgb_image = np.array(rgb_data)
        if rgb_image.ndim == 3 and rgb_image.shape[2] == 4:
            rgb_image = rgb_image[:, :, :3]

        output_path = output_dir / f"checkerboard_{image_idx:03d}.png"
        save_rgb_image(rgb_image, output_path)
        print(f"Saved {output_path}")


def main() -> None:
    args = parse_args()
    if args.headless:
        simulation_app.set_setting("/app/window/drawMouse", False)
        simulation_app.set_setting("/app/livestream/enabled", False)

    output_dir = Path(args.output_dir).resolve()
    resolution = (args.resolution_width, args.resolution_height)

    world = World(stage_units_in_meters=1.0)
    add_ground_plane(world)
    target_center = build_calibration_board(
        world,
        board_type=args.board_type,
        rows=args.board_rows,
        cols=args.board_cols,
        square_size=args.board_square_size,
        marker_size=args.board_marker_size,
    )
    robot, hand_path = add_franka_robot(world)

    camera_path, rgb_annotator = create_hand_camera(
        f"{hand_path}/hand_camera",
        resolution=resolution,
    )

    # Let all references and sensors initialize before driving the robot.
    world.reset()
    robot.initialize()
    if FrankaKinematicsSolver is None:
        raise RuntimeError("Franka kinematics solver is not available in this Isaac Sim install.")
    ik_solver = FrankaKinematicsSolver(robot)
    print(f"Camera attached at {camera_path}")
    print(f"Saving images to: {output_dir}")
    print(f"Full output path: {output_dir}")
    warm_up(world)
    candidate_camera_poses = generate_hemisphere_camera_poses(
        target_center,
        args.radii,
        args.elevations_deg,
        args.azimuth_count,
        args.yaw_offsets_deg,
        args.pitch_offsets_deg,
    )
    feasible_camera_poses = filter_feasible_camera_poses(world, robot, ik_solver, candidate_camera_poses)

    world.reset()
    robot.initialize()
    ik_solver = FrankaKinematicsSolver(robot)
    warm_up(world)
    capture_sequence(
        world,
        robot,
        rgb_annotator,
        output_dir,
        target_center,
        feasible_camera_poses[: args.num_images] if args.num_images > 0 else feasible_camera_poses,
    )
    print("Image capture complete. Leaving Isaac Sim running until you close it.")
    run_until_closed(world)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
