#!/home/parth/anaconda3/envs/env_isaacsim/bin/python
"""Isaac Sim loop for iterative uncertainty-aware camera calibration capture.

This script:
1. builds a ChArUco or checkerboard scene in Isaac Sim
2. samples a bank of candidate camera poses
3. captures an initial seed set of images
4. evaluates the remaining poses against the current weakest calibration direction
5. recommends and optionally captures the next best batch of poses
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
from datetime import datetime
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
from omni.isaac.core.objects import VisualSphere
from omni.isaac.core.utils.prims import create_prim
from omni.isaac.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdLux

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from OASIS import FIM as fim
from OASIS import calibration_analysis as cal_analysis
from calibration_board_utils import (
    build_aprilgrid_corner_points_local,
    build_charuco_corner_points_local,
    create_charuco_board,
    ensure_aprilgrid_texture,
    ensure_pdf_texture,
    create_textured_board_mesh,
    detect_geometric_aprilgrid,
    detect_kalibr_aprilgrid,
    detect_true_apriltag_grid,
    ensure_charuco_texture,
    ensure_kalibr_aprilgrid_pdf,
    get_aruco_dictionary,
)
from plot_reprojection_errors_on_images import generate_reprojection_overlays
from ransac_filtering import filter_planar_correspondences_ransac
from render_distortion_utils import distort_rgb_image, has_nonzero_distortion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="isaac_interactive_outputs")
    parser.add_argument("--board-type", choices=["checkerboard", "charuco", "aprilgrid"], default="charuco")
    parser.add_argument("--board-rows", type=int, default=11)
    parser.add_argument("--board-cols", type=int, default=15)
    parser.add_argument("--board-square-size", type=float, default=0.021)
    parser.add_argument("--board-marker-size", type=float, default=0.016)
    parser.add_argument(
        "--aprilgrid-pdf",
        default=None,
        help="Optional Kalibr AprilGrid PDF to rasterize and use as the AprilGrid board texture.",
    )
    parser.add_argument(
        "--aprilgrid-use-kalibr-pdf-generator",
        action="store_true",
        help="Generate the AprilGrid texture PDF using Kalibr's `kalibr_create_target_pdf` command.",
    )
    parser.add_argument(
        "--kalibr-create-target-cmd",
        default="kalibr_create_target_pdf",
        help="Kalibr target generator executable name/path used with --aprilgrid-use-kalibr-pdf-generator.",
    )
    parser.add_argument(
        "--aprilgrid-board-size",
        type=float,
        default=None,
        help="Physical side length of a square AprilGrid PDF target, in meters. Use 0.8 for april_6x6_80x80cm_A0.pdf.",
    )
    parser.add_argument(
        "--aprilgrid-black-border-bits",
        type=int,
        default=1,
        help="Black border bits used when generating an AprilGrid texture without --aprilgrid-pdf.",
    )
    parser.add_argument("--resolution-width", type=int, default=1280)
    parser.add_argument("--resolution-height", type=int, default=720)
    parser.add_argument("--radii", type=float, nargs="+", default=[0.28, 0.34, 0.40, 0.46, 0.52])
    parser.add_argument("--elevations-deg", type=float, nargs="+", default=[12.0, 22.0, 32.0, 42.0, 52.0, 62.0])
    parser.add_argument("--azimuth-count", type=int, default=24)
    parser.add_argument("--yaw-offsets-deg", type=float, nargs="+", default=[0.0])
    parser.add_argument("--pitch-offsets-deg", type=float, nargs="+", default=[0.0])
    parser.add_argument("--startup-wait-frames", type=int, default=30)
    parser.add_argument("--pose-settle-frames", type=int, default=8)
    parser.add_argument("--capture-subframes", type=int, default=2)
    parser.add_argument("--initial-k", type=int, default=5)
    parser.add_argument("--max-captures", type=int, default=20)
    parser.add_argument(
        "--recommendation-batch-size",
        type=int,
        default=None,
        help=(
            "Number of new poses to optimize together after the seed captures. "
            "Default: all remaining captures up to --max-captures."
        ),
    )
    parser.add_argument("--initial-indices", type=int, nargs="*", default=None)
    parser.add_argument("--pixel-noise-sigma", type=float, default=1.0)
    parser.add_argument(
        "--min-aprilgrid-corners-for-calib",
        type=int,
        default=8,
        help="Minimum detected AprilGrid corners required per image when estimating intrinsics from image detections.",
    )
    parser.add_argument(
        "--ransac-filter-correspondences",
        action="store_true",
        help="Apply planar homography RANSAC per view to reject outlier target correspondences before calibration.",
    )
    parser.add_argument(
        "--ransac-reproj-threshold",
        type=float,
        default=3.0,
        help="Reprojection threshold in pixels for per-view planar homography RANSAC.",
    )
    parser.add_argument(
        "--ransac-min-inliers",
        type=int,
        default=8,
        help="Minimum number of inliers required to accept the per-view RANSAC filter result.",
    )
    parser.add_argument(
        "--aprilgrid-detector-backend",
        choices=["opencv", "geometric", "apriltag"],
        default="opencv",
        help=(
            "AprilGrid detector backend. 'opencv' decodes AprilTags with cv2.aruco. "
            "'geometric' fits the rendered grid geometry, and 'apriltag' uses a true AprilTag detector library if installed."
        ),
    )
    parser.add_argument(
        "--aprilgrid-subpix-window",
        type=int,
        default=5,
        help="Half-window size passed to AprilGrid cornerSubPix refinement.",
    )
    parser.add_argument(
        "--aprilgrid-subpix-max-iters",
        type=int,
        default=80,
        help="Maximum iterations for AprilGrid cornerSubPix refinement.",
    )
    parser.add_argument(
        "--aprilgrid-subpix-epsilon",
        type=float,
        default=0.01,
        help="Termination epsilon for AprilGrid cornerSubPix refinement.",
    )
    parser.add_argument(
        "--aprilgrid-max-subpix-displacement2",
        type=float,
        default=9.0,
        help="Maximum squared displacement allowed after OpenCV AprilGrid subpixel refinement.",
    )
    parser.add_argument("--dist-k1", type=float, default=0.0)
    parser.add_argument("--dist-k2", type=float, default=0.0)
    parser.add_argument("--dist-p1", type=float, default=0.0)
    parser.add_argument("--dist-p2", type=float, default=0.0)
    parser.add_argument("--dist-k3", type=float, default=0.0)
    parser.add_argument("--stop-min-eig", type=float, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--plot-points", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--rosbag-output",
        default=None,
        help="Optional ROS1 bag path to write selected capture images for Kalibr.",
    )
    parser.add_argument(
        "--rosbag-topics",
        nargs="+",
        default=["/cam0/image_raw"],
        help="ROS image topic names to write into --rosbag-output. Multiple topics duplicate the same image stream.",
    )
    parser.add_argument("--rosbag-fps", type=float, default=10.0, help="Timestamp spacing for exported bag images.")
    parser.add_argument("--rosbag-frame-id", default="cam0", help="Header frame_id for exported ROS Image messages.")
    parser.add_argument(
        "--rosbag-encoding",
        choices=["mono8", "rgb8"],
        default="mono8",
        help="ROS Image encoding for --rosbag-output. Kalibr is most reliable with mono8.",
    )
    parser.add_argument(
        "--kalibr-target-output",
        default=None,
        help="Optional Kalibr target YAML path. Currently writes AprilGrid targets.",
    )
    parser.add_argument(
        "--stay-open",
        action="store_true",
        help="Keep Isaac Sim open after the run finishes. By default Isaac Sim closes automatically.",
    )
    return parser.parse_args()


def save_rgb_image(rgb_array: np.ndarray, output_path: pathlib.Path) -> None:
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


def save_detection_verification(
    image_path: pathlib.Path,
    output_path: pathlib.Path,
    board_type: str,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    aruco_dict_name: str = "DICT_4X4_250",
    aprilgrid_detector_backend: str = "opencv",
    aprilgrid_board_size: float | None = None,
    aprilgrid_subpix_window: int = 5,
    aprilgrid_subpix_max_iters: int = 80,
    aprilgrid_subpix_epsilon: float = 0.01,
    aprilgrid_max_subpix_displacement2: float = 9.0,
) -> dict:
    import cv2

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        return {
            "verification_image": None,
            "detected": False,
            "num_markers": 0,
            "num_charuco_corners": 0,
            "note": "failed_to_read_image",
        }

    annotated = image_bgr.copy()

    if board_type == "checkerboard":
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray,
            (cols, rows),
            cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK,
        )
        if found and corners is not None:
            cv2.drawChessboardCorners(annotated, (cols, rows), corners, True)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
            return {
                "verification_image": str(output_path),
                "detected": True,
                "num_markers": 0,
                "num_charuco_corners": int(corners.reshape(-1, 2).shape[0]),
                "note": "checkerboard_corners_detected",
            }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        return {
            "verification_image": str(output_path),
            "detected": False,
            "num_markers": 0,
            "num_charuco_corners": 0,
            "note": "checkerboard_not_detected",
        }

    if board_type == "aprilgrid":
        if not hasattr(cv2, "aruco") or not hasattr(cv2.aruco, "detectMarkers"):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
            return {
                "verification_image": str(output_path),
                "detected": False,
                "num_markers": 0,
                "num_aprilgrid_corners": 0,
                "note": "opencv_aruco_detection_unavailable",
            }
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if aprilgrid_detector_backend == "geometric":
            tag_spacing = max(square_size / marker_size - 1.0, 0.0)
            detection = detect_geometric_aprilgrid(
                gray,
                rows=rows,
                cols=cols,
                tag_size=marker_size,
                tag_spacing=tag_spacing,
                board_width=aprilgrid_board_size,
                board_height=aprilgrid_board_size,
                subpix_window_half_width=aprilgrid_subpix_window,
                subpix_max_iters=aprilgrid_subpix_max_iters,
                subpix_epsilon=aprilgrid_subpix_epsilon,
            )
        elif aprilgrid_detector_backend == "apriltag":
            detection = detect_true_apriltag_grid(
                gray,
                rows=rows,
                cols=cols,
                subpix_window_half_width=aprilgrid_subpix_window,
                subpix_max_iters=aprilgrid_subpix_max_iters,
                subpix_epsilon=aprilgrid_subpix_epsilon,
                max_subpix_displacement2=aprilgrid_max_subpix_displacement2,
            )
        else:
            detection = detect_kalibr_aprilgrid(
                gray,
                rows=rows,
                cols=cols,
                subpix_window_half_width=aprilgrid_subpix_window,
                subpix_max_iters=aprilgrid_subpix_max_iters,
                subpix_epsilon=aprilgrid_subpix_epsilon,
                max_subpix_displacement2=aprilgrid_max_subpix_displacement2,
            )
        num_markers = int(detection["num_markers"])
        if num_markers > 0:
            for tag_detection in detection["detections"]:
                pts = np.round(tag_detection["corners"]).astype(int)
                cv2.polylines(annotated, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
                for corner_idx, ((x, y), point_idx) in enumerate(zip(pts, tag_detection["point_indices"])):
                    cv2.circle(annotated, (int(x), int(y)), 5, (0, 0, 255), thickness=-1)
                    cv2.putText(
                        annotated,
                        str(point_idx),
                        (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                center = pts.mean(axis=0).astype(int)
                cv2.putText(
                    annotated,
                    f"id={int(tag_detection['id'])}",
                    (int(center[0]) - 20, int(center[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        return {
            "verification_image": str(output_path),
            "detected": bool(detection["success"]),
            "num_markers": num_markers,
            "num_aprilgrid_corners": int(np.count_nonzero(detection["observed"])),
            "note": (
                f"{aprilgrid_detector_backend}_aprilgrid_detected"
                if detection["success"]
                else f"{aprilgrid_detector_backend}_aprilgrid_not_detected"
            ),
        }

    if not hasattr(cv2, "aruco"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        return {
            "verification_image": str(output_path),
            "detected": False,
            "num_markers": 0,
            "num_charuco_corners": 0,
            "note": "opencv_aruco_module_unavailable",
        }

    if not hasattr(cv2.aruco, "detectMarkers"):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), annotated)
        return {
            "verification_image": str(output_path),
            "detected": False,
            "num_markers": 0,
            "num_charuco_corners": 0,
            "note": "opencv_aruco_detection_unavailable",
        }

    aruco_dict = get_aruco_dictionary(aruco_dict_name)
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        detector_params = cv2.aruco.DetectorParameters_create()
    else:
        detector_params = cv2.aruco.DetectorParameters()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
    num_markers = 0 if ids is None else int(len(ids))
    num_charuco_corners = 0
    note = "charuco_markers_detected" if num_markers > 0 else "charuco_not_detected"
    if num_markers > 0:
        cv2.aruco.drawDetectedMarkers(annotated, corners, ids)
        if hasattr(cv2.aruco, "CharucoBoard_create") or hasattr(cv2.aruco, "CharucoBoard"):
            board = create_charuco_board(rows, cols, square_size, marker_size, aruco_dict_name)
            cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
            num_corners, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if num_corners is not None and num_corners > 0 and charuco_corners is not None:
                num_charuco_corners = int(num_corners)
                cv2.aruco.drawDetectedCornersCharuco(annotated, charuco_corners, charuco_ids)
                note = "charuco_corners_detected"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), annotated)
    return {
        "verification_image": str(output_path),
        "detected": bool(num_markers > 0),
        "num_markers": num_markers,
        "num_charuco_corners": num_charuco_corners,
        "note": note,
    }


def add_ground_plane(world: World) -> None:
    world.scene.add_default_ground_plane()


def add_dome_light() -> None:
    stage = get_current_stage()
    dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
    dome_light.CreateIntensityAttr(1200.0)
    dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    dome_light.CreateExposureAttr(0.0)


def _add_board_tile(world: World, prim_path: str, name: str, position: np.ndarray, scale: np.ndarray, color: np.ndarray) -> None:
    from omni.isaac.core.objects import VisualCuboid

    world.scene.add(VisualCuboid(prim_path=prim_path, name=name, position=position, scale=scale, color=color))


def build_calibration_board(
    world: World,
    board_type: str,
    board_path: str = "/World/CalibrationBoard",
    rows: int = 7,
    cols: int = 9,
    square_size: float = 0.035,
    marker_size: float | None = None,
    aprilgrid_pdf: str | None = None,
    aprilgrid_use_kalibr_pdf_generator: bool = False,
    kalibr_create_target_cmd: str = "kalibr_create_target_pdf",
    aprilgrid_board_size: float | None = None,
    aprilgrid_black_border_bits: int = 1,
    board_center: Iterable[float] = (0.4, 0.0, 0.0),
) -> np.ndarray:
    board_center = np.array(tuple(board_center), dtype=float)
    if marker_size is None:
        marker_size = square_size * 0.78
    create_prim(board_path, "Xform")
    stage = get_current_stage()

    board_width = cols * square_size
    board_height = rows * square_size
    if board_type == "aprilgrid" and aprilgrid_board_size is not None:
        board_width = float(aprilgrid_board_size)
        board_height = float(aprilgrid_board_size)

    backing_dims = np.array([board_width + 0.03, board_height + 0.03, 0.006])
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
            pathlib.Path(__file__).resolve().parent
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

    if board_type == "aprilgrid":
        tag_spacing = max(square_size / marker_size - 1.0, 0.0)
        grid_width = cols * marker_size + max(cols - 1, 0) * (square_size - marker_size)
        outer_margin_pitches = 0.0
        if aprilgrid_board_size is not None:
            outer_margin_pitches = max((float(aprilgrid_board_size) - grid_width) / 2.0 / square_size, 0.0)
        if aprilgrid_pdf:
            texture_path = ensure_pdf_texture(
                pathlib.Path(aprilgrid_pdf),
                pathlib.Path(__file__).resolve().parent
                / "generated_assets"
                / f"aprilgrid_{cols}x{rows}_from_pdf.png",
            )
        elif aprilgrid_use_kalibr_pdf_generator:
            generated_pdf = ensure_kalibr_aprilgrid_pdf(
                pathlib.Path(__file__).resolve().parent
                / "generated_assets"
                / f"aprilgrid_{cols}x{rows}_kalibr_tag{marker_size:.4f}_spacing{tag_spacing:.4f}.pdf",
                rows=rows,
                cols=cols,
                tag_size=marker_size,
                tag_spacing=tag_spacing,
                kalibr_create_target_cmd=kalibr_create_target_cmd,
            )
            texture_path = ensure_pdf_texture(
                generated_pdf,
                pathlib.Path(__file__).resolve().parent
                / "generated_assets"
                / f"aprilgrid_{cols}x{rows}_from_kalibr_pdf.png",
            )
        else:
            texture_path = ensure_aprilgrid_texture(
                pathlib.Path(__file__).resolve().parent
                / "generated_assets"
                / f"aprilgrid_{cols}x{rows}_tag{marker_size:.4f}_spacing{tag_spacing:.4f}_dict36h11.png",
                rows=rows,
                cols=cols,
                tag_size=marker_size,
                tag_spacing=tag_spacing,
                aruco_dict_name="DICT_APRILTAG_36h11",
                black_border_bits=aprilgrid_black_border_bits,
                outer_margin_pitches=outer_margin_pitches,
            )
        create_textured_board_mesh(
            stage,
            f"{board_path}/AprilGridSurface",
            texture_path,
            width=board_width,
            height=board_height,
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
    # USD cameras look along local -Z with +Y up. The calibration code assumes
    # a pinhole camera with +Z forward and +Y down in image coordinates.
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


def create_pose_markers(world: World, target_center: np.ndarray, camera_poses: list[tuple[np.ndarray, np.ndarray]]) -> list[str]:
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
        marker_path = f"/World/PoseMarkers/CameraPose_{pose_idx:04d}"
        world.scene.add(
            VisualSphere(
                prim_path=marker_path,
                name=f"camera_pose_marker_{pose_idx:04d}",
                position=camera_position,
                radius=0.01,
                color=np.array([0.15, 0.65, 0.95]),
            )
        )
        marker_paths.append(marker_path)
    return marker_paths


def set_marker_color(marker_path: str, color: np.ndarray) -> None:
    prim = get_current_stage().GetPrimAtPath(marker_path)
    if not prim.IsValid():
        return
    displayable = UsdGeom.Gprim(prim)
    displayable.GetDisplayColorAttr().Set([Gf.Vec3f(*color.tolist())])


def set_pose_markers_visible(visible: bool) -> None:
    prim = get_current_stage().GetPrimAtPath("/World/PoseMarkers")
    if not prim.IsValid():
        return
    imageable = UsdGeom.Imageable(prim)
    imageable.MakeVisible() if visible else imageable.MakeInvisible()


def update_marker_colors(marker_paths: list[str], selected_indices: list[int], recommended_indices: Iterable[int] | None) -> None:
    selected = set(selected_indices)
    recommended = set(recommended_indices or [])
    for idx, marker_path in enumerate(marker_paths):
        if idx in selected:
            set_marker_color(marker_path, np.array([0.12, 0.72, 0.28]))
        elif idx in recommended:
            set_marker_color(marker_path, np.array([0.95, 0.25, 0.15]))
        else:
            set_marker_color(marker_path, np.array([0.15, 0.65, 0.95]))


def camera_intrinsics_px(camera_prim: UsdGeom.Camera, resolution: tuple[int, int]) -> np.ndarray:
    width_px, height_px = [float(v) for v in resolution]
    focal_length = float(camera_prim.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera_prim.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera_prim.GetVerticalApertureAttr().Get())
    fx = focal_length * width_px / horizontal_aperture
    fy = focal_length * height_px / vertical_aperture
    cx = width_px / 2.0
    cy = height_px / 2.0
    return np.array([fx, fy, cx, cy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)


def apply_distortion_parameters(intrinsics: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=float).copy()
    if intrinsics.size < 9:
        intrinsics = np.pad(intrinsics, (0, 9 - intrinsics.size))
    intrinsics[4:9] = np.array([args.dist_k1, args.dist_k2, args.dist_p1, args.dist_p2, args.dist_k3], dtype=float)
    return intrinsics


def build_target_points_world(
    board_type: str,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    board_center: np.ndarray,
    aprilgrid_board_size: float | None = None,
    z_offset: float = 0.0038,
) -> np.ndarray:
    width = cols * square_size
    height = rows * square_size
    if board_type == "aprilgrid" and aprilgrid_board_size is not None:
        width = float(aprilgrid_board_size)
        height = float(aprilgrid_board_size)
    if board_type == "charuco":
        local_points = build_charuco_corner_points_local(
            rows=rows,
            cols=cols,
            square_size=square_size,
            marker_size=marker_size,
            aruco_dict_name="DICT_4X4_250",
        )
    elif board_type == "aprilgrid":
        tag_spacing = max(square_size / marker_size - 1.0, 0.0)
        local_points = build_aprilgrid_corner_points_local(
            rows=rows,
            cols=cols,
            tag_size=marker_size,
            tag_spacing=tag_spacing,
            board_width=width if aprilgrid_board_size is not None else None,
            board_height=height if aprilgrid_board_size is not None else None,
        )
    else:
        inner_cols = max(cols - 1, 0)
        inner_rows = max(rows - 1, 0)
        xs = np.arange(inner_cols, dtype=float) + 1.0
        ys = np.arange(inner_rows, dtype=float) + 1.0
        grid_x, grid_y = np.meshgrid(xs, ys)
        local_points = np.stack(
            [grid_x.reshape(-1) * square_size, grid_y.reshape(-1) * square_size, np.zeros(grid_x.size, dtype=float)],
            axis=1,
        )

    points_world = np.zeros((local_points.shape[0], 3), dtype=float)
    points_world[:, 0] = board_center[0] - width / 2.0 + local_points[:, 0]
    points_world[:, 1] = board_center[1] + height / 2.0 - local_points[:, 1]
    points_world[:, 2] = board_center[2] + z_offset + local_points[:, 2]
    return points_world


def generate_measurements(
    target_points: np.ndarray,
    candidate_rotations: np.ndarray,
    candidate_translations: np.ndarray,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
    pixel_noise_sigma: float,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    width, height = image_size
    measurements = np.full((candidate_rotations.shape[0], target_points.shape[0], 2), np.nan, dtype=float)
    for idx, (rotation_wc, translation_wc) in enumerate(zip(candidate_rotations, candidate_translations)):
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
            noisy[valid] += rng.normal(scale=pixel_noise_sigma, size=(np.count_nonzero(valid), 2))
        noisy[~valid] = np.nan
        measurements[idx] = noisy
    return measurements


def choose_initial_indices(num_candidates: int, initial_k: int, explicit: list[int] | None) -> list[int]:
    if explicit:
        filtered = [idx for idx in explicit if 0 <= idx < num_candidates]
        return filtered[: min(initial_k, len(filtered))]
    if initial_k <= 0:
        return []
    if initial_k >= num_candidates:
        return list(range(num_candidates))
    return sorted(np.linspace(0, num_candidates - 1, initial_k, dtype=int).tolist())


def selection_weights(num_candidates: int, selected_indices: list[int]) -> np.ndarray:
    weights = np.zeros(num_candidates, dtype=float)
    if selected_indices:
        weights[np.asarray(selected_indices, dtype=int)] = 1.0
    return weights


def current_eigen_summary(
    problem: fim.CalibrationProblem,
    info_blocks: fim.CandidateInfoBlocks,
    prior: fim.CalibrationPrior,
    selected_indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    weights = selection_weights(problem.num_candidates, selected_indices)
    h_cal = fim.compute_calibration_schur_compact(problem, weights, info_blocks, prior=prior)
    eigvals, eigvecs = np.linalg.eigh(h_cal)
    return eigvals, eigvecs[:, 0]


def recommend_next_batch(
    problem: fim.CalibrationProblem,
    info_blocks: fim.CandidateInfoBlocks,
    prior: fim.CalibrationPrior,
    selected_indices: list[int],
    batch_size: int,
) -> tuple[list[int], float]:
    selected_set = set(selected_indices)
    weights = selection_weights(problem.num_candidates, selected_indices)
    best_score = float("-inf")
    batch_indices: list[int] = []

    for _ in range(batch_size):
        candidate_best_idx = -1
        candidate_best_score = float("-inf")
        for idx in range(problem.num_candidates):
            if idx in selected_set:
                continue
            trial = weights.copy()
            trial[idx] = 1.0
            score = fim.compute_min_eig_score(problem, trial, info_blocks, prior=prior)
            if score > candidate_best_score:
                candidate_best_score = score
                candidate_best_idx = idx

        if candidate_best_idx < 0:
            break

        weights[candidate_best_idx] = 1.0
        selected_set.add(candidate_best_idx)
        batch_indices.append(candidate_best_idx)
        best_score = candidate_best_score

    return batch_indices, best_score


def capture_pose_image(
    world: World,
    camera_prim: UsdGeom.Camera,
    rgb_annotator: object,
    pose: tuple[np.ndarray, np.ndarray],
    output_path: pathlib.Path,
    pose_settle_frames: int,
    capture_subframes: int,
    intrinsics: np.ndarray,
    original_output_path: pathlib.Path | None = None,
) -> None:
    position, orientation = pose
    set_pose_markers_visible(False)
    set_camera_pose(camera_prim, position, orientation)
    for _ in range(pose_settle_frames):
        world.step(render=True)
    rep.orchestrator.step(rt_subframes=capture_subframes, pause_timeline=False)
    rgb_data = rgb_annotator.get_data()
    rgb_image = np.asarray(rgb_data, dtype=np.uint8)
    if rgb_image.ndim == 3 and rgb_image.shape[-1] == 4:
        rgb_image = rgb_image[..., :3]
    if original_output_path is not None:
        save_rgb_image(rgb_image, original_output_path)
    rgb_image = distort_rgb_image(rgb_image, intrinsics)
    save_rgb_image(rgb_image, output_path)
    set_pose_markers_visible(True)


def maybe_prompt(selected_count: int, candidate_indices: list[int], predicted_score: float) -> bool:
    try:
        reply = input(
            f"[captures {selected_count + 1}-{selected_count + len(candidate_indices)}] "
            f"recommend poses {candidate_indices} "
            f"(predicted min_eig after batch={predicted_score:.6f}). Capture them? [Y/n/q]: "
        ).strip().lower()
    except EOFError:
        return True
    if reply in {"", "y", "yes"}:
        return True
    if reply in {"q", "quit"}:
        raise KeyboardInterrupt
    return False


def load_image_for_ros(image_path: pathlib.Path, encoding: str) -> np.ndarray:
    try:
        from PIL import Image

        mode = "L" if encoding == "mono8" else "RGB"
        return np.asarray(Image.open(image_path).convert(mode), dtype=np.uint8)
    except ImportError:
        import cv2

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image for rosbag export: {image_path}")
        if encoding == "mono8":
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def export_images_to_rosbag(
    image_paths: list[pathlib.Path],
    bag_path: pathlib.Path,
    topics: list[str],
    fps: float,
    frame_id: str,
    encoding: str,
) -> dict:
    if not image_paths:
        return {"created": False, "bag_path": str(bag_path), "reason": "no_images"}
    try:
        import rosbag
        import rospy
        from sensor_msgs.msg import Image
        from std_msgs.msg import Header
    except ImportError as exc:
        return {
            "created": False,
            "bag_path": str(bag_path),
            "reason": "ros_python_import_failed",
            "error": str(exc),
        }

    bag_path.parent.mkdir(parents=True, exist_ok=True)
    safe_fps = fps if fps > 0.0 else 10.0
    normalized_topics = topics or ["/cam0/image_raw"]
    with rosbag.Bag(str(bag_path), "w") as bag:
        for seq, image_path in enumerate(image_paths):
            image = load_image_for_ros(image_path, encoding)
            stamp = rospy.Time.from_sec(seq / safe_fps)
            msg = Image()
            msg.header = Header(seq=seq, stamp=stamp, frame_id=frame_id)
            msg.height = int(image.shape[0])
            msg.width = int(image.shape[1])
            msg.encoding = encoding
            msg.is_bigendian = 0
            msg.step = int(image.shape[1] if encoding == "mono8" else image.shape[1] * 3)
            msg.data = image.tobytes()
            for topic in normalized_topics:
                bag.write(topic, msg, stamp)

    return {
        "created": True,
        "bag_path": str(bag_path),
        "topics": normalized_topics,
        "num_images": len(image_paths),
        "fps": float(safe_fps),
        "frame_id": frame_id,
        "encoding": encoding,
        "duplicated_stream_to_multiple_topics": len(normalized_topics) > 1,
    }


def write_kalibr_target_yaml(
    output_path: pathlib.Path,
    board_type: str,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
) -> dict:
    if board_type != "aprilgrid":
        return {
            "created": False,
            "target_path": str(output_path),
            "reason": "kalibr_target_yaml_supported_for_aprilgrid_only",
        }
    tag_spacing = max(square_size / marker_size - 1.0, 0.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        "target_type: 'aprilgrid'\n"
        f"tagCols: {cols}\n"
        f"tagRows: {rows}\n"
        f"tagSize: {marker_size:.12g}\n"
        f"tagSpacing: {tag_spacing:.12g}\n"
    )
    output_path.write_text(payload, encoding="ascii")
    return {
        "created": True,
        "target_path": str(output_path),
        "target_type": "aprilgrid",
        "tagCols": int(cols),
        "tagRows": int(rows),
        "tagSize": float(marker_size),
        "tagSpacing": float(tag_spacing),
    }


def export_iteration_summary(
    output_dir: pathlib.Path,
    args: argparse.Namespace,
    problem: fim.CalibrationProblem,
    selected_indices: list[int],
    history: list[dict],
    marker_count: int,
    provisional_intrinsics_init: np.ndarray | None = None,
    seed_intrinsics_report: dict | None = None,
    seed_batch_report: dict | None = None,
    rosbag_report: dict | None = None,
    kalibr_target_report: dict | None = None,
    final_intrinsics_report: dict | None = None,
    uncertainty_report: dict | None = None,
) -> None:
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "num_candidates": problem.num_candidates,
        "selected_indices": selected_indices,
        "history": history,
        "image_size": list(problem.image_size),
        "intrinsics_gt": problem.intrinsics_gt.tolist(),
        "intrinsics_init": problem.intrinsics_init.tolist(),
        "provisional_intrinsics_init": None if provisional_intrinsics_init is None else np.asarray(provisional_intrinsics_init, dtype=float).tolist(),
        "pixel_noise_sigma": float(problem.pixel_noise_sigma),
        "effective_fim_pixel_noise_sigma": fim.effective_pixel_noise_sigma(problem.pixel_noise_sigma),
        "marker_count": marker_count,
        "capture_image_layout": (
            {
                "distortion_enabled": True,
                "distorted_dir": str(output_dir / "captures" / "distorted"),
                "original_dir": str(output_dir / "captures" / "original"),
            }
            if has_nonzero_distortion(problem.intrinsics_gt)
            else {
                "distortion_enabled": False,
                "captures_dir": str(output_dir / "captures"),
            }
        ),
        "config": {
            "initial_k": args.initial_k,
            "max_captures": args.max_captures,
            "recommendation_batch_size": args.recommendation_batch_size,
            "stop_min_eig": args.stop_min_eig,
            "interactive": bool(args.interactive),
            "stay_open": bool(args.stay_open),
            "rosbag_output": args.rosbag_output,
            "rosbag_topics": args.rosbag_topics,
            "rosbag_fps": args.rosbag_fps,
            "rosbag_frame_id": args.rosbag_frame_id,
            "rosbag_encoding": args.rosbag_encoding,
            "kalibr_target_output": args.kalibr_target_output,
            "aprilgrid_pdf": args.aprilgrid_pdf,
            "aprilgrid_use_kalibr_pdf_generator": bool(args.aprilgrid_use_kalibr_pdf_generator),
            "kalibr_create_target_cmd": args.kalibr_create_target_cmd,
            "aprilgrid_detector_backend": args.aprilgrid_detector_backend,
            "aprilgrid_subpix_window": int(args.aprilgrid_subpix_window),
            "aprilgrid_subpix_max_iters": int(args.aprilgrid_subpix_max_iters),
            "aprilgrid_subpix_epsilon": float(args.aprilgrid_subpix_epsilon),
            "aprilgrid_max_subpix_displacement2": float(args.aprilgrid_max_subpix_displacement2),
            "aprilgrid_board_size": args.aprilgrid_board_size,
            "aprilgrid_black_border_bits": args.aprilgrid_black_border_bits,
            "min_aprilgrid_corners_for_calib": args.min_aprilgrid_corners_for_calib,
            "ransac_filter_correspondences": bool(args.ransac_filter_correspondences),
            "ransac_reproj_threshold": float(args.ransac_reproj_threshold),
            "ransac_min_inliers": int(args.ransac_min_inliers),
            "distortion_gt": {
                "k1": float(args.dist_k1),
                "k2": float(args.dist_k2),
                "p1": float(args.dist_p1),
                "p2": float(args.dist_p2),
                "k3": float(args.dist_k3),
            },
            "render_distortion_applied": True,
        },
    }
    payload["intrinsics_init_stages"] = {
        "provisional_offset_init": None if provisional_intrinsics_init is None else np.asarray(provisional_intrinsics_init, dtype=float).tolist(),
        "seed_estimated_init": None if seed_intrinsics_report is None else seed_intrinsics_report.get("intrinsics_vector"),
        "current_problem_intrinsics_init": np.asarray(problem.intrinsics_init, dtype=float).tolist(),
        "final_estimated_intrinsics": None if final_intrinsics_report is None else final_intrinsics_report.get("intrinsics_vector"),
    }
    if seed_intrinsics_report is not None:
        payload["seed_intrinsics_report"] = seed_intrinsics_report
    if seed_batch_report is not None:
        payload["seed_batch_report"] = seed_batch_report
    if rosbag_report is not None:
        payload["rosbag_report"] = rosbag_report
    if kalibr_target_report is not None:
        payload["kalibr_target_report"] = kalibr_target_report
    if final_intrinsics_report is not None:
        payload["final_intrinsics_report"] = final_intrinsics_report
    if uncertainty_report is not None:
        payload["uncertainty_report"] = uncertainty_report
    summary_path = output_dir / "interactive_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2), encoding="ascii")


def estimate_intrinsics_from_selection(
    problem: fim.CalibrationProblem,
    selected_indices: list[int],
    use_ransac_filter: bool = False,
    ransac_reproj_threshold: float = 3.0,
    ransac_min_inliers: int = 8,
) -> dict | None:
    if not selected_indices:
        return None

    import cv2

    object_points = []
    image_points = []
    view_reports = []
    planar_points = np.asarray(problem.target_points, dtype=np.float32).copy()
    planar_points[:, 2] = 0.0

    for idx in selected_indices:
        measurements = np.asarray(problem.measurements[idx], dtype=np.float32)
        valid = np.isfinite(measurements).all(axis=1)
        if np.count_nonzero(valid) < 4:
            view_reports.append(
                {"selection_index": int(idx), "used": False, "reason": "too_few_valid_measurements", "num_points": int(np.count_nonzero(valid))}
            )
            continue
        object_view = planar_points[valid]
        image_view = measurements[valid]
        ransac_report = None
        if use_ransac_filter:
            ransac_report = filter_planar_correspondences_ransac(
                object_view,
                image_view,
                reproj_threshold=ransac_reproj_threshold,
                min_inliers=ransac_min_inliers,
            )
            if ransac_report["used"]:
                inlier_mask = np.asarray(ransac_report["mask"], dtype=bool)
                object_view = object_view[inlier_mask]
                image_view = image_view[inlier_mask]
        if object_view.shape[0] < 4:
            view_reports.append(
                {
                    "selection_index": int(idx),
                    "used": False,
                    "reason": "too_few_points_after_ransac" if use_ransac_filter else "too_few_valid_measurements",
                    "num_points": int(object_view.shape[0]),
                    "ransac_report": ransac_report,
                }
            )
            continue
        object_points.append(object_view.reshape(-1, 1, 3))
        image_points.append(image_view.reshape(-1, 1, 2))
        view_reports.append(
            {
                "selection_index": int(idx),
                "used": True,
                "num_points": int(object_view.shape[0]),
                "ransac_report": ransac_report,
            }
        )

    if not object_points:
        return None

    fx, fy, cx, cy, dist_guess = fim.split_intrinsics(problem.intrinsics_init)
    camera_matrix_init = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs_init = np.asarray(dist_guess, dtype=np.float64).reshape(-1, 1)
    if dist_coeffs_init.shape[0] >= 5:
        dist_coeffs_init[4, 0] = 0.0
    flags = cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_K3
    retval, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        tuple(int(v) for v in problem.image_size),
        camera_matrix_init,
        dist_coeffs_init,
        flags=flags,
    )
    dist_vector = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    if dist_vector.size < 5:
        dist_vector = np.pad(dist_vector, (0, 5 - dist_vector.size))
    intrinsics = np.array(
        [camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2], *dist_vector[:5]],
        dtype=float,
    )
    return {
        "num_views_used": len(object_points),
        "rms_reprojection_error": float(retval),
        "camera_matrix": camera_matrix.astype(float).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist(),
        "intrinsics_vector": intrinsics.tolist(),
        "intrinsics_gt": np.asarray(problem.intrinsics_gt, dtype=float).tolist(),
        "intrinsics_init": np.asarray(problem.intrinsics_init, dtype=float).tolist(),
        "intrinsics_error_vs_gt": (intrinsics - np.asarray(problem.intrinsics_gt, dtype=float)).tolist(),
        "ransac_filter_used": bool(use_ransac_filter),
        "ransac_reproj_threshold": float(ransac_reproj_threshold),
        "ransac_min_inliers": int(ransac_min_inliers),
        "view_reports": view_reports,
    }


def estimate_intrinsics_from_aprilgrid_images(
    image_paths: list[pathlib.Path],
    target_points: np.ndarray,
    image_size: tuple[int, int],
    intrinsics_gt: np.ndarray,
    intrinsics_init: np.ndarray,
    rows: int,
    cols: int,
    min_observed_corners: int,
    detector_backend: str,
    square_size: float,
    marker_size: float,
    aprilgrid_board_size: float | None,
    use_ransac_filter: bool = False,
    ransac_reproj_threshold: float = 3.0,
    ransac_min_inliers: int = 8,
    aprilgrid_subpix_window: int = 5,
    aprilgrid_subpix_max_iters: int = 80,
    aprilgrid_subpix_epsilon: float = 0.01,
    aprilgrid_max_subpix_displacement2: float = 9.0,
) -> dict | None:
    if not image_paths:
        return None

    import cv2

    object_points = []
    image_points = []
    detections = []
    planar_points = np.asarray(target_points, dtype=np.float32).copy()
    planar_points[:, 2] = 0.0

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            detections.append({"image_path": str(image_path), "used": False, "reason": "image_read_failed"})
            continue
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        if detector_backend == "geometric":
            tag_spacing = max(square_size / marker_size - 1.0, 0.0)
            detection = detect_geometric_aprilgrid(
                gray,
                rows=rows,
                cols=cols,
                tag_size=marker_size,
                tag_spacing=tag_spacing,
                board_width=aprilgrid_board_size,
                board_height=aprilgrid_board_size,
                subpix_window_half_width=aprilgrid_subpix_window,
                subpix_max_iters=aprilgrid_subpix_max_iters,
                subpix_epsilon=aprilgrid_subpix_epsilon,
            )
        elif detector_backend == "apriltag":
            detection = detect_true_apriltag_grid(
                gray,
                rows=rows,
                cols=cols,
                subpix_window_half_width=aprilgrid_subpix_window,
                subpix_max_iters=aprilgrid_subpix_max_iters,
                subpix_epsilon=aprilgrid_subpix_epsilon,
                max_subpix_displacement2=aprilgrid_max_subpix_displacement2,
            )
        else:
            detection = detect_kalibr_aprilgrid(
                gray,
                rows=rows,
                cols=cols,
                subpix_window_half_width=aprilgrid_subpix_window,
                subpix_max_iters=aprilgrid_subpix_max_iters,
                subpix_epsilon=aprilgrid_subpix_epsilon,
                max_subpix_displacement2=aprilgrid_max_subpix_displacement2,
            )
        observed = np.asarray(detection["observed"], dtype=bool)
        observed_count = int(np.count_nonzero(observed))
        if observed_count < int(min_observed_corners):
            detections.append(
                {
                    "image_path": str(image_path),
                    "used": False,
                    "num_markers": int(detection["num_markers"]),
                    "num_observed_corners": observed_count,
                    "reason": "too_few_detected_corners",
                }
            )
            continue
        object_view = planar_points[observed]
        image_view = np.asarray(detection["image_points"], dtype=np.float32)[observed]
        ransac_report = None
        if use_ransac_filter:
            ransac_report = filter_planar_correspondences_ransac(
                object_view,
                image_view,
                reproj_threshold=ransac_reproj_threshold,
                min_inliers=ransac_min_inliers,
            )
            if ransac_report["used"]:
                inlier_mask = np.asarray(ransac_report["mask"], dtype=bool)
                object_view = object_view[inlier_mask]
                image_view = image_view[inlier_mask]
        if object_view.shape[0] < 4:
            detections.append(
                {
                    "image_path": str(image_path),
                    "used": False,
                    "num_markers": int(detection["num_markers"]),
                    "num_observed_corners": observed_count,
                    "num_points_after_ransac": int(object_view.shape[0]),
                    "reason": "too_few_points_after_ransac" if use_ransac_filter else "too_few_detected_corners",
                    "ransac_report": ransac_report,
                }
            )
            continue
        object_points.append(object_view.reshape(-1, 1, 3))
        image_points.append(image_view.reshape(-1, 1, 2))
        detections.append(
            {
                "image_path": str(image_path),
                "used": True,
                "num_markers": int(detection["num_markers"]),
                "num_observed_corners": observed_count,
                "num_points_after_ransac": int(object_view.shape[0]),
                "ransac_report": ransac_report,
            }
        )

    if not object_points:
        return None

    fx, fy, cx, cy, dist_guess = fim.split_intrinsics(intrinsics_init)
    camera_matrix_init = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist_coeffs_init = np.asarray(dist_guess, dtype=np.float64).reshape(-1, 1)
    flags = cv2.CALIB_USE_INTRINSIC_GUESS
    retval, camera_matrix, dist_coeffs, _rvecs, _tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        tuple(int(v) for v in image_size),
        camera_matrix_init,
        dist_coeffs_init,
        flags=flags,
    )
    dist_vector = np.asarray(dist_coeffs, dtype=float).reshape(-1)
    if dist_vector.size < 5:
        dist_vector = np.pad(dist_vector, (0, 5 - dist_vector.size))
    intrinsics = np.array(
        [camera_matrix[0, 0], camera_matrix[1, 1], camera_matrix[0, 2], camera_matrix[1, 2], *dist_vector[:5]],
        dtype=float,
    )
    return {
        "num_views_used": len(object_points),
        "num_views_checked": len(image_paths),
        "rms_reprojection_error": float(retval),
        "camera_matrix": camera_matrix.astype(float).tolist(),
        "dist_coeffs": np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist(),
        "intrinsics_vector": intrinsics.tolist(),
        "intrinsics_gt": np.asarray(intrinsics_gt, dtype=float).tolist(),
        "intrinsics_init": np.asarray(intrinsics_init, dtype=float).tolist(),
        "intrinsics_error_vs_gt": (intrinsics - np.asarray(intrinsics_gt, dtype=float)).tolist(),
        "observation_source": "kalibr_style_aprilgrid_image_detections",
        "aprilgrid_detector_backend": detector_backend,
        "calibration_model": "opencv_pinhole_radtan_k3_fixed",
        "min_observed_corners_per_view": int(min_observed_corners),
        "ransac_filter_used": bool(use_ransac_filter),
        "ransac_reproj_threshold": float(ransac_reproj_threshold),
        "ransac_min_inliers": int(ransac_min_inliers),
        "detections": detections,
    }


def main() -> None:
    args = parse_args()
    if args.headless:
        simulation_app.set_setting("/app/window/drawMouse", False)
        simulation_app.set_setting("/app/livestream/enabled", False)

    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    captures_dir = output_dir / "captures"
    verification_dir = output_dir / "verification"
    captures_dir.mkdir(parents=True, exist_ok=True)
    verification_dir.mkdir(parents=True, exist_ok=True)
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
        aprilgrid_pdf=args.aprilgrid_pdf,
        aprilgrid_use_kalibr_pdf_generator=args.aprilgrid_use_kalibr_pdf_generator,
        kalibr_create_target_cmd=args.kalibr_create_target_cmd,
        aprilgrid_board_size=args.aprilgrid_board_size,
        aprilgrid_black_border_bits=args.aprilgrid_black_border_bits,
    )
    camera_poses = generate_hemisphere_camera_poses(
        np.asarray(target_center, dtype=float),
        args.radii,
        args.elevations_deg,
        args.azimuth_count,
        args.yaw_offsets_deg,
        args.pitch_offsets_deg,
    )
    print(f"Prepared {len(camera_poses)} candidate camera poses")

    marker_paths = []
    if args.plot_points:
        marker_paths = create_pose_markers(world, np.asarray(target_center, dtype=float), camera_poses)

    camera_prim, rgb_annotator = create_free_camera("/World/InteractiveCalibrationCamera", resolution)
    rep.orchestrator.set_capture_on_play(False)

    print(f"Waiting {args.startup_wait_frames} frames before the iterative loop")
    warm_up(world, frames=args.startup_wait_frames)
    world.reset()
    warm_up(world, frames=max(12, args.pose_settle_frames))

    intrinsics_gt = apply_distortion_parameters(camera_intrinsics_px(camera_prim, resolution), args)
    distortion_enabled = has_nonzero_distortion(intrinsics_gt)
    if distortion_enabled:
        original_captures_dir = captures_dir / "original"
        distorted_captures_dir = captures_dir / "distorted"
        original_captures_dir.mkdir(parents=True, exist_ok=True)
        distorted_captures_dir.mkdir(parents=True, exist_ok=True)
        active_captures_dir = distorted_captures_dir
        print(f"Saving original renders to: {original_captures_dir}")
        print(f"Saving distorted renders to: {distorted_captures_dir}")
    else:
        original_captures_dir = None
        active_captures_dir = captures_dir
    provisional_intrinsics_init = intrinsics_gt + fim.default_intrinsics_init_offset(intrinsics_gt.size)
    target_points = build_target_points_world(
        args.board_type,
        args.board_rows,
        args.board_cols,
        args.board_square_size,
        args.board_marker_size,
        np.asarray(target_center, dtype=float),
        aprilgrid_board_size=args.aprilgrid_board_size,
    )
    candidate_translations = np.asarray([pose[0] for pose in camera_poses], dtype=float)
    candidate_rotations = np.asarray(
        [_usd_rotation_to_calibration_rotation(_quaternion_to_rotation_matrix(pose[1])) for pose in camera_poses],
        dtype=float,
    )
    measurements = generate_measurements(
        target_points=target_points,
        candidate_rotations=candidate_rotations,
        candidate_translations=candidate_translations,
        intrinsics=intrinsics_gt,
        image_size=resolution,
        pixel_noise_sigma=args.pixel_noise_sigma,
    )
    problem = fim.CalibrationProblem(
        target_points=target_points,
        candidate_rotations=candidate_rotations,
        candidate_translations=candidate_translations,
        measurements=measurements,
        intrinsics_gt=intrinsics_gt,
        intrinsics_init=provisional_intrinsics_init,
        image_size=resolution,
        pixel_noise_sigma=args.pixel_noise_sigma,
    )

    selected_indices = choose_initial_indices(problem.num_candidates, args.initial_k, args.initial_indices)
    history: list[dict] = []
    bag_image_paths: list[pathlib.Path] = []

    for order, idx in enumerate(selected_indices):
        image_name = f"seed_{order:03d}_pose_{idx:04d}.png"
        image_path = active_captures_dir / image_name
        original_image_path = None if original_captures_dir is None else original_captures_dir / image_name
        capture_pose_image(
            world,
            camera_prim,
            rgb_annotator,
            camera_poses[idx],
            image_path,
            pose_settle_frames=args.pose_settle_frames,
            capture_subframes=args.capture_subframes,
            intrinsics=intrinsics_gt,
            original_output_path=original_image_path,
        )
        verification = save_detection_verification(
            image_path=image_path,
            output_path=verification_dir / f"seed_{order:03d}_pose_{idx:04d}_verification.png",
            board_type=args.board_type,
            rows=args.board_rows,
            cols=args.board_cols,
            square_size=args.board_square_size,
            marker_size=args.board_marker_size,
            aprilgrid_detector_backend=args.aprilgrid_detector_backend,
            aprilgrid_board_size=args.aprilgrid_board_size,
            aprilgrid_subpix_window=args.aprilgrid_subpix_window,
            aprilgrid_subpix_max_iters=args.aprilgrid_subpix_max_iters,
            aprilgrid_subpix_epsilon=args.aprilgrid_subpix_epsilon,
            aprilgrid_max_subpix_displacement2=args.aprilgrid_max_subpix_displacement2,
        )
        seed_history = {
            "step_type": "seed",
            "selection_order": order,
            "candidate_index": idx,
            "image_path": str(image_path),
            "verification": verification,
        }
        if original_image_path is not None:
            seed_history["original_image_path"] = str(original_image_path)
        history.append(seed_history)
        bag_image_paths.append(image_path)

    if args.board_type == "aprilgrid":
        seed_intrinsics_report = estimate_intrinsics_from_aprilgrid_images(
            bag_image_paths,
            target_points,
            resolution,
            intrinsics_gt,
            provisional_intrinsics_init,
            args.board_rows,
            args.board_cols,
            args.min_aprilgrid_corners_for_calib,
            args.aprilgrid_detector_backend,
            args.board_square_size,
            args.board_marker_size,
            args.aprilgrid_board_size,
            args.ransac_filter_correspondences,
            args.ransac_reproj_threshold,
            args.ransac_min_inliers,
            args.aprilgrid_subpix_window,
            args.aprilgrid_subpix_max_iters,
            args.aprilgrid_subpix_epsilon,
            args.aprilgrid_max_subpix_displacement2,
        )
    else:
        seed_intrinsics_report = estimate_intrinsics_from_selection(
            problem,
            selected_indices,
            use_ransac_filter=args.ransac_filter_correspondences,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
            ransac_min_inliers=args.ransac_min_inliers,
        )
    if seed_intrinsics_report is not None:
        problem.intrinsics_init = np.asarray(seed_intrinsics_report["intrinsics_vector"], dtype=float)
        seed_intrinsics_report["source"] = "initial_seed_captures"
        seed_intrinsics_report["provisional_intrinsics_init"] = provisional_intrinsics_init.tolist()
        print("Seed-estimated intrinsics will be used as the optimization linearization point:")
        print(f"  views used: {seed_intrinsics_report['num_views_used']}")
        print(f"  rms reprojection error: {seed_intrinsics_report['rms_reprojection_error']:.6f}")
        print(f"  intrinsics [fx, fy, cx, cy, k1, k2, p1, p2, k3]: {seed_intrinsics_report['intrinsics_vector']}")
    else:
        seed_intrinsics_report = {
            "source": "provisional_offset_fallback",
            "note": "Seed intrinsic estimation failed; using the provisional offset from ground truth.",
            "provisional_intrinsics_init": provisional_intrinsics_init.tolist(),
        }
        print("Seed intrinsic estimation failed; using provisional intrinsics_init fallback.")

    fim.save_calibration_problem_npz(str(output_dir / "all_candidate_problem.npz"), problem)
    prior = fim.build_prior_blocks(problem)
    info_blocks = fim.construct_candidate_inf_blocks(problem)

    seed_eigvals, seed_weakest_vec = current_eigen_summary(problem, info_blocks, prior, selected_indices)
    seed_batch_report = {
        "description": "Calibration information after seed captures and before recommended-batch optimization.",
        "seed_indices": selected_indices.copy(),
        "intrinsics_init_source": seed_intrinsics_report["source"],
        "eigvals_before_optimization": seed_eigvals.tolist(),
        "min_eig_before_optimization": float(seed_eigvals[0]),
        "weakest_direction_before_optimization": seed_weakest_vec.tolist(),
    }

    total_target = min(args.max_captures, problem.num_candidates)
    while len(selected_indices) < total_target:
        eigvals, weakest_vec = current_eigen_summary(problem, info_blocks, prior, selected_indices)
        current_min_eig = float(eigvals[0])
        if args.stop_min_eig is not None and current_min_eig >= args.stop_min_eig:
            print(f"Stopping because min_eig {current_min_eig:.6f} >= target {args.stop_min_eig:.6f}")
            break

        remaining_target = total_target - len(selected_indices)
        batch_size = remaining_target
        if args.recommendation_batch_size is not None:
            batch_size = min(batch_size, max(1, args.recommendation_batch_size))

        next_indices, predicted_score = recommend_next_batch(
            problem,
            info_blocks,
            prior,
            selected_indices,
            batch_size=batch_size,
        )
        if not next_indices:
            print("No remaining feasible candidate pose found.")
            break

        update_marker_colors(marker_paths, selected_indices, next_indices)
        visible_points = [int(info_blocks.visible_counts[idx]) for idx in next_indices]
        print(
            f"Current selection size: {len(selected_indices)} | "
            f"current min_eig: {current_min_eig:.6f} | "
            f"recommend next batch: {next_indices} | "
            f"predicted batch min_eig: {predicted_score:.6f} | "
            f"visible points: {visible_points}"
        )
        print(f"Weakest direction components: {weakest_vec.tolist()}")

        capture_recommended = True
        if args.interactive:
            try:
                capture_recommended = maybe_prompt(len(selected_indices), next_indices, predicted_score)
            except KeyboardInterrupt:
                print("Stopping interactive loop at user request.")
                break
        if not capture_recommended:
            print("Skipping recommended batch and stopping loop.")
            break

        batch_start_order = len(selected_indices)
        batch_history = {
            "step_type": "recommended_batch",
            "batch_start_order": batch_start_order,
            "batch_indices": next_indices,
            "predicted_min_eig_after_batch": float(predicted_score),
            "captures": [],
        }
        for batch_order, next_idx in enumerate(next_indices):
            selection_order = len(selected_indices)
            image_name = f"iter_{selection_order:03d}_pose_{next_idx:04d}.png"
            image_path = active_captures_dir / image_name
            original_image_path = None if original_captures_dir is None else original_captures_dir / image_name
            capture_pose_image(
                world,
                camera_prim,
                rgb_annotator,
                camera_poses[next_idx],
                image_path,
                pose_settle_frames=args.pose_settle_frames,
                capture_subframes=args.capture_subframes,
                intrinsics=intrinsics_gt,
                original_output_path=original_image_path,
            )
            verification = save_detection_verification(
                image_path=image_path,
                output_path=verification_dir / f"iter_{selection_order:03d}_pose_{next_idx:04d}_verification.png",
                board_type=args.board_type,
                rows=args.board_rows,
                cols=args.board_cols,
                square_size=args.board_square_size,
                marker_size=args.board_marker_size,
                aprilgrid_detector_backend=args.aprilgrid_detector_backend,
                aprilgrid_board_size=args.aprilgrid_board_size,
                aprilgrid_subpix_window=args.aprilgrid_subpix_window,
                aprilgrid_subpix_max_iters=args.aprilgrid_subpix_max_iters,
                aprilgrid_subpix_epsilon=args.aprilgrid_subpix_epsilon,
                aprilgrid_max_subpix_displacement2=args.aprilgrid_max_subpix_displacement2,
            )
            selected_indices.append(next_idx)
            capture_history = {
                "selection_order": selection_order,
                "candidate_index": next_idx,
                "batch_order": batch_order,
                "visible_points": int(info_blocks.visible_counts[next_idx]),
                "image_path": str(image_path),
                "verification": verification,
            }
            if original_image_path is not None:
                capture_history["original_image_path"] = str(original_image_path)
            batch_history["captures"].append(capture_history)
            bag_image_paths.append(image_path)
        history.append(batch_history)

    update_marker_colors(marker_paths, selected_indices, None)
    final_weights = selection_weights(problem.num_candidates, selected_indices)
    final_score = fim.compute_min_eig_score(problem, final_weights, info_blocks, prior=prior)
    uncertainty_report = cal_analysis.before_after_calibration_summary(problem, selected_indices, prior=prior)
    if args.board_type == "aprilgrid":
        final_intrinsics_report = estimate_intrinsics_from_aprilgrid_images(
            bag_image_paths,
            target_points,
            resolution,
            intrinsics_gt,
            problem.intrinsics_init,
            args.board_rows,
            args.board_cols,
            args.min_aprilgrid_corners_for_calib,
            args.aprilgrid_detector_backend,
            args.board_square_size,
            args.board_marker_size,
            args.aprilgrid_board_size,
            args.ransac_filter_correspondences,
            args.ransac_reproj_threshold,
            args.ransac_min_inliers,
            args.aprilgrid_subpix_window,
            args.aprilgrid_subpix_max_iters,
            args.aprilgrid_subpix_epsilon,
            args.aprilgrid_max_subpix_displacement2,
        )
    else:
        final_intrinsics_report = estimate_intrinsics_from_selection(
            problem,
            selected_indices,
            use_ransac_filter=args.ransac_filter_correspondences,
            ransac_reproj_threshold=args.ransac_reproj_threshold,
            ransac_min_inliers=args.ransac_min_inliers,
        )
    rosbag_report = None
    if args.rosbag_output:
        rosbag_report = export_images_to_rosbag(
            bag_image_paths,
            pathlib.Path(args.rosbag_output).expanduser().resolve(),
            args.rosbag_topics,
            args.rosbag_fps,
            args.rosbag_frame_id,
            args.rosbag_encoding,
        )
        if rosbag_report.get("created"):
            print(f"Saved ROS bag to: {rosbag_report['bag_path']}")
        else:
            print(f"ROS bag export skipped: {rosbag_report.get('reason')} {rosbag_report.get('error', '')}")

    kalibr_target_report = None
    if args.kalibr_target_output:
        kalibr_target_report = write_kalibr_target_yaml(
            pathlib.Path(args.kalibr_target_output).expanduser().resolve(),
            args.board_type,
            args.board_rows,
            args.board_cols,
            args.board_square_size,
            args.board_marker_size,
        )
        if kalibr_target_report.get("created"):
            print(f"Saved Kalibr target YAML to: {kalibr_target_report['target_path']}")
        else:
            print(f"Kalibr target YAML export skipped: {kalibr_target_report.get('reason')}")

    export_iteration_summary(
        output_dir,
        args,
        problem,
        selected_indices,
        history,
        len(marker_paths),
        provisional_intrinsics_init=provisional_intrinsics_init,
        seed_intrinsics_report=seed_intrinsics_report,
        seed_batch_report=seed_batch_report,
        rosbag_report=rosbag_report,
        kalibr_target_report=kalibr_target_report,
        final_intrinsics_report=final_intrinsics_report,
        uncertainty_report=uncertainty_report,
    )
    summary_path = output_dir / "interactive_summary.json"
    try:
        overlay_kwargs = {
            "summary_path": summary_path,
            "output_dir": output_dir / "reprojection_overlays",
            "report_key": "final_intrinsics_report",
        }
        if args.board_type == "charuco":
            overlay_kwargs.update(
                {
                    "board_type": "charuco",
                    "board_rows": int(args.board_rows),
                    "board_cols": int(args.board_cols),
                    "board_square_size": float(args.board_square_size),
                    "board_marker_size": float(args.board_marker_size),
                }
            )
        generate_reprojection_overlays(**overlay_kwargs)
    except Exception as exc:
        print(f"Reprojection overlay export skipped: {exc}")
    print(f"Final selected indices: {selected_indices}")
    print(f"Final min_eig score: {final_score:.6f}")
    print("Uncertainty before vs after selection:")
    print(f"  before min_eig: {uncertainty_report['before']['min_eig']:.6f}")
    print(f"  after min_eig: {uncertainty_report['after']['min_eig']:.6f}")
    print(f"  before std_dev: {uncertainty_report['before']['std_dev']}")
    print(f"  after std_dev: {uncertainty_report['after']['std_dev']}")
    if final_intrinsics_report is not None:
        print("Final estimated intrinsics from selected captures:")
        print(f"  views used: {final_intrinsics_report['num_views_used']}")
        print(f"  rms reprojection error: {final_intrinsics_report['rms_reprojection_error']:.6f}")
        print(f"  intrinsics [fx, fy, cx, cy, k1, k2, p1, p2, k3]: {final_intrinsics_report['intrinsics_vector']}")
        print(f"  intrinsics error vs gt: {final_intrinsics_report['intrinsics_error_vs_gt']}")
        print(f"  camera matrix: {final_intrinsics_report['camera_matrix']}")
    else:
        print("Final intrinsic estimation was skipped because no valid selected views were available.")
    print(f"Saved full candidate problem to: {output_dir / 'all_candidate_problem.npz'}")
    print(f"Saved interactive summary to: {summary_path}")
    print(f"Saved reprojection overlays to: {output_dir / 'reprojection_overlays'}")
    if args.stay_open and not args.headless:
        print("Leaving Isaac Sim running because --stay-open was provided.")
        run_until_closed(world)
    else:
        print("Closing Isaac Sim.")
        simulation_app.close()


if __name__ == "__main__":
    main()
