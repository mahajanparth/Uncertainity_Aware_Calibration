#!/usr/bin/env python3
"""Detect checkerboard or ChArUco corners in images and draw the UV detections.

Examples:
    python detect_checkerboard_uv.py \
        --input /home/parth/Desktop/nvidia_env/free_camera_outputs_100_fast/rgb \
        --rows 7 --cols 9

    python detect_checkerboard_uv.py \
        --input image.png \
        --rows 7 --cols 9 \
        --output-dir annotated_checkerboard_uv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from calibration_board_utils import create_charuco_board


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Input image path or directory of images.")
    parser.add_argument("--output-dir", default="checkerboard_uv_output")
    parser.add_argument("--board-type", choices=["checkerboard", "charuco"], default="charuco")
    parser.add_argument("--plot-heatmap", action="store_true", help="Aggregate all detected UV points into one heatmap image.")
    parser.add_argument("--heatmap-blur", type=int, default=31, help="Gaussian blur kernel size for the aggregate heatmap.")
    parser.add_argument("--rows", type=int, default=11, help="Number of checkerboard inner-corner rows or ChArUco square rows.")
    parser.add_argument("--cols", type=int, default=15, help="Number of checkerboard inner-corner cols or ChArUco square cols.")
    parser.add_argument("--square-size", type=float, default=0.021, help="Board square size in meters.")
    parser.add_argument("--marker-size", type=float, default=0.016, help="ChArUco marker size in meters.")
    parser.add_argument("--aruco-dict", default="DICT_4X4_250", help="OpenCV ArUco dictionary name.")
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".png", ".jpg", ".jpeg", ".bmp"],
        help="Image extensions to scan when --input is a directory.",
    )
    parser.add_argument(
        "--subpix-window",
        type=int,
        default=11,
        help="Subpixel corner refinement window size.",
    )
    return parser.parse_args()


def get_aruco_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build does not include the aruco module.")

    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")

    dictionary_id = getattr(cv2.aruco, dictionary_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def collect_images(input_path: Path, extensions: list[str]) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    normalized_exts = {ext.lower() for ext in extensions}
    return sorted(
        path for path in input_path.iterdir() if path.is_file() and path.suffix.lower() in normalized_exts
    )


def detect_checkerboard(
    image_bgr: np.ndarray,
    rows: int,
    cols: int,
    subpix_window: int,
) -> tuple[bool, np.ndarray | None]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    pattern_size = (cols, rows)

    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FAST_CHECK
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (subpix_window, subpix_window),
        (-1, -1),
        criteria,
    )
    return True, refined


def detect_charuco(
    image_bgr: np.ndarray,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    aruco_dict_name: str,
) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build does not include cv2.aruco, required for ChArUco detection.")

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    aruco_dict = get_aruco_dictionary(aruco_dict_name)
    board = create_charuco_board(rows, cols, square_size, marker_size, aruco_dict_name)

    if hasattr(cv2.aruco, "DetectorParameters_create"):
        detector_params = cv2.aruco.DetectorParameters_create()
    else:
        detector_params = cv2.aruco.DetectorParameters()

    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)
    if ids is None or len(ids) == 0:
        return False, None, None

    cv2.aruco.refineDetectedMarkers(gray, board, corners, ids, rejectedCorners=None)
    num_corners, charuco_corners, _ = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
    if num_corners is None or num_corners <= 0 or charuco_corners is None:
        return False, None, None

    charuco_ids = np.asarray(_, dtype=np.int32).reshape(-1)
    return True, charuco_corners, charuco_ids


def annotate_uv_image(
    image_bgr: np.ndarray,
    corners: np.ndarray,
    rows: int,
    cols: int,
) -> np.ndarray:
    annotated = image_bgr.copy()
    cv2.drawChessboardCorners(annotated, (cols, rows), corners, True)

    for idx, corner in enumerate(corners.reshape(-1, 2)):
        u, v = corner
        point = (int(round(u)), int(round(v)))
        cv2.circle(annotated, point, 4, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"{idx}",
            (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"({u:.1f}, {v:.1f})",
            (point[0] + 6, point[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    draw_average_uv_point(annotated, corners)
    return annotated


def annotate_charuco_uv_image(
    image_bgr: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    annotated = image_bgr.copy()
    for idx, corner in enumerate(corners.reshape(-1, 2)):
        u, v = corner
        point = (int(round(u)), int(round(v)))
        cv2.circle(annotated, point, 5, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"{idx}",
            (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"({u:.1f}, {v:.1f})",
            (point[0] + 6, point[1] + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    draw_average_uv_point(annotated, corners)
    return annotated


def draw_average_uv_point(image_bgr: np.ndarray, corners: np.ndarray) -> None:
    uv_points = corners.reshape(-1, 2)
    avg_u, avg_v = uv_points.mean(axis=0)
    point = (int(round(avg_u)), int(round(avg_v)))
    cv2.circle(image_bgr, point, 8, (255, 0, 255), 2)
    cv2.drawMarker(
        image_bgr,
        point,
        (255, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.putText(
        image_bgr,
        f"avg ({avg_u:.1f}, {avg_v:.1f})",
        (point[0] + 8, point[1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 0, 255),
        1,
        cv2.LINE_AA,
    )


def save_uv_points(output_path: Path, corners: np.ndarray) -> None:
    uv_points = corners.reshape(-1, 2)
    header = "corner_index,u,v"
    lines = [header]
    for idx, (u, v) in enumerate(uv_points):
        lines.append(f"{idx},{u:.6f},{v:.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def save_charuco_points(output_path: Path, corners: np.ndarray, corner_ids: np.ndarray) -> None:
    uv_points = corners.reshape(-1, 2)
    header = "charuco_id,u,v"
    lines = [header]
    for corner_id, (u, v) in zip(corner_ids.reshape(-1), uv_points):
        lines.append(f"{int(corner_id)},{u:.6f},{v:.6f}")
    output_path.write_text("\n".join(lines) + "\n", encoding="ascii")


def build_uv_heatmap(
    image_shape: tuple[int, int, int],
    all_corners: list[np.ndarray],
    blur_kernel: int,
) -> np.ndarray:
    height, width = image_shape[:2]
    density = np.zeros((height, width), dtype=np.float32)

    for corners in all_corners:
        for u, v in corners.reshape(-1, 2):
            x = int(round(u))
            y = int(round(v))
            if 0 <= x < width and 0 <= y < height:
                density[y, x] += 1.0

    if blur_kernel % 2 == 0:
        blur_kernel += 1
    if blur_kernel > 1:
        density = cv2.GaussianBlur(density, (blur_kernel, blur_kernel), 0)

    if float(density.max()) <= 0.0:
        return np.zeros((height, width, 3), dtype=np.uint8)

    normalized = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.applyColorMap(normalized, cv2.COLORMAP_JET)


def process_image(
    image_path: Path,
    output_dir: Path,
    board_type: str,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    aruco_dict: str,
    subpix_window: int,
) -> tuple[bool, np.ndarray | None, tuple[int, int, int] | None]:
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"Failed to read image: {image_path}")
        return False, None, None

    charuco_ids = None
    if board_type == "checkerboard":
        found, corners = detect_checkerboard(image_bgr, rows, cols, subpix_window)
    else:
        found, corners, charuco_ids = detect_charuco(image_bgr, rows, cols, square_size, marker_size, aruco_dict)

    stem = image_path.stem
    if not found or corners is None:
        failure_path = output_dir / f"{stem}_not_found.txt"
        failure_path.write_text(f"{board_type} not detected.\n", encoding="ascii")
        print(f"No {board_type} board found in {image_path.name}")
        return False, None, image_bgr.shape, None

    if board_type == "checkerboard":
        annotated = annotate_uv_image(image_bgr, corners, rows, cols)
    else:
        annotated = annotate_charuco_uv_image(image_bgr, corners)
    annotated_path = output_dir / f"{stem}_uv.png"
    csv_path = output_dir / f"{stem}_uv.csv"
    cv2.imwrite(str(annotated_path), annotated)
    if board_type == "checkerboard":
        save_uv_points(csv_path, corners)
    else:
        if charuco_ids is None:
            raise RuntimeError("ChArUco detection succeeded but did not return corner IDs.")
        save_charuco_points(csv_path, corners, charuco_ids)
    print(f"Saved {annotated_path}")
    print(f"Saved {csv_path}")
    return True, corners, image_bgr.shape, charuco_ids


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = collect_images(input_path, args.extensions)
    if not image_paths:
        raise FileNotFoundError(f"No matching images found in: {input_path}")

    detections = 0
    all_corners = []
    heatmap_shape = None
    detection_records = []
    for image_path in image_paths:
        success, corners, image_shape, corner_ids = process_image(
            image_path=image_path,
            output_dir=output_dir,
            board_type=args.board_type,
            rows=args.rows,
            cols=args.cols,
            square_size=args.square_size,
            marker_size=args.marker_size,
            aruco_dict=args.aruco_dict,
            subpix_window=args.subpix_window,
        )
        if image_shape is not None and heatmap_shape is None:
            heatmap_shape = image_shape
        if success and corners is not None:
            detections += 1
            all_corners.append(corners)
            record = {
                "image_name": image_path.name,
                "image_path": str(image_path),
                "detected": True,
                "num_points": int(corners.reshape(-1, 2).shape[0]),
                "uv": corners.reshape(-1, 2).astype(float).tolist(),
            }
            if corner_ids is not None:
                record["corner_ids"] = np.asarray(corner_ids, dtype=int).reshape(-1).tolist()
            detection_records.append(record)
        else:
            detection_records.append(
                {
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "detected": False,
                    "num_points": 0,
                    "uv": [],
                    "corner_ids": [],
                }
            )

    if args.plot_heatmap and heatmap_shape is not None:
        heatmap = build_uv_heatmap(heatmap_shape, all_corners, args.heatmap_blur)
        heatmap_path = output_dir / f"{args.board_type}_uv_heatmap.png"
        cv2.imwrite(str(heatmap_path), heatmap)
        print(f"Saved {heatmap_path}")

    summary = {
        "board_type": args.board_type,
        "rows": args.rows,
        "cols": args.cols,
        "square_size": args.square_size,
        "marker_size": args.marker_size,
        "aruco_dict": args.aruco_dict,
        "num_images": len(image_paths),
        "num_detections": detections,
        "images": detection_records,
    }
    summary_path = output_dir / "detections.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="ascii")
    print(f"Saved {summary_path}")

    print(f"Detected {args.board_type} boards in {detections} / {len(image_paths)} images")


if __name__ == "__main__":
    main()
