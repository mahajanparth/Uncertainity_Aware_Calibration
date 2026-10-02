from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


def _import_true_apriltag_detector():
    try:
        from pupil_apriltags import Detector  # type: ignore

        return "pupil_apriltags", Detector
    except ImportError:
        pass
    try:
        import apriltag  # type: ignore

        return "apriltag", apriltag
    except ImportError:
        pass
    try:
        from dt_apriltags import Detector  # type: ignore

        return "dt_apriltags", Detector
    except ImportError:
        pass
    return None, None


def get_aruco_dictionary(dictionary_name: str):
    if not hasattr(cv2, "aruco"):
        raise RuntimeError("This OpenCV build does not include cv2.aruco.")
    if not hasattr(cv2.aruco, dictionary_name):
        raise ValueError(f"Unknown ArUco dictionary: {dictionary_name}")
    dictionary_id = getattr(cv2.aruco, dictionary_name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dictionary_id)
    return cv2.aruco.Dictionary_get(dictionary_id)


def create_charuco_board(
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    aruco_dict_name: str = "DICT_4X4_250",
):
    aruco_dict = get_aruco_dictionary(aruco_dict_name)
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(cols, rows, square_size, marker_size, aruco_dict)
    if hasattr(cv2.aruco, "CharucoBoard"):
        return cv2.aruco.CharucoBoard((cols, rows), square_size, marker_size, aruco_dict)
    raise RuntimeError("This OpenCV build does not provide a ChArUco board constructor.")


def ensure_charuco_texture(
    texture_path: Path,
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    aruco_dict_name: str = "DICT_4X4_250",
    pixels_per_square: int = 200,
) -> Path:
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    image_size = (cols * pixels_per_square, rows * pixels_per_square)
    aruco_dict = get_aruco_dictionary(aruco_dict_name)

    if hasattr(cv2.aruco, "CharucoBoard_create") or hasattr(cv2.aruco, "CharucoBoard"):
        board = create_charuco_board(rows, cols, square_size, marker_size, aruco_dict_name)
        if hasattr(board, "generateImage"):
            board_image = board.generateImage(image_size)
        else:
            board_image = board.draw(image_size)
    else:
        board_image = _render_charuco_texture_manual(
            cols=cols,
            rows=rows,
            aruco_dict=aruco_dict,
            image_size=image_size,
            marker_ratio=marker_size / square_size,
        )

    cv2.imwrite(str(texture_path), board_image)
    return texture_path


def ensure_aprilgrid_texture(
    texture_path: Path,
    rows: int,
    cols: int,
    tag_size: float,
    tag_spacing: float,
    aruco_dict_name: str = "DICT_APRILTAG_36h11",
    pixels_per_tag_pitch: int = 320,
    black_border_bits: int = 1,
    outer_margin_pitches: float = 0.0,
) -> Path:
    texture_path.parent.mkdir(parents=True, exist_ok=True)
    aruco_dict = get_aruco_dictionary(aruco_dict_name)
    pitch = tag_size * (1.0 + tag_spacing)
    marker_ratio = tag_size / pitch
    cell_px = pixels_per_tag_pitch
    marker_px = max(32, int(round(cell_px * marker_ratio)))
    gap_px = max(0, cell_px - marker_px)
    margin_px = int(round(max(0.0, outer_margin_pitches) * cell_px))
    image = np.full((rows * marker_px + max(rows - 1, 0) * gap_px + 2 * margin_px,
                     cols * marker_px + max(cols - 1, 0) * gap_px + 2 * margin_px), 255, dtype=np.uint8)

    for row in range(rows):
        for col in range(cols):
            marker_id = row * cols + col
            marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
            if hasattr(aruco_dict, "generateImageMarker"):
                aruco_dict.generateImageMarker(marker_id, marker_px, marker, black_border_bits)
            else:
                cv2.aruco.drawMarker(aruco_dict, marker_id, marker_px, marker, black_border_bits)
            y0 = margin_px + row * (marker_px + gap_px)
            x0 = margin_px + col * (marker_px + gap_px)
            image[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker

    cv2.imwrite(str(texture_path), image)
    return texture_path


def detect_kalibr_aprilgrid(
    image_gray: np.ndarray,
    rows: int,
    cols: int,
    aruco_dict_name: str = "DICT_APRILTAG_36h11",
    min_tags_for_valid_obs: int = 4,
    min_border_distance: float = 4.0,
    do_subpix_refinement: bool = True,
    max_subpix_displacement2: float = 1.5,
    subpix_window_half_width: int = 2,
    subpix_max_iters: int = 30,
    subpix_epsilon: float = 0.1,
) -> dict:
    """Detect AprilGrid tags and arrange corners using Kalibr's grid indexing."""
    aruco_dict = get_aruco_dictionary(aruco_dict_name)
    if hasattr(cv2.aruco, "DetectorParameters_create"):
        detector_params = cv2.aruco.DetectorParameters_create()
    else:
        detector_params = cv2.aruco.DetectorParameters()
    corners, ids, _ = cv2.aruco.detectMarkers(image_gray, aruco_dict, parameters=detector_params)

    grid_cols = 2 * cols
    num_grid_corners = 4 * rows * cols
    image_points = np.full((num_grid_corners, 2), np.nan, dtype=np.float64)
    observed = np.zeros(num_grid_corners, dtype=bool)
    detections = []
    if ids is None:
        return {
            "success": False,
            "num_markers": 0,
            "detections": detections,
            "image_points": image_points,
            "observed": observed,
        }

    seen_ids = set()
    for tag_corners, tag_id_value in zip(corners, ids.reshape(-1)):
        tag_id = int(tag_id_value)
        pts_raw = np.asarray(tag_corners, dtype=np.float32).reshape(4, 2)
        if tag_id in seen_ids or tag_id < 0 or tag_id >= rows * cols:
            continue
        if (
            (pts_raw[:, 0] < min_border_distance).any()
            or (pts_raw[:, 0] > image_gray.shape[1] - min_border_distance).any()
            or (pts_raw[:, 1] < min_border_distance).any()
            or (pts_raw[:, 1] > image_gray.shape[0] - min_border_distance).any()
        ):
            continue

        pts = pts_raw.copy()
        active = [True, True, True, True]
        if do_subpix_refinement:
            refined = pts.reshape(-1, 1, 2).copy()
            cv2.cornerSubPix(
                image_gray,
                refined,
                (int(subpix_window_half_width), int(subpix_window_half_width)),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    int(subpix_max_iters),
                    float(subpix_epsilon),
                ),
            )
            pts = refined.reshape(4, 2)
            displacement2 = np.sum((pts - pts_raw) ** 2, axis=1)
            active = [bool(value <= max_subpix_displacement2) for value in displacement2]

        base_id = (tag_id // cols) * grid_cols * 2 + (tag_id % cols) * 2
        point_indices = [base_id, base_id + 1, base_id + grid_cols + 1, base_id + grid_cols]
        for local_idx, point_idx in enumerate(point_indices):
            image_points[point_idx] = pts[local_idx]
            observed[point_idx] = active[local_idx]
        detections.append(
            {
                "id": tag_id,
                "corners": pts.astype(np.float64),
                "point_indices": point_indices,
                "active": active,
            }
        )
        seen_ids.add(tag_id)

    detections.sort(key=lambda item: item["id"])
    return {
        "success": len(detections) >= min_tags_for_valid_obs,
        "num_markers": len(detections),
        "detections": detections,
        "image_points": image_points,
        "observed": observed,
    }


def detect_true_apriltag_grid(
    image_gray: np.ndarray,
    rows: int,
    cols: int,
    min_tags_for_valid_obs: int = 4,
    min_border_distance: float = 4.0,
    do_subpix_refinement: bool = True,
    max_subpix_displacement2: float = 9.0,
    subpix_window_half_width: int = 5,
    subpix_max_iters: int = 80,
    subpix_epsilon: float = 0.01,
) -> dict:
    """Detect AprilGrid with a true AprilTag library if installed locally."""
    backend_name, detector_impl = _import_true_apriltag_detector()
    grid_cols = 2 * cols
    num_grid_corners = 4 * rows * cols
    image_points = np.full((num_grid_corners, 2), np.nan, dtype=np.float64)
    observed = np.zeros(num_grid_corners, dtype=bool)
    detections = []

    if detector_impl is None:
        return {
            "success": False,
            "num_markers": 0,
            "detections": detections,
            "image_points": image_points,
            "observed": observed,
            "note": "true_apriltag_detector_unavailable",
        }

    if backend_name == "pupil_apriltags":
        detector = detector_impl(families="tag36h11")
        raw_detections = detector.detect(image_gray)
    elif backend_name == "dt_apriltags":
        detector = detector_impl(families="tag36h11")
        raw_detections = detector.detect(image_gray)
    else:
        options = detector_impl.DetectorOptions(families="tag36h11")
        detector = detector_impl.Detector(options)
        raw_detections = detector.detect(image_gray)

    seen_ids = set()
    for det in raw_detections:
        tag_id = int(getattr(det, "tag_id", getattr(det, "tag_id", -1)))
        if tag_id in seen_ids or tag_id < 0 or tag_id >= rows * cols:
            continue
        corners = getattr(det, "corners", None)
        if corners is None:
            continue
        pts_raw = _order_quad_points(np.asarray(corners, dtype=np.float32).reshape(4, 2))
        if (
            (pts_raw[:, 0] < min_border_distance).any()
            or (pts_raw[:, 0] > image_gray.shape[1] - min_border_distance).any()
            or (pts_raw[:, 1] < min_border_distance).any()
            or (pts_raw[:, 1] > image_gray.shape[0] - min_border_distance).any()
        ):
            continue

        pts = pts_raw.copy()
        active = [True, True, True, True]
        if do_subpix_refinement:
            refined = pts.reshape(-1, 1, 2).copy()
            cv2.cornerSubPix(
                image_gray,
                refined,
                (int(subpix_window_half_width), int(subpix_window_half_width)),
                (-1, -1),
                (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    int(subpix_max_iters),
                    float(subpix_epsilon),
                ),
            )
            pts = refined.reshape(4, 2)
            displacement2 = np.sum((pts - pts_raw) ** 2, axis=1)
            active = [bool(value <= max_subpix_displacement2) for value in displacement2]

        base_id = (tag_id // cols) * grid_cols * 2 + (tag_id % cols) * 2
        point_indices = [base_id, base_id + 1, base_id + grid_cols + 1, base_id + grid_cols]
        for local_idx, point_idx in enumerate(point_indices):
            image_points[point_idx] = pts[local_idx]
            observed[point_idx] = active[local_idx]
        detections.append(
            {
                "id": tag_id,
                "corners": pts.astype(np.float64),
                "point_indices": point_indices,
                "active": active,
            }
        )
        seen_ids.add(tag_id)

    detections.sort(key=lambda item: item["id"])
    return {
        "success": len(detections) >= min_tags_for_valid_obs,
        "num_markers": len(detections),
        "detections": detections,
        "image_points": image_points,
        "observed": observed,
        "note": f"true_apriltag_detected_{backend_name}",
        "detector_backend": backend_name,
    }


def _order_quad_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    ordered = pts[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    # Angle sorting in image coordinates gives either TL, TR, BR, BL or
    # TL, BL, BR, TR. Keep the clockwise order expected by OpenCV/Kalibr.
    if ordered[1, 1] > ordered[3, 1]:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered.astype(np.float32)


def detect_geometric_aprilgrid(
    image_gray: np.ndarray,
    rows: int,
    cols: int,
    tag_size: float,
    tag_spacing: float,
    board_width: float | None = None,
    board_height: float | None = None,
    min_border_distance: float = 4.0,
    refine_corners: bool = True,
    subpix_window_half_width: int = 5,
    subpix_max_iters: int = 30,
    subpix_epsilon: float = 0.03,
) -> dict:
    """Detect a rendered AprilGrid by fitting a homography to the tag-grid boundary.

    This is a local Kalibr-style AprilGrid detector for synthetic/rendered targets.
    It does not decode AprilTag IDs. Instead it uses the known ordered grid layout,
    estimates the visible grid quadrilateral, and returns the same 2x2-per-tag
    corner indexing Kalibr uses for aprilgrid targets.
    """
    if image_gray.ndim != 2:
        raise ValueError("detect_geometric_aprilgrid expects a grayscale image.")

    grid_cols = 2 * cols
    num_grid_corners = 4 * rows * cols
    image_points = np.full((num_grid_corners, 2), np.nan, dtype=np.float64)
    observed = np.zeros(num_grid_corners, dtype=bool)
    detections = []

    blur = cv2.GaussianBlur(image_gray, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {
            "success": False,
            "num_markers": 0,
            "detections": detections,
            "image_points": image_points,
            "observed": observed,
            "note": "geometric_aprilgrid_no_contours",
        }

    quad_candidates = []
    max_area = max(cv2.contourArea(contour) for contour in contours)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(300.0, 0.25 * max_area):
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        approx = cv2.approxPolyDP(hull, 0.03 * perimeter, True)
        if len(approx) != 4:
            continue
        quad = _order_quad_points(approx.reshape(4, 2))
        width_a = np.linalg.norm(quad[1] - quad[0])
        width_b = np.linalg.norm(quad[2] - quad[3])
        height_a = np.linalg.norm(quad[3] - quad[0])
        height_b = np.linalg.norm(quad[2] - quad[1])
        width = max((width_a + width_b) * 0.5, 1.0)
        height = max((height_a + height_b) * 0.5, 1.0)
        aspect = width / height
        if aspect < 0.45 or aspect > 2.2:
            continue
        quad_candidates.append({"area": area, "quad": quad, "center": quad.mean(axis=0)})

    if len(quad_candidates) >= 4:
        # Rendered Isaac images usually expose each tag's black square as its
        # own quadrilateral. Assign those quads into the known Kalibr grid by
        # image-space row/column order. This deliberately avoids AprilTag ID
        # decoding, which differs between Kalibr PDFs and OpenCV's dictionary.
        centers = np.asarray([item["center"] for item in quad_candidates], dtype=np.float32)
        row_count = min(rows, max(1, int(round(len(quad_candidates) / max(cols, 1)))))
        if row_count > 1:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.1)
            _compactness, labels, row_centers = cv2.kmeans(
                centers[:, 1].reshape(-1, 1),
                row_count,
                None,
                criteria,
                5,
                cv2.KMEANS_PP_CENTERS,
            )
            row_order = np.argsort(row_centers.reshape(-1))
            row_map = {int(label): int(rank) for rank, label in enumerate(row_order)}
            rows_of_candidates: list[list[dict]] = [[] for _ in range(row_count)]
            for item, label in zip(quad_candidates, labels.reshape(-1)):
                rows_of_candidates[row_map[int(label)]].append(item)
        else:
            rows_of_candidates = [quad_candidates]

        for row_idx, row_items in enumerate(rows_of_candidates):
            row_items.sort(key=lambda item: float(item["center"][0]))
            if not row_items:
                continue
            x_values = np.asarray([item["center"][0] for item in row_items], dtype=np.float64)
            x_min = float(x_values.min())
            x_max = float(x_values.max())
            used_cols = set()
            for order, item in enumerate(row_items):
                if len(row_items) == cols:
                    col_idx = order
                elif x_max > x_min:
                    col_idx = int(round((float(item["center"][0]) - x_min) / (x_max - x_min) * (cols - 1)))
                else:
                    col_idx = order
                while col_idx in used_cols and col_idx + 1 < cols:
                    col_idx += 1
                if row_idx < 0 or row_idx >= rows or col_idx < 0 or col_idx >= cols or col_idx in used_cols:
                    continue
                used_cols.add(col_idx)
                tag_id = row_idx * cols + col_idx
                base_id = row_idx * grid_cols * 2 + col_idx * 2
                point_indices = [base_id, base_id + 1, base_id + grid_cols + 1, base_id + grid_cols]
                quad = item["quad"].astype(np.float64)
                active = (
                    np.isfinite(quad).all(axis=1)
                    & (quad[:, 0] >= min_border_distance)
                    & (quad[:, 0] < image_gray.shape[1] - min_border_distance)
                    & (quad[:, 1] >= min_border_distance)
                    & (quad[:, 1] < image_gray.shape[0] - min_border_distance)
                )
                for local_idx, point_idx in enumerate(point_indices):
                    image_points[point_idx] = quad[local_idx]
                    observed[point_idx] = bool(active[local_idx])
                detections.append(
                    {
                        "id": tag_id,
                        "corners": quad,
                        "point_indices": point_indices,
                        "active": [bool(value) for value in active],
                    }
                )

        detections.sort(key=lambda item: item["id"])
        if detections:
            return {
                "success": bool(np.count_nonzero(observed) >= 4 * min(len(detections), 4)),
                "num_markers": int(sum(all(det["active"]) for det in detections)),
                "detections": detections,
                "image_points": image_points,
                "observed": observed,
                "note": "geometric_aprilgrid_quad_detected",
            }

    min_area = 0.02 * float(image_gray.shape[0] * image_gray.shape[1])
    large_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_area]
    if not large_contours:
        large_contours = [max(contours, key=cv2.contourArea)]
    contour = max(large_contours, key=cv2.contourArea)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)
    approx = cv2.approxPolyDP(hull, 0.03 * perimeter, True)
    if len(approx) == 4:
        quad = approx.reshape(4, 2).astype(np.float32)
    else:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect).astype(np.float32)
    quad = _order_quad_points(quad)

    pitch = tag_size * (1.0 + tag_spacing)
    gap = pitch - tag_size
    grid_width = cols * tag_size + max(cols - 1, 0) * gap
    grid_height = rows * tag_size + max(rows - 1, 0) * gap
    x_margin = 0.0 if board_width is None else max((float(board_width) - grid_width) / 2.0, 0.0)
    y_margin = 0.0 if board_height is None else max((float(board_height) - grid_height) / 2.0, 0.0)
    src = np.array(
        [
            [x_margin, y_margin],
            [x_margin + grid_width, y_margin],
            [x_margin + grid_width, y_margin + grid_height],
            [x_margin, y_margin + grid_height],
        ],
        dtype=np.float32,
    )
    homography = cv2.getPerspectiveTransform(src, quad)
    local_points = build_aprilgrid_corner_points_local(
        rows=rows,
        cols=cols,
        tag_size=tag_size,
        tag_spacing=tag_spacing,
        board_width=board_width,
        board_height=board_height,
    )[:, :2].astype(np.float32)
    projected = cv2.perspectiveTransform(local_points.reshape(-1, 1, 2), homography).reshape(-1, 2)

    active = (
        np.isfinite(projected).all(axis=1)
        & (projected[:, 0] >= min_border_distance)
        & (projected[:, 0] < image_gray.shape[1] - min_border_distance)
        & (projected[:, 1] >= min_border_distance)
        & (projected[:, 1] < image_gray.shape[0] - min_border_distance)
    )
    if refine_corners and np.any(active):
        refined = projected[active].reshape(-1, 1, 2).astype(np.float32)
        cv2.cornerSubPix(
            image_gray,
            refined,
            (int(subpix_window_half_width), int(subpix_window_half_width)),
            (-1, -1),
            (
                cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                int(subpix_max_iters),
                float(subpix_epsilon),
            ),
        )
        projected[active] = refined.reshape(-1, 2)

    image_points[:] = projected.astype(np.float64)
    observed[:] = active
    for tag_id in range(rows * cols):
        base_id = (tag_id // cols) * grid_cols * 2 + (tag_id % cols) * 2
        point_indices = [base_id, base_id + 1, base_id + grid_cols + 1, base_id + grid_cols]
        detections.append(
            {
                "id": tag_id,
                "corners": image_points[point_indices].astype(np.float64),
                "point_indices": point_indices,
                "active": [bool(observed[idx]) for idx in point_indices],
            }
        )

    return {
        "success": bool(np.count_nonzero(observed) >= 4 * min(rows * cols, 4)),
        "num_markers": int(sum(all(det["active"]) for det in detections)),
        "detections": detections,
        "image_points": image_points,
        "observed": observed,
        "note": "geometric_aprilgrid_detected",
        "grid_quad": quad.astype(float).tolist(),
    }


def ensure_pdf_texture(
    pdf_path: Path,
    texture_path: Path,
    dpi: int = 200,
    white_threshold: int = 250,
) -> Path:
    pdf_path = pdf_path.expanduser().resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"AprilGrid PDF does not exist: {pdf_path}")
    texture_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_prefix = Path(tmp_dir) / "pdf_texture"
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", str(dpi), str(pdf_path), str(output_prefix)],
            check=True,
        )
        rendered_path = output_prefix.with_suffix(".png")
        image = cv2.imread(str(rendered_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise RuntimeError(f"Failed to rasterize AprilGrid PDF: {pdf_path}")

    ys, xs = np.where(image < white_threshold)
    if xs.size == 0 or ys.size == 0:
        raise RuntimeError(f"No non-white AprilGrid content found in PDF: {pdf_path}")

    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    width = x_max - x_min
    height = y_max - y_min
    side = max(width, height)
    cx = (x_min + x_max) // 2
    cy = (y_min + y_max) // 2
    half = side // 2
    x0 = max(0, cx - half)
    y0 = max(0, cy - half)
    x1 = min(image.shape[1], x0 + side)
    y1 = min(image.shape[0], y0 + side)
    crop = image[y0:y1, x0:x1]

    cv2.imwrite(str(texture_path), crop)
    return texture_path


def ensure_kalibr_aprilgrid_pdf(
    pdf_path: Path,
    rows: int,
    cols: int,
    tag_size: float,
    tag_spacing: float,
    kalibr_create_target_cmd: str = "kalibr_create_target_pdf",
) -> Path:
    """Generate an AprilGrid PDF using Kalibr's target generator."""
    pdf_path = pdf_path.expanduser().resolve()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        cmd = [
            kalibr_create_target_cmd,
            "--type",
            "apriltag",
            "--nx",
            str(int(cols)),
            "--ny",
            str(int(rows)),
            "--tsize",
            f"{float(tag_size):.12g}",
            "--tspace",
            f"{float(tag_spacing):.12g}",
        ]
        try:
            subprocess.run(cmd, check=True, cwd=str(tmp_path))
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Kalibr target generator not found: '{kalibr_create_target_cmd}'. "
                "Install Kalibr or pass --kalibr-create-target-cmd with the executable path."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"Kalibr target generation failed with command: {' '.join(cmd)}") from exc

        generated_pdfs = sorted(tmp_path.glob("*.pdf"))
        if not generated_pdfs:
            raise RuntimeError("Kalibr target generator did not produce a PDF.")
        latest_pdf = max(generated_pdfs, key=lambda p: p.stat().st_mtime)
        pdf_path.write_bytes(latest_pdf.read_bytes())
    return pdf_path


def build_aprilgrid_corner_points_local(
    rows: int,
    cols: int,
    tag_size: float,
    tag_spacing: float,
    board_width: float | None = None,
    board_height: float | None = None,
) -> np.ndarray:
    pitch = tag_size * (1.0 + tag_spacing)
    gap = pitch - tag_size
    if board_width is None:
        x_margin = 0.0
    else:
        grid_width = cols * tag_size + max(cols - 1, 0) * gap
        if board_width < grid_width:
            raise ValueError(f"AprilGrid board_width {board_width} is smaller than tag grid width {grid_width}.")
        x_margin = (board_width - grid_width) / 2.0
    if board_height is None:
        y_margin = 0.0
    else:
        grid_height = rows * tag_size + max(rows - 1, 0) * gap
        if board_height < grid_height:
            raise ValueError(f"AprilGrid board_height {board_height} is smaller than tag grid height {grid_height}.")
        y_margin = (board_height - grid_height) / 2.0
    points = []
    for grid_row in range(2 * rows):
        for grid_col in range(2 * cols):
            x = (grid_col // 2) * pitch + (grid_col % 2) * tag_size + x_margin
            y = (grid_row // 2) * pitch + (grid_row % 2) * tag_size + y_margin
            points.append((x, y, 0.0))
    return np.asarray(points, dtype=np.float64)


def build_charuco_corner_points_local(
    rows: int,
    cols: int,
    square_size: float,
    marker_size: float,
    aruco_dict_name: str = "DICT_4X4_250",
) -> np.ndarray:
    if hasattr(cv2.aruco, "CharucoBoard_create") or hasattr(cv2.aruco, "CharucoBoard"):
        board = create_charuco_board(rows, cols, square_size, marker_size, aruco_dict_name)
        if hasattr(board, "getChessboardCorners"):
            return np.asarray(board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
        return np.asarray(board.chessboardCorners, dtype=np.float64).reshape(-1, 3)

    # Fallback for OpenCV builds that can render/detect ArUco markers but do not
    # expose CharucoBoard_create. The ChArUco chessboard corners lie on the
    # inner checkerboard grid at integer multiples of square_size.
    inner_cols = max(cols - 1, 0)
    inner_rows = max(rows - 1, 0)
    xs = np.arange(inner_cols, dtype=np.float64) + 1.0
    ys = np.arange(inner_rows, dtype=np.float64) + 1.0
    grid_x, grid_y = np.meshgrid(xs, ys)
    return np.stack(
        [
            grid_x.reshape(-1) * square_size,
            grid_y.reshape(-1) * square_size,
            np.zeros(grid_x.size, dtype=np.float64),
        ],
        axis=1,
    )


def _render_charuco_texture_manual(
    cols: int,
    rows: int,
    aruco_dict,
    image_size: tuple[int, int],
    marker_ratio: float,
) -> np.ndarray:
    width_px, height_px = image_size
    square_px = min(width_px // cols, height_px // rows)
    board_width = cols * square_px
    board_height = rows * square_px
    image = np.full((board_height, board_width), 255, dtype=np.uint8)

    marker_px = max(8, int(round(square_px * marker_ratio)))
    marker_id = 0
    for row in range(rows):
        for col in range(cols):
            if (row + col) % 2 == 0:
                y0 = row * square_px
                x0 = col * square_px
                image[y0 : y0 + square_px, x0 : x0 + square_px] = 0
                continue

            marker = np.zeros((marker_px, marker_px), dtype=np.uint8)
            aruco_dict.generateImageMarker(marker_id, marker_px, marker, 1)
            marker_id += 1

            y0 = row * square_px + (square_px - marker_px) // 2
            x0 = col * square_px + (square_px - marker_px) // 2
            image[y0 : y0 + marker_px, x0 : x0 + marker_px] = marker

    return image


def create_textured_board_mesh(
    stage,
    board_path: str,
    texture_path: Path,
    width: float,
    height: float,
    center: np.ndarray,
    z_offset: float = 0.0038,
) -> None:
    from pxr import Sdf, UsdGeom, UsdShade

    mesh = UsdGeom.Mesh.Define(stage, board_path)
    half_w = width / 2.0
    half_h = height / 2.0
    z = float(center[2] + z_offset)
    points = [
        (-half_w, -half_h, 0.0),
        (half_w, -half_h, 0.0),
        (half_w, half_h, 0.0),
        (-half_w, half_h, 0.0),
    ]
    points = [(p[0] + center[0], p[1] + center[1], p[2] + z) for p in points]
    mesh.CreatePointsAttr(points)
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateNormalsAttr([(0.0, 0.0, 1.0)])
    mesh.SetNormalsInterpolation("constant")
    primvars_api = UsdGeom.PrimvarsAPI(mesh)
    st = primvars_api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.varying)
    st.Set([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])

    material_path = f"{board_path}_Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    tex_shader = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    tex_shader.CreateIdAttr("UsdUVTexture")
    tex_shader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(texture_path)))
    tex_shader.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    tex_shader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

    st_reader = UsdShade.Shader.Define(stage, f"{material_path}/PrimvarReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    st_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

    tex_shader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(st_reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex_shader.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh).Bind(material)
