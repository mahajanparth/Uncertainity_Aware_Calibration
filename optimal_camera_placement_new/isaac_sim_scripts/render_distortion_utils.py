from __future__ import annotations

import numpy as np


def augment_intrinsics_with_zero_distortion(intrinsics: np.ndarray) -> np.ndarray:
    intrinsics = np.asarray(intrinsics, dtype=float).reshape(-1)
    if intrinsics.size >= 9:
        return intrinsics[:9].copy()
    if intrinsics.size < 4:
        raise ValueError("Intrinsics vector must contain at least [fx, fy, cx, cy].")
    return np.concatenate([intrinsics[:4], np.zeros(5, dtype=float)])


def has_nonzero_distortion(intrinsics: np.ndarray, atol: float = 1e-12) -> bool:
    intrinsics = augment_intrinsics_with_zero_distortion(intrinsics)
    return bool(np.any(np.abs(intrinsics[4:9]) > atol))


def distort_rgb_image(rgb_image: np.ndarray, intrinsics: np.ndarray, iterations: int = 8) -> np.ndarray:
    intrinsics = augment_intrinsics_with_zero_distortion(intrinsics)
    if not has_nonzero_distortion(intrinsics):
        return rgb_image

    try:
        import cv2
    except ImportError:
        return rgb_image

    fx, fy, cx, cy, k1, k2, p1, p2, k3 = [float(v) for v in intrinsics[:9]]
    height, width = rgb_image.shape[:2]
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x_dist = (u - cx) / fx
    y_dist = (v - cy) / fy

    x = x_dist.copy()
    y = y_dist.copy()
    for _ in range(max(1, int(iterations))):
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        radial = np.where(np.abs(radial) < 1e-12, 1.0, radial)
        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (x_dist - delta_x) / radial
        y = (y_dist - delta_y) / radial

    map_x = (fx * x + cx).astype(np.float32)
    map_y = (fy * y + cy).astype(np.float32)
    return cv2.remap(
        rgb_image,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
