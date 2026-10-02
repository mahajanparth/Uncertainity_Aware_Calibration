from __future__ import annotations

import cv2
import numpy as np


def filter_planar_correspondences_ransac(
    object_points_xyz: np.ndarray,
    image_points_xy: np.ndarray,
    reproj_threshold: float = 3.0,
    min_inliers: int = 8,
    confidence: float = 0.995,
    max_iters: int = 2000,
) -> dict:
    """Filter planar object/image correspondences with homography RANSAC.

    Args:
        object_points_xyz: Nx3 planar object points.
        image_points_xy: Nx2 image points.
    Returns:
        Dict with `mask`, `used`, and summary fields.
    """
    object_points_xyz = np.asarray(object_points_xyz, dtype=np.float32).reshape(-1, 3)
    image_points_xy = np.asarray(image_points_xy, dtype=np.float32).reshape(-1, 2)
    count = int(min(object_points_xyz.shape[0], image_points_xy.shape[0]))
    if count < 4:
        return {
            "used": False,
            "reason": "too_few_points",
            "num_input_points": count,
            "num_inliers": count,
            "mask": np.ones(count, dtype=bool).tolist(),
        }

    planar_xy = object_points_xyz[:count, :2]
    image_xy = image_points_xy[:count]
    homography, mask = cv2.findHomography(
        planar_xy,
        image_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(reproj_threshold),
        maxIters=int(max_iters),
        confidence=float(confidence),
    )
    if homography is None or mask is None:
        return {
            "used": False,
            "reason": "homography_failed",
            "num_input_points": count,
            "num_inliers": count,
            "mask": np.ones(count, dtype=bool).tolist(),
        }

    inlier_mask = mask.reshape(-1).astype(bool)
    num_inliers = int(np.count_nonzero(inlier_mask))
    if num_inliers < int(min_inliers):
        return {
            "used": False,
            "reason": "too_few_inliers",
            "num_input_points": count,
            "num_inliers": num_inliers,
            "mask": np.ones(count, dtype=bool).tolist(),
        }

    projected = cv2.perspectiveTransform(planar_xy.reshape(-1, 1, 2), homography).reshape(-1, 2)
    reproj_error = np.linalg.norm(projected - image_xy, axis=1)
    return {
        "used": True,
        "reason": "ransac_inliers",
        "num_input_points": count,
        "num_inliers": num_inliers,
        "num_outliers": int(count - num_inliers),
        "mask": inlier_mask.tolist(),
        "homography": homography.astype(float).tolist(),
        "mean_inlier_reproj_error": float(np.mean(reproj_error[inlier_mask])) if num_inliers > 0 else None,
        "max_inlier_reproj_error": float(np.max(reproj_error[inlier_mask])) if num_inliers > 0 else None,
    }
