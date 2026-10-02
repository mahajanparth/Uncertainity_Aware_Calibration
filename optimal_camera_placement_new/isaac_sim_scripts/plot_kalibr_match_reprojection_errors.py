#!/usr/bin/env python3
"""Plot per-point reprojection errors for Kalibr-exported 2D-3D matches."""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import cv2
import numpy as np

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from OASIS import FIM as fim


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matches", required=True, help="Path to kalibr_matches.json")
    parser.add_argument("--output-dir", required=True, help="Directory for overlay images and report JSON.")
    parser.add_argument(
        "--intrinsics-source",
        choices=("gt", "kalibr-camchain", "oasis-report"),
        default="kalibr-camchain",
        help="Which intrinsics to use for reprojection.",
    )
    parser.add_argument("--summary", default=None, help="interactive_summary.json, required for --intrinsics-source gt.")
    parser.add_argument("--kalibr-camchain", default=None, help="cam_april-camchain.yaml, required for --intrinsics-source kalibr-camchain.")
    parser.add_argument(
        "--oasis-report",
        default=None,
        help="OASIS intrinsics report JSON, required for --intrinsics-source oasis-report.",
    )
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--label-threshold", type=float, default=0.5)
    return parser.parse_args()


def load_json(path: pathlib.Path) -> dict:
    return json.loads(path.read_text(encoding="ascii"))


def load_kalibr_camchain(path: pathlib.Path) -> np.ndarray:
    intrinsics = None
    distortion = None
    for line in path.read_text(encoding="ascii").splitlines():
        stripped = line.strip()
        if stripped.startswith("intrinsics:"):
            intrinsics = json.loads(stripped.split(":", 1)[1].strip().replace("'", '"'))
        elif stripped.startswith("distortion_coeffs:"):
            distortion = json.loads(stripped.split(":", 1)[1].strip().replace("'", '"'))
    if intrinsics is None or distortion is None:
        raise RuntimeError(f"Failed to parse intrinsics/distortion from {path}")
    full = list(intrinsics) + list(distortion)
    while len(full) < 9:
        full.append(0.0)
    return np.asarray(full[:9], dtype=float)


def resolve_intrinsics(args: argparse.Namespace) -> tuple[np.ndarray, str]:
    if args.intrinsics_source == "gt":
        if not args.summary:
            raise RuntimeError("--summary is required for --intrinsics-source gt")
        summary = load_json(pathlib.Path(args.summary).expanduser().resolve())
        return np.asarray(summary["intrinsics_gt"], dtype=float), "interactive_summary.intrinsics_gt"
    if args.intrinsics_source == "oasis-report":
        if not args.oasis_report:
            raise RuntimeError("--oasis-report is required for --intrinsics-source oasis-report")
        report = load_json(pathlib.Path(args.oasis_report).expanduser().resolve())
        if "final_intrinsics_report" in report:
            intr = report["final_intrinsics_report"]["intrinsics_vector"]
        elif "estimate" in report:
            intr = report["estimate"]["intrinsics_vector"]
        else:
            intr = report["intrinsics_vector"]
        return np.asarray(intr, dtype=float), str(pathlib.Path(args.oasis_report).expanduser().resolve())
    if not args.kalibr_camchain:
        raise RuntimeError("--kalibr-camchain is required for --intrinsics-source kalibr-camchain")
    path = pathlib.Path(args.kalibr_camchain).expanduser().resolve()
    return load_kalibr_camchain(path), str(path)


def camera_from_intrinsics(intrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    intrinsics = np.asarray(intrinsics, dtype=float).reshape(-1)
    camera_matrix = np.array(
        [[intrinsics[0], 0.0, intrinsics[2]], [0.0, intrinsics[1], intrinsics[3]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((5, 1), dtype=np.float64)
    if intrinsics.size > 4:
        tail = intrinsics[4 : min(intrinsics.size, 9)]
        dist[: tail.size, 0] = tail
    return camera_matrix, dist


def error_color(error_px: float, max_error_px: float) -> tuple[int, int, int]:
    ratio = float(np.clip(error_px / max(max_error_px, 1.0e-9), 0.0, 1.0))
    blue = int(round(255.0 * (1.0 - ratio)))
    red = int(round(255.0 * ratio))
    green = int(round(255.0 * (1.0 - abs(ratio - 0.5) * 2.0)))
    return blue, green, red


def draw_overlay(image_bgr: np.ndarray, observed_uv: np.ndarray, projected_uv: np.ndarray, errors_px: np.ndarray, label_threshold: float) -> np.ndarray:
    annotated = image_bgr.copy()
    max_error = float(np.max(errors_px)) if errors_px.size else 1.0
    for obs, proj, err in zip(observed_uv, projected_uv, errors_px):
        obs_pt = tuple(int(round(v)) for v in obs)
        proj_pt = tuple(int(round(v)) for v in proj)
        color = error_color(float(err), max_error)
        cv2.line(annotated, obs_pt, proj_pt, color, 1, cv2.LINE_AA)
        cv2.circle(annotated, obs_pt, 3, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(annotated, proj_pt, 3, (0, 255, 0), 1, cv2.LINE_AA)
        if float(err) >= float(label_threshold):
            cv2.putText(
                annotated,
                f"{float(err):.2f}",
                (obs_pt[0] + 4, obs_pt[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )
    return annotated


def main() -> None:
    args = parse_args()
    matches_path = pathlib.Path(args.matches).expanduser().resolve()
    output_dir = pathlib.Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = load_json(matches_path)
    images = list(payload.get("images", []))
    if args.max_images is not None:
        images = images[: max(0, int(args.max_images))]

    intrinsics, intrinsics_source_path = resolve_intrinsics(args)
    camera_matrix, dist_coeffs = camera_from_intrinsics(intrinsics)

    report_rows = []
    for image_record in images:
        if not image_record.get("detected", False):
            report_rows.append(
                {
                    "image_path": image_record.get("image_path"),
                    "used": False,
                    "reason": "kalibr_detection_missing",
                }
            )
            continue

        image_path = pathlib.Path(image_record["image_path"])
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            report_rows.append({"image_path": str(image_path), "used": False, "reason": "image_not_readable"})
            continue

        object_points = np.asarray(image_record["xyz_target"], dtype=np.float64).reshape(-1, 3)
        image_points = np.asarray(image_record["uv"], dtype=np.float64).reshape(-1, 2)
        if object_points.shape[0] < 4:
            report_rows.append(
                {
                    "image_path": str(image_path),
                    "used": False,
                    "reason": "too_few_points",
                    "num_points": int(object_points.shape[0]),
                }
            )
            continue

        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            report_rows.append({"image_path": str(image_path), "used": False, "reason": "solvepnp_failed"})
            continue

        projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
        projected = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        errors = np.linalg.norm(projected - image_points, axis=1)
        annotated = draw_overlay(image_bgr, image_points, projected, errors, args.label_threshold)

        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        rms_error = float(math.sqrt(np.mean(errors * errors)))
        cv2.putText(
            annotated,
            f"mean={mean_error:.3f}px rms={rms_error:.3f}px max={max_error:.3f}px",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), annotated)
        report_rows.append(
            {
                "image_path": str(image_path),
                "overlay_path": str(output_path),
                "used": True,
                "num_points": int(object_points.shape[0]),
                "mean_error_px": mean_error,
                "rms_error_px": rms_error,
                "max_error_px": max_error,
            }
        )

    summary_report = {
        "matches_path": str(matches_path),
        "output_dir": str(output_dir),
        "intrinsics_source": args.intrinsics_source,
        "intrinsics_source_path": intrinsics_source_path,
        "intrinsics_vector": intrinsics.tolist(),
        "images": report_rows,
    }
    (output_dir / "reprojection_overlay_report.json").write_text(json.dumps(summary_report, indent=2), encoding="ascii")
    num_used = sum(1 for row in report_rows if row.get("used"))
    print(f"Saved Kalibr reprojection overlays for {num_used} images to: {output_dir}")


if __name__ == "__main__":
    main()
