#!/usr/bin/env python3
"""Export selected images from an interactive summary JSON to a ROS1 bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True, help="Path to interactive_summary.json.")
    parser.add_argument("--bag", required=True, help="Output ROS1 bag path.")
    parser.add_argument("--topics", nargs="+", default=["/cam0/image_raw"], help="ROS image topics to write.")
    parser.add_argument("--fps", type=float, default=10.0, help="Timestamp spacing for exported images.")
    parser.add_argument("--frame-id", default="cam0", help="ROS header frame_id.")
    parser.add_argument(
        "--encoding",
        choices=["mono8", "rgb8"],
        default="mono8",
        help="ROS Image encoding. Kalibr is most reliable with mono8.",
    )
    return parser.parse_args()


def collect_image_paths(summary: dict) -> list[Path]:
    image_paths: list[Path] = []
    for entry in summary.get("history", []):
        if entry.get("step_type") == "seed":
            image_path = entry.get("image_path")
            if image_path:
                image_paths.append(Path(image_path))
            continue
        if entry.get("step_type") == "recommended_batch":
            for capture in entry.get("captures", []):
                image_path = capture.get("image_path")
                if image_path:
                    image_paths.append(Path(image_path))
    return image_paths


def resolve_image_paths(image_paths: list[Path], summary_path: Path) -> list[Path]:
    summary_root = summary_path.parent
    repo_root = summary_root.parents[1] if len(summary_root.parents) > 1 else summary_root
    resolved = []
    for image_path in image_paths:
        if image_path.exists():
            resolved.append(image_path)
            continue

        parts = image_path.parts
        if "isaac_outputs" in parts:
            rel = Path(*parts[parts.index("isaac_outputs") :])
            candidate = repo_root / rel
            if candidate.exists():
                resolved.append(candidate)
                continue

        candidate = summary_root / image_path.name
        if candidate.exists():
            resolved.append(candidate)
            continue

        resolved.append(image_path)
    return resolved


def load_image(image_path: Path, encoding: str) -> np.ndarray:
    try:
        from PIL import Image

        mode = "L" if encoding == "mono8" else "RGB"
        return np.asarray(Image.open(image_path).convert(mode), dtype=np.uint8)
    except ImportError:
        import cv2

        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        if encoding == "mono8":
            return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def main() -> None:
    args = parse_args()

    import rosbag
    import rospy
    from sensor_msgs.msg import Image
    from std_msgs.msg import Header

    summary_path = Path(args.summary).expanduser().resolve()
    bag_path = Path(args.bag).expanduser().resolve()
    summary = json.loads(summary_path.read_text(encoding="ascii"))
    image_paths = resolve_image_paths(collect_image_paths(summary), summary_path)
    if not image_paths:
        raise RuntimeError(f"No image paths found in {summary_path}")

    missing = [path for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing image file: {missing[0]}")

    bag_path.parent.mkdir(parents=True, exist_ok=True)
    fps = args.fps if args.fps > 0.0 else 10.0
    with rosbag.Bag(str(bag_path), "w") as bag:
        for seq, image_path in enumerate(image_paths):
            image = load_image(image_path, args.encoding)
            stamp = rospy.Time.from_sec(seq / fps)
            msg = Image()
            msg.header = Header(seq=seq, stamp=stamp, frame_id=args.frame_id)
            msg.height = int(image.shape[0])
            msg.width = int(image.shape[1])
            msg.encoding = args.encoding
            msg.is_bigendian = 0
            msg.step = int(image.shape[1] if args.encoding == "mono8" else image.shape[1] * 3)
            msg.data = image.tobytes()
            for topic in args.topics:
                bag.write(topic, msg, stamp)

    print(f"Wrote {len(image_paths)} images to {bag_path}")
    print(f"Topics: {', '.join(args.topics)}")


if __name__ == "__main__":
    main()
