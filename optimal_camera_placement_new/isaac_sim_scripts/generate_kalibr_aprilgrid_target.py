#!/usr/bin/env python3
"""Generate a Kalibr-style AprilGrid texture and target YAML."""

from __future__ import annotations

import argparse
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from calibration_board_utils import ensure_aprilgrid_texture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="Output PNG path.")
    parser.add_argument("--yaml", default=None, help="Optional Kalibr target YAML path.")
    parser.add_argument("--rows", type=int, required=True, help="Number of AprilTag rows.")
    parser.add_argument("--cols", type=int, required=True, help="Number of AprilTag columns.")
    parser.add_argument("--tag-size", type=float, required=True, help="Tag size in meters.")
    parser.add_argument(
        "--tag-spacing",
        type=float,
        required=True,
        help="Spacing ratio used by Kalibr: spacing_m / tag_size_m.",
    )
    parser.add_argument("--pixels-per-pitch", type=int, default=320)
    parser.add_argument("--black-border-bits", type=int, default=1)
    parser.add_argument("--outer-margin-pitches", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = pathlib.Path(args.output)
    ensure_aprilgrid_texture(
        output,
        rows=args.rows,
        cols=args.cols,
        tag_size=args.tag_size,
        tag_spacing=args.tag_spacing,
        pixels_per_tag_pitch=args.pixels_per_pitch,
        black_border_bits=args.black_border_bits,
        outer_margin_pitches=args.outer_margin_pitches,
    )
    print(f"Saved AprilGrid texture: {output}")

    if args.yaml:
        yaml_path = pathlib.Path(args.yaml)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        yaml_path.write_text(
            "target_type: 'aprilgrid'\n"
            f"tagCols: {args.cols}\n"
            f"tagRows: {args.rows}\n"
            f"tagSize: {args.tag_size:.12g}\n"
            f"tagSpacing: {args.tag_spacing:.12g}\n",
            encoding="ascii",
        )
        print(f"Saved Kalibr target YAML: {yaml_path}")


if __name__ == "__main__":
    main()
