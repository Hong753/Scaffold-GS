#!/usr/bin/env python3
"""Check camera geometry using fresh image matches, without training or CUDA."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source_path", help="COLMAP dataset containing images/ and sparse[/0]/")
    parser.add_argument("-m", "--model_path", help="Read dataset location and image folder from a saved run")
    parser.add_argument("--images", help="Override image folder; absolute or relative to the dataset")
    parser.add_argument("-o", "--output", type=Path, default=Path("outputs/pose_geometry_check"))
    parser.add_argument("--samples", type=int, default=48, help="Sequence-wide anchors for local matching")
    parser.add_argument("--local_gaps", type=int, nargs="+", default=[1, 5, 15], help="Gaps in the complete registered camera sequence")
    parser.add_argument("--revisit_pairs", type=int, default=96, help="Budget of distant candidate pairs")
    parser.add_argument("--revisit_gap", type=int, default=300, help="Minimum sequence gap for a revisit candidate")
    parser.add_argument("--max_offset", type=int, default=5, help="Sweep both image poses by -N through +N sequence entries")
    parser.add_argument("--max_width", type=int, default=1600, help="Maximum image width for analysis; errors use these pixels")
    parser.add_argument("--max_features", type=int, default=4000)
    parser.add_argument("--min_matches", type=int, default=24)
    parser.add_argument("--ratio", type=float, default=.65, help="Mutual SIFT descriptor ratio threshold")
    parser.add_argument("--ransac_px", type=float, default=1.5)
    parser.add_argument("--previews", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not (args.source_path or args.model_path):
        parser.error("Supply -s DATASET or -m TRAINING_RUN")
    if min(args.samples, args.max_width, args.max_features, args.min_matches, args.threads) < 1:
        parser.error("Sampling, image, feature and thread limits must be positive")
    if args.min_matches < 16 or not 0 < args.ratio < 1 or args.ransac_px <= 0:
        parser.error("Need at least 16 matches, 0 < ratio < 1 and positive ransac_px")
    if min(args.revisit_pairs, args.max_offset, args.previews) < 0 or min(args.local_gaps + [args.revisit_gap]) < 1:
        parser.error("Pair/offset/preview budgets must be nonnegative and sequence gaps positive")
    try:
        from utils.pose_geometry_check import GeometryDataset, run_check
        data = GeometryDataset(args.source_path, args.model_path, args.images)
        run_check(data, args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Geometry check: {error}\n")


if __name__ == "__main__":
    main()
