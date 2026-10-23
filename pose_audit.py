#!/usr/bin/env python3
"""Visualize COLMAP camera poses and their alignment with training RGB images."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model_path", help="Saved training run; reads cfg_args and frame_selection.json")
    parser.add_argument("-s", "--source_path", help="Override COLMAP dataset location; required without -m")
    parser.add_argument("--resolution", type=int, help="Override training image resolution")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--max_points", type=int, default=100000)
    parser.add_argument("--max_frustums", type=int, default=120)
    parser.add_argument("--report_only", action="store_true", help="Write an offline image-overlay gallery and exit")
    parser.add_argument("--views", type=int, default=24, help="Evenly spaced images for offline report")
    parser.add_argument("--output", type=Path, help="Report directory; default RUN/pose_audit")
    args = parser.parse_args()
    if not args.model_path and not args.source_path:
        parser.error("Supply -m RUN_DIRECTORY or -s COLMAP_DATASET")
    if min(args.max_points, args.max_frustums, args.views) < 1:
        parser.error("max_points, max_frustums and views must be positive")
    try:
        from utils.pose_audit_data import PoseAuditData
        print("Indexing COLMAP poses and sparse points (no CUDA or Gaussian model required)…", flush=True)
        data = PoseAuditData(args.model_path, args.source_path, args.resolution)
        print(f"Indexed {len(data.frames):,} cameras and {len(data.xyz):,} points; image pixels load only on selection.", flush=True)
        if args.report_only:
            output = args.output or (data.root or Path("outputs")) / "pose_audit"
            print(f"Report: {data.export_report(output, args.views).resolve()}")
            return
        try:
            from utils.pose_audit_app import PoseAuditApp
        except ModuleNotFoundError as error:
            if error.name == "viser":
                parser.error("Install the browser dependency: pip install -r requirements-viewer.txt")
            raise
        app = PoseAuditApp(data, args.host, args.port, args.max_points, args.max_frustums)
        print(f"Pose audit: http://localhost:{args.port} (or use this machine's IP)", flush=True)
        app.run()
    except (OSError, ValueError) as error:
        parser.exit(1, f"Pose audit: {error}\n")


if __name__ == "__main__":
    main()
