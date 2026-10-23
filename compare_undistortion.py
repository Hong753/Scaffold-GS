#!/usr/bin/env python3
"""Run matched raw/rectified training jobs and summarize common-grid PSNR."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys


def training_command(args, output, rectified):
    evaluations = sorted({step for step in (7000, 30000, args.iterations) if step <= args.iterations})
    command = [sys.executable, "train.py", "-s", str(Path(args.source).resolve()),
               "-m", str((output / ("undistorted" if rectified else "baseline")).resolve()),
               "--eval", "--gpu", args.gpu, "--iterations", str(args.iterations),
               "--resolution", str(args.resolution), "--appearance_dim", str(args.appearance_dim),
               "--voxel_size", str(args.voxel_size), "--update_init_factor", "16", "--ratio", "1",
               "--frame_selection_stride", str(args.frame_selection_stride),
               "--compare_undistortion", "--comparison_eval_views", str(args.eval_views),
               "--undistort_cache_dir", str(Path(args.cache_dir).resolve()),
               "--skip_final_render", "--port", str(args.port),
               "--test_iterations", *map(str, evaluations)]
    if rectified:
        command.append("--undistort_images")
    return command


def summarize(output):
    rows = []
    for path in sorted((output / "baseline/comparison_metrics").glob("*.json")):
        other = output / "undistorted/comparison_metrics" / path.name
        if not other.is_file():
            continue
        baseline, rectified = json.loads(path.read_text()), json.loads(other.read_text())
        supports = lambda data: [(item["image"], item["raw_grid_pixels"]) for item in data["views"]]
        if supports(baseline) != supports(rectified):
            raise ValueError(f"Evaluation frames/support differ for {path.name}; comparison is invalid")
        rows.append({"iteration": baseline["iteration"], "split": baseline["split"],
                     "views": len(baseline["views"]),
                     "baseline_raw_grid_psnr": baseline["raw_grid_psnr"],
                     "undistorted_raw_grid_psnr": rectified["raw_grid_psnr"],
                     "delta_db": rectified["raw_grid_psnr"] - baseline["raw_grid_psnr"],
                     "baseline_native_psnr": baseline["native_psnr"],
                     "undistorted_native_psnr": rectified["native_psnr"]})
    if not rows:
        raise ValueError("No matching comparison metrics found; both training jobs must reach an evaluation step")
    # Frame-selection manifests should be identical because scoring uses the
    # original images in both jobs. Never accept a changed training subset.
    manifests = [output / name / "frame_selection.json" for name in ("baseline", "undistorted")]
    if any(path.exists() for path in manifests):
        if not all(path.exists() for path in manifests):
            raise ValueError("Only one run has a training-frame selection manifest")
        selections = [json.loads(path.read_text())["selected_images"] for path in manifests]
        if selections[0] != selections[1]:
            raise ValueError("Training subsets differ; comparison is invalid")
    (output / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["| Step | Split | Views | Raw baseline PSNR | Undistorted PSNR | Delta dB |",
             "| ---: | :--- | ---: | ---: | ---: | ---: |"]
    for row in rows:
        lines.append(f"| {row['iteration']} | {row['split']} | {row['views']} | "
                     f"{row['baseline_raw_grid_psnr']:.3f} | {row['undistorted_raw_grid_psnr']:.3f} | "
                     f"{row['delta_db']:+.3f} |")
    lines += ["", "PSNR above uses identical original-image pixels in both runs.",
              "Undistorted predictions are warped back to the original image grid.",
              "Native-grid PSNR is also recorded in comparison.json; its targets differ between runs.",
              "Preview panels: original RGB | prediction on original grid | absolute error ×4."]
    table = "\n".join(lines) + "\n"
    (output / "comparison.md").write_text(table)
    print(table)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-s", "--source")
    parser.add_argument("-o", "--output", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--iterations", type=int, default=40000)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--frame_selection_stride", type=int, default=3)
    parser.add_argument("--appearance_dim", type=int, default=0)
    parser.add_argument("--voxel_size", type=float, default=0.001)
    parser.add_argument("--eval_views", type=int, default=32, help="Fixed views per split")
    parser.add_argument("--cache_dir", default="outputs/undistorted")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument("--summarize", action="store_true", help="Summarize completed jobs without training")
    parser.add_argument("--dry_run", action="store_true", help="Print commands without creating/running jobs")
    args = parser.parse_args()
    if args.summarize:
        if args.output is None:
            parser.error("--summarize requires --output")
        summarize(args.output)
        return
    if not args.source:
        parser.error("--source is required for training")
    if args.iterations < 1 or args.eval_views < 1 or args.frame_selection_stride < 1:
        parser.error("iterations, eval_views and frame_selection_stride must be positive")
    output = args.output or Path("outputs/undistortion_comparison") / datetime.now().strftime("%Y%m%d_%H%M%S")
    output = output.resolve()
    commands = [training_command(args, output, rectified) for rectified in (False, True)]
    if args.dry_run:
        for command in commands:
            print(shlex.join(command))
        return
    if output.exists():
        parser.error(f"Output already exists: {output}; choose a new path or use --summarize")
    output.mkdir(parents=True)
    (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
    root = Path(__file__).resolve().parent
    for name, command in zip(("baseline", "undistorted"), commands):
        print(f"Starting {name}: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=root, check=True)
    summarize(output)
    print(f"Comparison saved: {output / 'comparison.md'}")


if __name__ == "__main__":
    main()
