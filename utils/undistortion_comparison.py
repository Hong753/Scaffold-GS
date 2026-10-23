"""Small, deterministic evaluation sets for the distortion ablation."""

import json
from pathlib import Path


def evaluation_cameras(cameras, count):
    if count < 1:
        raise ValueError("comparison_eval_views must be positive")
    ordered = sorted(cameras, key=lambda camera: (camera.image_name, camera.image_path))
    if len(ordered) <= count:
        return ordered
    return [ordered[round(index * (len(ordered) - 1) / (count - 1))]
            for index in range(count)] if count > 1 else [ordered[len(ordered) // 2]]


def write_evaluation(model_path, iteration, split, records):
    if not records:
        return
    output = Path(model_path) / "comparison_metrics"
    output.mkdir(parents=True, exist_ok=True)
    payload = {"iteration": iteration, "split": split, "views": records,
               "native_psnr": sum(item["native_psnr"] for item in records) / len(records),
               "raw_grid_psnr": sum(item["raw_grid_psnr"] for item in records) / len(records)}
    (output / f"{iteration:06d}_{split}.json").write_text(json.dumps(payload, indent=2) + "\n")
