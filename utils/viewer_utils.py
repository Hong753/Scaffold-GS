"""Checkpoint and camera helpers shared by the native browser viewer and tests."""

import ast
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re

import numpy as np


def read_model_config(model_path):
    """Read the repository's Namespace(...) config without evaluating code."""
    path = Path(model_path) / "cfg_args"
    expression = ast.parse(path.read_text(), mode="eval").body
    if not (isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name)
            and expression.func.id == "Namespace" and not expression.args
            and all(keyword.arg is not None for keyword in expression.keywords)):
        raise ValueError(f"Expected a saved Namespace(...) configuration in {path}")
    return {keyword.arg: ast.literal_eval(keyword.value) for keyword in expression.keywords}


def resolve_checkpoint(model_path, iteration=-1):
    root = Path(model_path).expanduser().resolve()
    if root.parent.name == "point_cloud" and re.fullmatch(r"iteration_\d+", root.name):
        requested = int(root.name.split("_")[-1])
        if iteration not in (-1, requested):
            raise ValueError("--iteration disagrees with the supplied checkpoint directory")
        iteration, root = requested, root.parent.parent
    config = read_model_config(root)
    if iteration == -1:
        iterations = [int(path.name.split("_")[-1]) for path in (root / "point_cloud").glob("iteration_*")
                      if path.is_dir() and re.fullmatch(r"iteration_\d+", path.name)]
        if not iterations:
            raise FileNotFoundError(f"No iteration checkpoints under {root / 'point_cloud'}")
        iteration = max(iterations)
    checkpoint = root / "point_cloud" / f"iteration_{iteration}"
    native = checkpoint / "point_cloud_scaffold.ply"
    if not native.is_file():
        native = checkpoint / "point_cloud.ply"
    if not native.is_file():
        raise FileNotFoundError(f"No native anchor PLY in {checkpoint}")
    with native.open("rb") as stream:
        header = bytearray()
        while len(header) < 65536:
            line = stream.readline()
            if not line:
                break
            header.extend(line)
            if line.strip() == b"end_header":
                break
    if b"f_anchor_feat_0" not in header or b"f_offset_0" not in header:
        raise ValueError("This is a baked PLY. The viewer needs point_cloud_scaffold.ply "
                         "(or a legacy native point_cloud.ply) plus the saved MLPs.")
    required = ["opacity_mlp.pt", "cov_mlp.pt", "color_mlp.pt"]
    if config.get("use_feat_bank", False):
        required.append("feature_bank_mlp.pt")
    if config.get("appearance_dim", 32):
        required.append("embedding_appearance.pt")
    for name in required:
        if not (checkpoint / name).is_file():
            raise FileNotFoundError(f"Missing {checkpoint / name}; native MLP checkpoints are required")
    return root, checkpoint, native, config, iteration


def quaternion_matrix(wxyz):
    q = np.asarray(wxyz, dtype=np.float64)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Camera quaternion must be finite and nonzero")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])


def matrix_quaternion(matrix):
    # Largest eigenvector avoids branch singularities at 180-degree rotations.
    m = np.asarray(matrix, dtype=np.float64)
    k = np.array([
        [m[0, 0]-m[1, 1]-m[2, 2], m[1, 0]+m[0, 1], m[2, 0]+m[0, 2], m[2, 1]-m[1, 2]],
        [m[1, 0]+m[0, 1], m[1, 1]-m[0, 0]-m[2, 2], m[2, 1]+m[1, 2], m[0, 2]-m[2, 0]],
        [m[2, 0]+m[0, 2], m[2, 1]+m[1, 2], m[2, 2]-m[0, 0]-m[1, 1], m[1, 0]-m[0, 1]],
        [m[2, 1]-m[1, 2], m[0, 2]-m[2, 0], m[1, 0]-m[0, 1], np.trace(m)],
    ]) / 3
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1][[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def world_to_camera(wxyz, position):
    """Viser and Scaffold both use OpenCV cameras: +X right, +Y down, +Z forward."""
    rotation = quaternion_matrix(wxyz)
    position = np.asarray(position, dtype=np.float64)
    if position.shape != (3,) or not np.isfinite(position).all():
        raise ValueError("Camera position must be a finite 3-vector")
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = rotation.T
    result[:3, 3] = -rotation.T @ position
    return result


def render_dimensions(aspect, max_size):
    if not math.isfinite(aspect) or not 0.01 <= aspect <= 100:
        raise ValueError("Invalid browser viewport aspect ratio")
    if aspect >= 1:
        return int(max_size), max(1, round(max_size / aspect))
    return max(1, round(max_size * aspect)), int(max_size)


@dataclass
class CameraPreset:
    name: str
    position: np.ndarray
    wxyz: np.ndarray
    fov: float
    appearance_id: int = 0


def read_camera_presets(root):
    path = Path(root) / "cameras.json"
    if not path.is_file():
        return []
    training_ids = {}
    manifest = Path(root) / "frame_selection.json"
    if manifest.is_file():
        selected = json.loads(manifest.read_text())["selected_images"]
        training_ids = {Path(path).stem: index for index, path in enumerate(selected)}
    presets = []
    for record in json.loads(path.read_text()):
        presets.append(CameraPreset(
            name=str(record["img_name"]), position=np.asarray(record["position"], dtype=np.float64),
            wxyz=matrix_quaternion(record["rotation"]),
            fov=2 * math.atan(float(record["height"]) / (2 * float(record["fy"]))),
            appearance_id=training_ids.get(str(record["img_name"]), 0),
        ))
    return presets


def overview_preset(center, radius, up):
    center, up = np.asarray(center), np.asarray(up)
    up = up / np.linalg.norm(up)
    side = np.cross(up, [1., 0., 0.])
    if np.linalg.norm(side) < 0.1:
        side = np.cross(up, [0., 1., 0.])
    side /= np.linalg.norm(side)
    position = center + (side + 0.4 * up) * max(radius, 1e-3) * 2.5
    forward = (center - position) / np.linalg.norm(center - position)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return CameraPreset("Overview", position, matrix_quaternion(np.column_stack((right, down, forward))),
                        math.radians(60))
