"""CPU checks for PLY baking; CUDA-only training extensions are not exercised."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from plyfile import PlyData

from utils.neural_gaussians import generate_neural_gaussians
from utils.sh_utils import SH2RGB


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def model_class():
    # Isolate the model's file I/O from unavailable CUDA training extensions
    # and Scene's dataset imports. The model and its real MLPs run on CPU.
    scene = ModuleType("scene")
    scene.__path__ = [str(ROOT / "scene")]
    embedding = ModuleType("scene.embedding")
    embedding.Embedding = torch.nn.Embedding
    scatter = ModuleType("torch_scatter")
    knn = ModuleType("simple_knn._C")

    def unavailable(*args, **kwargs):
        raise AssertionError("A CUDA training extension was called during PLY export")

    scatter.scatter_max = unavailable
    knn.distCUDA2 = unavailable
    modules = {"scene": scene, "scene.embedding": embedding,
               "torch_scatter": scatter, "simple_knn": ModuleType("simple_knn"),
               "simple_knn._C": knn}
    spec = importlib.util.spec_from_file_location("_ply_test_model", ROOT / "scene/gaussian_model.py")
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module.GaussianModel


def make_model(model_class, feature_bank=False, appearance=False, distances=False):
    torch.manual_seed(17)
    with patch.object(torch.nn.Module, "cuda", lambda module: module):
        model = model_class(feat_dim=4, n_offsets=3, use_feat_bank=feature_bank,
                            appearance_dim=2 if appearance else 0,
                            add_opacity_dist=distances, add_cov_dist=distances,
                            add_color_dist=distances)
        model.set_appearance(2)
    model._anchor = torch.nn.Parameter(torch.tensor([
        [0., 0., 0.], [100., 0., 0.], [0., -30., 2.], [1., 2., 3.], [-1., -2., -3.]]))
    model._offset = torch.nn.Parameter(torch.randn(5, 3, 3) * 0.2)
    model._anchor_feat = torch.nn.Parameter(torch.randn(5, 4))
    model._scaling = torch.nn.Parameter(torch.randn(5, 6) * 0.2)
    model._opacity = torch.nn.Parameter(torch.full((5, 1), -2.0))
    model._rotation = torch.nn.Parameter(torch.tensor([[1., 0., 0., 0.]]).repeat(5, 1))
    # Exactly two active children per anchor, independent of the selected view.
    with torch.no_grad():
        model.mlp_opacity[2].weight.zero_()
        model.mlp_opacity[2].bias.copy_(torch.atanh(torch.tensor([0.25, -0.5, 0.8])))
    return model


def columns(vertices, names):
    return np.column_stack([vertices[name] for name in names])


CAMERA = SimpleNamespace(camera_center=torch.tensor([2., 3., 5.]), uid=1)


@pytest.mark.parametrize("feature_bank", [False, True])
@pytest.mark.parametrize("appearance", [False, True])
@pytest.mark.parametrize("distances", [False, True])
def test_export_matches_decoded_gaussians(model_class, tmp_path, feature_bank, appearance, distances):
    model = make_model(model_class, feature_bank, appearance, distances)
    path = tmp_path / "point_cloud.ply"
    assert model.save_ply(path, CAMERA, chunk_size=2) == 10
    # Match PlayCanvas's literal-space header tokenizer, which rejects a
    # space-padded vertex count even though plyfile accepts that whitespace.
    header = path.read_bytes().split(b"end_header\n", 1)[0].decode("ascii")
    vertex_line = next(line for line in header.splitlines() if line.startswith("element vertex "))
    assert int(vertex_line.split(" ")[2], 10) == 10
    ply = PlyData.read(path)
    assert not ply.text and ply.byte_order == "<"
    vertices = ply["vertex"].data
    xyz, rgb, opacity, scale, rotation = [x.detach().numpy() for x in
        generate_neural_gaussians(CAMERA, model)]
    np.testing.assert_allclose(columns(vertices, ["x", "y", "z"]), xyz, atol=1e-6)
    np.testing.assert_allclose(SH2RGB(columns(vertices, ["f_dc_0", "f_dc_1", "f_dc_2"])), rgb, atol=1e-6)
    np.testing.assert_allclose(1 / (1 + np.exp(-vertices["opacity"])), opacity[:, 0], atol=1e-6)
    np.testing.assert_allclose(np.exp(columns(vertices, ["scale_0", "scale_1", "scale_2"])), scale, atol=1e-6)
    np.testing.assert_allclose(columns(vertices, ["rot_0", "rot_1", "rot_2", "rot_3"]), rotation, atol=1e-6)
    assert not any(name.startswith("f_offset_") for name in vertices.dtype.names)
    assert set(name for name in vertices.dtype.names if name.startswith("scale_")) == {"scale_0", "scale_1", "scale_2"}
    # The distant anchor is retained: export never uses a camera frustum.
    assert vertices["x"].max() > 90
    # Independent check of explicit center construction and active-child order.
    centers = model._anchor[:, None] + model._offset * model._scaling[:, None, :3].exp()
    np.testing.assert_allclose(xyz, centers[:, [0, 2]].reshape(-1, 3).detach().numpy(), atol=1e-6)


def test_chunks_default_camera_and_training_modes(model_class, tmp_path):
    model = make_model(model_class, feature_bank=True, appearance=True)
    model.mlp_cov.eval()
    modes = [module.training for module in [model.mlp_opacity, model.mlp_cov, model.mlp_color,
                                             model.mlp_feature_bank, model.embedding_appearance]]
    model._ply_export_camera = CAMERA
    for chunk_size in [1, 99]:
        model.save_ply(tmp_path / "{}.ply".format(chunk_size), chunk_size=chunk_size)
    a = PlyData.read(tmp_path / "1.ply")["vertex"].data
    b = PlyData.read(tmp_path / "99.ply")["vertex"].data
    for name in a.dtype.names:
        np.testing.assert_allclose(a[name], b[name], atol=1e-6)
    assert modes == [module.training for module in [model.mlp_opacity, model.mlp_cov, model.mlp_color,
                                                   model.mlp_feature_bank, model.embedding_appearance]]
    # A standalone model without a Scene also exports successfully.
    model._ply_export_camera = None
    assert model.save_ply(tmp_path / "standalone.ply") == 10


def test_failed_export_preserves_file_and_modes(model_class, tmp_path):
    model = make_model(model_class)
    model.mlp_cov.eval()
    destination = tmp_path / "existing.ply"
    destination.write_bytes(b"original")
    with torch.no_grad():
        model.mlp_color[2].bias.fill_(float("nan"))
    with pytest.raises(ValueError, match="Non-finite"):
        model.save_ply(destination, CAMERA, chunk_size=1)
    assert destination.read_bytes() == b"original"
    assert list(tmp_path.iterdir()) == [destination]
    assert model.mlp_opacity.training and not model.mlp_cov.training


def test_empty_inactive_and_coincident_camera(model_class, tmp_path):
    model = make_model(model_class)
    coincident = SimpleNamespace(camera_center=model._anchor[0], uid=0)
    assert model.save_ply(tmp_path / "coincident.ply", coincident) == 10
    with torch.no_grad():
        model.mlp_opacity[2].bias.fill_(-1)
    assert model.save_ply(tmp_path / "inactive.ply", CAMERA) == 0
    assert len(PlyData.read(tmp_path / "inactive.ply")["vertex"].data) == 0
    for name in ["_anchor", "_offset", "_anchor_feat", "_scaling"]:
        setattr(model, name, torch.nn.Parameter(getattr(model, name)[:0]))
    assert model.save_ply(tmp_path / "empty.ply") == 0
    with pytest.raises(ValueError, match="chunk_size"):
        model.save_ply(tmp_path / "invalid.ply", chunk_size=0)


def test_native_and_legacy_checkpoint_loading(model_class, tmp_path):
    model = make_model(model_class)
    viewer_path = tmp_path / "point_cloud.ply"
    native_path = tmp_path / "point_cloud_scaffold.ply"
    legacy_path = tmp_path / "legacy.ply"
    model.save_ply(viewer_path, CAMERA)
    model.save_anchor_ply(native_path)
    model.save_anchor_ply(legacy_path)
    tensor = torch.tensor

    def cpu_tensor(*args, **kwargs):
        if kwargs.get("device") == "cuda":
            kwargs["device"] = "cpu"
        return tensor(*args, **kwargs)

    for path in [viewer_path, native_path, legacy_path]:
        restored = make_model(model_class)
        with patch("torch.tensor", side_effect=cpu_tensor):
            restored.load_ply_sparse_gaussian(path)
        for name in ["_anchor", "_offset", "_anchor_feat", "_scaling", "_rotation", "_opacity"]:
            torch.testing.assert_close(getattr(restored, name), getattr(model, name))
    native_path.unlink()
    with pytest.raises(ValueError, match="_scaffold.ply"):
        model.load_ply_sparse_gaussian(viewer_path)
