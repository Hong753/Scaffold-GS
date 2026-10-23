"""Independent geometry, synchronization holdouts, and CPU-only CLI coverage."""

from argparse import Namespace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import cv2
import numpy as np
from PIL import Image
import pytest

from utils.pose_geometry_check import (FeatureStore, GeometryDataset, offset_scores,
    pixel_calibration, plan_pairs, pose_fundamental, sampson, sequence_block,
    summarize_offsets, verify_matches)


def cameras(count=120):
    frames = []
    for i in range(count):
        rotation = cv2.Rodrigues(np.array([.035 * np.sin(i * .31), .12 * np.sin(i * .17), .02 * np.cos(i * .27)]))[0]
        center = np.array([.02 * i, .13 * np.sin(i * .23), .07 * np.cos(i * .19)])
        frames.append(SimpleNamespace(rotation=rotation, tvec=-rotation @ center, center=center, camera_id=1))
    return frames


def observations(frames, i, j, seed=9):
    rng = np.random.default_rng(seed)
    xyz = np.c_[rng.uniform(-1.8, 1.8, (120, 2)), rng.uniform(3.5, 7, 120)]
    K = np.array([[510., 0, 319.5], [0, 505., 239.5], [0, 0, 1.]])
    def project(frame):
        p = (xyz @ frame.rotation.T + frame.tvec) @ K.T
        return p[:, :2] / p[:, 2:]
    return dict(a=project(frames[i]), b=project(frames[j]), Ka=K, Kb=K)


def test_world_to_camera_epipolar_geometry_and_perturbation():
    frames = cameras()
    match = observations(frames, 12, 17)
    F = pose_fundamental(frames[12], frames[17], match['Ka'], match['Kb'])
    assert np.max(sampson(F, match['a'], match['b'])) < 1e-9
    wrong = pose_fundamental(frames[14], frames[19], match['Ka'], match['Kb'])
    assert np.median(sampson(wrong, match['a'], match['b'])) > 1
    assert pose_fundamental(frames[12], frames[12], match['Ka'], match['Kb']) is None


def test_pixel_calibration_preserves_colmap_corner_origin_at_resize():
    calibration = dict(width=2048, height=1536, matrix=[[1230., 0, 1023.], [0, 1220., 763.], [0, 0, 1]], distortion=[.03, 0, 0, 0])
    K, D = pixel_calibration(calibration, (1000, 750))
    xyz = np.array([[.3, -.2, 3.]])
    raw = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), np.array(calibration['matrix'], float), D)[0].reshape(-1, 2)
    resized = cv2.projectPoints(xyz, np.zeros(3), np.zeros(3), K, D)[0].reshape(-1, 2)
    np.testing.assert_allclose(resized, raw * [1000 / 2048, 750 / 1536] - .5)


def test_offset_sweep_recovers_known_shift_with_fixed_correspondences():
    truth = cameras()
    stored = [truth[max(i - 2, 0)] for i in range(len(truth))]
    match = observations(truth, 32, 38)
    scores = offset_scores(stored, 32, 38, match, 5)
    assert min(scores, key=scores.get) == '2'
    assert scores['2'] < 1e-9 and scores['0'] > 1
    assert offset_scores(stored, 1, 8, match, 5) is None
    changed = list(stored)
    changed[34] = SimpleNamespace(**{**vars(stored[34]), 'camera_id': 2})
    assert offset_scores(changed, 32, 38, match, 5) is None


def test_non_linear_camera_sequence_recovers_shift_across_held_out_segments():
    truth = cameras(240)
    stored = [truth[max(i - 2, 0)] for i in range(len(truth))]
    rows = []
    for block in range(4):
        for offset in range(12, 30, 3):
            i, j = block * 60 + offset, block * 60 + offset + 5
            match = observations(truth, i, j)
            rows.append(dict(sequence_block=sequence_block(i, j, 240, 5), offset_scores=offset_scores(stored, i, j, match, 5)))
    result = summarize_offsets(rows, 5)
    assert result['status'] == 'candidate_offset' and result['selected_offset'] == 2
    selected = next(row for row in result['sweep'] if row['offset'] == 2)
    assert selected['validation_median_px'] < 1e-8


def test_sequence_blocks_do_not_share_any_shifted_pose_support():
    assert sequence_block(8, 15, 120, 5) == 0
    assert sequence_block(40, 45, 120, 5) == 1
    assert sequence_block(25, 28, 120, 5) is None
    assert sequence_block(1, 8, 120, 5) is None


@pytest.mark.parametrize('validation_target,expected', [(2, 'candidate_offset'), (0, 'inconclusive')])
def test_offset_is_chosen_only_on_discovery_and_checked_on_separate_blocks(validation_target, expected):
    rows = []
    for block in range(4):
        target = 2 if block % 2 == 0 else validation_target
        for k in range(6):
            rows.append(dict(sequence_block=block, offset_scores={str(s): abs(s - target) + .2 + k * .01 for s in range(-5, 6)}))
    result = summarize_offsets(rows, 5)
    assert result['selected_offset'] == 2
    assert result['status'] == expected
    assert result['discovery_pairs'] == result['validation_pairs'] == 12
    assert summarize_offsets(rows[:6], 5)['status'] == 'unavailable'


def test_independent_matching_rejects_descriptor_outliers_without_using_saved_poses():
    frames = cameras()
    match = observations(frames, 2, 7)
    rng = np.random.default_rng(23)
    a, b = match['a'].copy(), match['b'].copy()
    a += rng.normal(0, .08, a.shape)
    b += rng.normal(0, .08, b.shape)
    b[:20] = rng.uniform([0, 0], [640, 480], (20, 2))
    descriptors = rng.normal(size=(len(a), 128)).astype(np.float32)
    measured, reason = verify_matches(a, b, descriptors, descriptors.copy(), ((640, 480), (640, 480)))
    assert reason is None
    assert len(measured['a']) >= 95
    assert measured['fitted']['median'] < .3
    wrong = pose_fundamental(frames[4], frames[9], match['Ka'], match['Kb'])
    assert np.median(sampson(wrong, measured['a'], measured['b'])) > 1
    assert verify_matches(a[:10], b[:10], descriptors[:10], descriptors[:10], ((640, 480), (640, 480)))[1] == 'too_few_features'


def make_source(root, count=12, flat=False):
    source = root / 'dataset'
    sparse = source / ('sparse' if flat else 'sparse/0')
    sparse.mkdir(parents=True)
    images = source / 'images'
    images.mkdir()
    (sparse / 'cameras.txt').write_text('1 SIMPLE_PINHOLE 160 120 140 80 60\n')
    lines = []
    for i in range(count):
        # Rounded, sign-flipped quaternions are normalized by the check.
        q = '-1.000001 0 0 0' if i % 2 else '1.000001 0 0 0'
        lines += [f'{i+1} {q} {-i*.02} 0 0 1 frame{i:04d}.png', '0 0 -1']
        noise = np.random.default_rng(7).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        Image.fromarray(np.roll(noise, i, axis=1)).save(images / f'frame{i:04d}.png')
    (sparse / 'images.txt').write_text('\n'.join(lines) + '\n')
    return source


@pytest.mark.parametrize('flat', [False, True])
def test_dataset_needs_no_cloud_tracks_cuda_or_rgb_decode_for_indexing(tmp_path, monkeypatch, flat):
    source = make_source(tmp_path, flat=flat)
    with monkeypatch.context() as guard:
        guard.setattr(Image.Image, 'load', lambda *a, **k: pytest.fail('Indexing decoded RGB'))
        data = GeometryDataset(source)
    assert len(data.frames) == len(data.available) == 12
    np.testing.assert_allclose(data.frames[1].rotation, np.eye(3))
    assert not list(source.rglob('points3D*'))
    assert 'scene' not in sys.modules


def test_sparse_uploaded_images_are_not_mislabeled_as_adjacent_frames(tmp_path):
    source = make_source(tmp_path)
    for path in (source / 'images').glob('*.png'):
        if path.name not in ['frame0000.png', 'frame0005.png', 'frame0010.png']:
            path.unlink()
    data = GeometryDataset(source)
    planned = plan_pairs(data, 48, [1, 3], 10, 4, 42)
    assert planned and all(kind == 'revisit' for i, j, kind in planned)
    assert all(j - i >= 4 for i, j, kind in planned)
    assert plan_pairs(data, 48, [1], 0, 4, 42) == []


def test_feature_cache_avoids_repeat_decoding_and_stays_bounded(tmp_path, monkeypatch):
    source = make_source(tmp_path)
    data = GeometryDataset(source)
    store = FeatureStore(data, tmp_path / 'output', max_width=100, max_features=50)
    xy, desc, size = store.get(0)
    assert size == (100, 75) and xy.shape == (len(desc), 2)
    fresh = FeatureStore(data, tmp_path / 'output', max_width=100, max_features=50)
    with monkeypatch.context() as guard:
        guard.setattr(Image.Image, 'load', lambda *a, **k: pytest.fail('Cache decoded RGB'))
        np.testing.assert_array_equal(fresh.get(0)[0], xy)
    for i in range(6):
        store.get(i)
    assert len(store.memory) == 4
    path = data.paths[0]
    Image.fromarray(np.zeros((120, 160, 3), np.uint8)).save(path)
    assert len(store.get(0)[0]) == 0


def test_cli_creates_shareable_report_even_when_geometry_is_degenerate(tmp_path):
    source = make_source(tmp_path, count=4)
    output = tmp_path / 'geometry'
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'pose_geometry_check.py'),
               '-s', str(source), '-o', str(output), '--samples', '4', '--local_gaps', '1', '--revisit_pairs', '0',
               '--max_width', '160', '--max_features', '150', '--previews', '2']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'SEND THIS FILE:' in result.stdout
    with zipfile.ZipFile(tmp_path / 'geometry_results.zip') as bundle:
        names = bundle.namelist()
        assert 'geometry/index.html' in names and 'geometry/cameras.npz' in names
        assert not any('feature_cache' in name for name in names)
        saved = json.loads(bundle.read('geometry/results.json'))
        assert saved['synchronization']['status'] == 'unavailable'
        assert saved['registered_images'] == 4
        assert saved['attempted_pairs'] == 3


def test_corrupt_images_are_skipped_by_cli_instead_of_loaded_leniently(tmp_path):
    source = make_source(tmp_path, count=4)
    (source / 'images/frame0001.png').write_bytes(b'not an image')
    output = tmp_path / 'geometry'
    command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'pose_geometry_check.py'),
               '-s', str(source), '-o', str(output), '--samples', '4', '--local_gaps', '1', '--revisit_pairs', '0',
               '--max_features', '150', '--previews', '0']
    result = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    saved = json.loads((output / 'results.json').read_text())
    assert sum('corrupt_image' in row.get('reason', '') for row in saved['pairs']) == 2
