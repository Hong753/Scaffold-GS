# Scaffold-GS: Structured 3D Gaussians for View-Adaptive Rendering

[Tao Lu](https://github.com/inspirelt), [Mulin Yu](https://scholar.google.com/citations?user=w0Od3hQAAAAJ), [Linning Xu](https://eveneveno.github.io/lnxu), [Yuanbo Xiangli](https://kam1107.github.io/), [Limin Wang](https://wanglimin.github.io/), [Dahua Lin](http://dahua.site/), [Bo Dai](https://daibo.info/) <br />


[[`Project Page`](https://city-super.github.io/scaffold-gs/)][[`arxiv`](https://arxiv.org/abs/2312.00109)][[`Viewer`](https://drive.google.com/file/d/17nPVnRRxO4zMQJ_fKHm13HzvD0wUZ8XF/view?usp=sharing)]

## Setup Environment
Create environment (Linux-x86_64):
```bash
conda create --name scaffoldgs python=3.12 -y
conda activate scaffoldgs

# Optional
conda install spyder -y

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
pip install -r requirements.txt
pip install --no-build-isolation git+https://github.com/camenduru/simple-knn
pip install --no-build-isolation submodules/diff-gaussian-rasterization
```

## News

**[2024.09.25]** 🎈We propose **Octree-AnyGS**, a general anchor-based framework that supports explicit Gaussians (2D-GS, 3D-GS) and neural Gaussians (Scaffold-GS). Additionally, **Octree-GS** has been adapted to the aforementioned Gaussian primitives, enabling Level-of-Detail representation for large-scale scenes. This framework holds potential for application to other Gaussian-based methods, with relevant SIBR visualizations forthcoming.(https://github.com/city-super/Octree-AnyGS)

**[2024.05.28]**  We update the [viewer](https://github.com/city-super/Scaffold-GS/tree/main/SIBR_viewers) to conform to the file structure at training.

**[2024.04.05]**  Scaffold-GS is selected as a 🎈**highlight** in CVPR2024.

**[2024.03.27]**  🎈We release [Octree-GS](https://city-super.github.io/octree-gs), supporting an explicit *LOD* representation, rendering faster in large-scale scene with high quality.

**[2024.03.26]**  🎈We release [GSDF](https://city-super.github.io/GSDF/), which improves rendering and reconstruction quality simultaneously.

**[2024.02.27]**  Accepted to [CVPR 2024](https://cvpr.thecvf.com/).

**[2024.01.22]**  We add the appearance embedding to Scaffold-GS to handle wild scenes.

**[2024.01.22]** 🎈👀 The [viewer](https://github.com/city-super/Scaffold-GS/tree/main/SIBR_viewers) for Scaffold-GS is available now. 

**[2023.12.10]** We release the code.

## TODO List
- [ ] Explore on removing the MLP module
- [ ] Improve the training configuration system

## Overview

<p align="center">
<img src="assets/pipeline.png" width=100% height=100% 
class="center">
</p>


We introduce Scaffold-GS, which uses anchor points to distribute local 3D Gaussians, and predicts their attributes on-the-fly based on viewing direction and distance within the view frustum.

Our method performs superior on scenes with challenging observing views. e.g. transparency, specularity, reflection, texture-less regions and fine-scale details.

<p align="center">
<img src="assets/teaser_big.png" width=100% height=100% 
class="center">
</p>





## Installation

We tested on a server configured with Ubuntu 18.04, cuda 11.6 and gcc 9.4.0. Other similar configurations should also work, but we have not verified each one individually.

1. Clone this repo:

```
git clone https://github.com/city-super/Scaffold-GS.git --recursive
cd Scaffold-GS
```

2. Install dependencies

```
SET DISTUTILS_USE_SDK=1 # Windows only
conda env create --file environment.yml
conda activate scaffold_gs
```

## Data

First, create a ```data/``` folder inside the project path by 

```
mkdir data
```

The data structure will be organised as follows:

```
data/
├── dataset_name
│   ├── scene1/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
│   ├── scene2/
│   │   ├── images
│   │   │   ├── IMG_0.jpg
│   │   │   ├── IMG_1.jpg
│   │   │   ├── ...
│   │   ├── sparse/
│   │       └──0/
...
```


### Public Data

The BungeeNeRF dataset is available in [Google Drive](https://drive.google.com/file/d/1nBLcf9Jrr6sdxKa1Hbd47IArQQ_X8lww/view?usp=sharing)/[百度网盘[提取码:4whv]](https://pan.baidu.com/s/1AUYUJojhhICSKO2JrmOnCA). The MipNeRF360 scenes are provided by the paper author [here](https://jonbarron.info/mipnerf360/). And we test on scenes ```bicycle, bonsai, counter, garden, kitchen, room, stump```. The SfM data sets for Tanks&Temples and Deep Blending are hosted by 3D-Gaussian-Splatting [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip). Download and uncompress them into the ```data/``` folder.

### Custom Data

The Python loader accepts COLMAP models in either `sparse/` or `sparse/0/`,
using binary or text camera files. If both layouts contain a camera model,
`sparse/` takes precedence. The selected directory is also used for `points3D`.

For custom data, you should process the image sequences with [Colmap](https://colmap.github.io/) to obtain the SfM points and camera poses. Then, place the results into ```data/``` folder.


## Training

### Training multiple scenes

To train multiple scenes in parallel, we provide batch training scripts: 

 - Tanks&Temples: ```train_tnt.sh```
 - MipNeRF360: ```train_mip360.sh```
 - BungeeNeRF: ```train_bungee.sh```
 - Deep Blending: ```train_db.sh```
 - Nerf Synthetic: base ->```train_nerfsynthetic.sh```; with warmup->```train_nerfsynthetic_withwarmup.sh```

 run them with 

 ```
bash train_xxx.sh
 ```

 > Notice 1: Make sure you have enough GPU cards and memories to run these scenes at the same time.

 > Notice 2: Each process occupies many cpu cores, which may slow down the training process. Set ```torch.set_num_threads(32)``` accordingly in the ```train.py``` to alleviate it.

### Training a single scene

Camera images are loaded **from disk on demand** by default. Cameras retain
image paths, output dimensions and preprocessing metadata, without caching
decoded images in CPU RAM or CUDA memory. Training transfers only the selected
view to the render device; validation, rendering and metrics also process one
view or image pair at a time. `--data_device` controls the default target device
for explicitly loading an image, rather than preloading the dataset.

Truncated or undecodable images are skipped, with a warning naming each source
path once per run. Training selects another camera within the same optimization
step and excludes known corrupt views from subsequent sampling. Validation and
rendering also skip those views, and validation averages use only valid images.
Decoding remains strict: partial images are never used as training targets.
Permission and filesystem errors still stop the run. If no valid training views
remain, training stops with a clear error. No image pixels are cached.

For training a single scene, modify the path and configurations in ```single_train.sh``` accordingly and run it:

```
bash ./single_train.sh
```

For sequential video frames, `single_train.sh` defaults to
`--frame_selection_stride 3`. Training frames are sorted by filename in natural
numeric order (`frame2` before `frame10`), grouped into consecutive blocks of
three, and the frame with the highest **variance of the Laplacian** is selected
from each block. A partial final block also contributes one frame. For 15,000
training images this selects 5,000 images. Selection runs after the train/test
split; held-out cameras keep their existing membership. Corrupt candidates are
excluded, and an entirely corrupt block contributes no frame.

Sharpness is computed on grayscale images resized to at most 1,024 pixels on
their longest side, using four CPU workers. This is a sharpness proxy suitable
for nearby video frames. Only the scoring workers temporarily hold image pixels;
optimization still loads the selected images from disk on demand.
The first run scans the training images with a progress bar. Scores and selected
paths are cached under `outputs/frame_selection/`, keyed by paths, file sizes,
modification times and selection settings. Each model also saves
`frame_selection.json` with all scores and selected paths; rendering a saved model
reuses that exact selection without rescoring.

```bash
# Default: best frame per group of 3.
bash ./single_train.sh

# Change the stride, or set it to 1 to use all frames.
bash ./single_train.sh --frame_selection_stride 5
bash ./single_train.sh --frame_selection_stride 1

# Optional scoring settings; max_size=0 scores at native image resolution.
bash ./single_train.sh --frame_selection_stride 3 --frame_selection_max_size 0 --frame_selection_workers 4
```

The same options work with `train.sh` and `train.py`. Their stride defaults to 1,
so existing batch scripts continue to use all their training frames.

- scene: scene name with a format of ```dataset_name/scene_name/``` or ```scene_name/```;
- exp_name: user-defined experiment name;
- gpu: specify the GPU id to run the code. '-1' denotes using the most idle GPU. 
- voxel_size: size for voxelizing the SfM points, smaller value denotes finer structure and higher overhead, '0' means using the median of each point's 1-NN distance as the voxel size.
- update_init_factor: initial resolution for growing new anchors. A larger one will start placing new anchor in a coarser resolution.

> For these public datasets, the configurations of 'voxel_size' and 'update_init_factor' can refer to the above batch training script. 


This script will store the log (with running-time code) into ```outputs/dataset_name/scene_name/exp_name/cur_time``` automatically.





### Compare on-demand COLMAP undistortion

`single_train.sh` launches the undistorted version by default, under the
`undistorted` experiment name. It keeps stride 3 and 40k training steps:

```bash
bash ./single_train.sh
# Optional writable cache location (the source dataset is never modified):
bash ./single_train.sh --undistort_images --undistort_cache_dir /path/to/cache
```

The training progress bar shows `Anchors` (the current total, including anchor
growth/pruning) and `View splats` (positive-opacity child Gaussians decoded for
the sampled camera, before final rasterizer culling). Each anchor has
`n_offsets` potential children; the decoded view count varies with the camera.
Direct `train.py`/`train.sh` calls still require `--undistort_images` to enable it.

Only the selected camera image is decoded. On its first use, its COLMAP lens
distortion is removed directly at the training resolution and a lossless RGBA
PNG is written to `outputs/undistorted/`. Subsequent selections load that PNG
from disk; cameras never retain decoded images in RAM or CUDA memory. Cache
keys include the absolute source path, size, modification time, calibration,
resolution and OpenCV/cache version. Remap tables are limited to two entries.
Corrupt cached files are regenerated; corrupt source images are still skipped
strictly. The cache grows only as frames are used and may be deleted to reclaim
space.

Supported COLMAP models: `SIMPLE_PINHOLE`, `PINHOLE`, `SIMPLE_RADIAL`, `RADIAL`,
`OPENCV`, and `FULL_OPENCV`. Focal lengths, poses and image dimensions are
preserved. The output principal point is centered to match the native renderer;
invalid border pixels are masked in L1/SSIM training and PSNR evaluation. Saved
ground-truth PNGs retain this mask in alpha. Offline LPIPS zeroes invalid borders
in both images; use the common-grid PSNR below for the ablation. Fisheye and
other camera models require COLMAP's offline `image_undistorter` instead. Feed
the calibration belonging to the input images; do not apply an old radial
calibration to images that have already been undistorted.

Run the controlled raw/undistorted comparison with:

```bash
python compare_undistortion.py \
  -s /data/hyejun_dataset/HandHeld/Orange_office_Colmap \
  -o outputs/office_undistortion_ablation --gpu 0
```

This launches two **fresh, sequential 40k-step runs** using the existing seed 0,
stride-3 sharp-frame selection, appearance dimension 0, resolution -1 (1600-pixel
width cap), voxel size 0.001, and the same optimization defaults. Frame selection
is scored on original images in both modes. It evaluates the same 32 training
and 32 test views spread across each sequence at 7k, 30k and 40k; short runs
always evaluate at their final step. Both jobs save their normal PLYs, while the
expensive full-dataset render/evaluation pass is skipped. For a shorter pilot,
append `--iterations 7000`; other options include `--resolution`, `--eval_views`,
`--appearance_dim`, `--frame_selection_stride`, `--voxel_size`, and `--cache_dir`.
Use `--dry_run` to inspect commands without running anything. A new output
directory is required so existing experiments cannot be overwritten.

The summary table uses **common raw-grid PSNR**: rectified predictions are warped
back to the original image grid and both models are evaluated against the same
original RGB pixels on identical valid support. Native-grid PSNR is also saved,
but its targets differ between modes because undistortion resamples images.
The comparison includes principal-point recentering when the original principal
point was off-center, so any gain is a calibration-consistency gain rather than
proof that the radial coefficient alone was responsible.

Outputs under the chosen comparison directory:

* `comparison.md` / `comparison.json`: train/test common-grid PSNR and differences.
* `baseline/` / `undistorted/`: separate trained models and logs.
* Each run's `undistortion.json`: calibration and sampled distortion displacement
  in pixels at the original COLMAP resolution.
* Each run's `comparison_metrics/`: per-view native/common-grid metrics.
* Each run's `comparison_previews/`: four views per split and evaluation step;
  panels show original RGB, prediction on the original grid, and absolute error ×4.

Rebuild the summary without training:

```bash
python compare_undistortion.py -o outputs/office_undistortion_ablation --summarize
```

### SuperSplat-compatible PLY saving

Training saves at **7,000 and 30,000 iterations** by default (and at the final
iteration if a different training length is selected). Override these with
`--save_iterations` when needed.

Open either of these files directly in SuperSplat:

```text
<model_path>/point_cloud/iteration_7000/point_cloud.ply
<model_path>/point_cloud/iteration_30000/point_cloud.ply
```

`GaussianModel.save_ply(path)` now expands anchors into their child Gaussians,
evaluates the neural decoders, and writes a binary little-endian standard 3DGS
PLY with SH-degree-0 color, logit opacity, log scale and quaternion rotation.
The reference is the training camera with the alphabetically first image name.
All anchors are evaluated in chunks, without camera-frustum filtering; children
with non-positive predicted opacity are omitted, as in the native renderer.
Color, opacity, scale and rotation are baked for that camera and remain fixed
when navigating in SuperSplat.

This is a **single-view approximation**. Children omitted by the reference
camera's opacity prediction cannot reappear from another view, and their
view-dependent shape and color cannot adapt. A file loading successfully in
SuperSplat does not establish visual parity with Scaffold-GS. Use the native
browser viewer below to inspect the model with its original neural rendering.

For a different bake, call
`gaussians.save_ply(path, viewpoint_camera=camera, chunk_size=65536)`.
Outside a `Scene`, a call without a camera uses a synthetic camera beyond the
anchor bounds and appearance UID 0. A trained model and its MLPs must be loaded
before export.

Each training save also keeps `point_cloud_scaffold.ply` and the MLP checkpoints
for native Scaffold-GS loading. The Python loader automatically selects this
native sidecar when given `point_cloud.ply`; legacy anchor-only checkpoints are
still supported. Keep these files together for subsequent native rendering.
`save_anchor_ply(path)` explicitly writes the original anchor representation.

## Evaluation

We've integrated the rendering and metrics calculation process into the training code. So, when completing training, the ```rendering results```, ```fps``` and ```quality metrics``` will be printed automatically. And the rendering results will be save in the log dir. Mind that the ```fps``` is roughly estimated by 

```
torch.cuda.synchronize();t_start=time.time()
rendering...
torch.cuda.synchronize();t_end=time.time()
```

which may differ somewhat from the original 3D-GS, but it does not affect the analysis.

Meanwhile, we keep the manual rendering function with a similar usage of the counterpart in [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting), one can run it by 

```
python render.py -m <path to trained model> # Generate renderings
python metrics.py -m <path to trained model> # Compute error metrics on renderings
```

## Viewer

### Native browser viewer

Run the viewer on the CUDA machine that already has the working `scaffoldgs`
environment and compiled rasterizer:

```bash
conda activate scaffoldgs
pip install -r requirements-viewer.txt
python viewer.py -m "/path/to/trained_run" --gpu 0 --port 8080
```

Open `http://<gpu-server-ip>:8080` in your browser, or `http://localhost:8080`
when the browser and viewer run on the same machine. The default host is
`0.0.0.0`; pass `--host 127.0.0.1` to bind locally. For SSH access, forward the
port from your desktop:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@gpu-server
```

Then open `http://localhost:8080` on that desktop. `-m` is the training output
folder containing `cfg_args` and `point_cloud/`, not a baked PLY. The latest
checkpoint is loaded automatically; use `--iteration 40000` to select one.

The viewer calls the **same native CUDA renderer and neural decoders** used for
Scaffold-GS evaluation. Opacity, color, scale and quaternion are recomputed for
the current camera. Rendering runs on the server GPU and images stream to the
browser. The server loads the native anchor PLY and saved MLPs directly, without
reading the source images or building dataset cameras. Existing trained models
work without retraining. Keep `cfg_args`, the iteration directory's
`point_cloud_scaffold.ply`, `opacity_mlp.pt`, `cov_mlp.pt`, and `color_mlp.pt`
together; models using appearance embeddings or the feature bank also need
their corresponding `.pt` files. Legacy native `point_cloud.ply` is supported.
`cameras.json` supplies saved camera presets and `frame_selection.json`, when
present, maps selected training cameras to their appearance IDs. Without camera
metadata the viewer starts from a scene overview.

Controls:

- Drag to orbit, right-drag to pan, and use the wheel to zoom.
- W/A/S/D move the camera, Q/E move down/up, and arrow keys look around.
- Browse saved camera views, reset the view, fit the scene, or level the horizon.
- Adjust resolution, background, splat scale and appearance-camera index.
- Download a full-resolution PNG of the current view.

Moving frames default to an 800-pixel longest side and JPEG transport. Once the
camera stops, a lossless PNG refreshes the view at up to 1,600 pixels. Adjust
these limits in the sidebar or with `--moving_resolution` and `--max_resolution`.
`--fps 15` caps the requested render rate; achieved speed depends on the model,
GPU and connection. Camera updates are coalesced so inference always starts
from the newest available pose. This provides navigation and inspection, rather
than SuperSplat's editing/export tools or a standalone browser-only renderer.

### SIBR viewer

The [viewer](https://github.com/city-super/Scaffold-GS/tree/main/SIBR_viewers) for Scaffold-GS is available now. 

Recommended dataset structure in the source path location:

```
<location>
|---sparse
    |---0
        |---cameras.bin
        |---images.bin
        |---points3D.bin
```

or

```
<location>
|---points3D.ply
|---transforms.json
```

Recommended checkpoint  structure in the model path location:

```
<location>
|---point_cloud
|   |---point_cloud.ply
|   |---color_mlp.pt
|   |---cov_mlp.pt
|   |---opacity_mlp.pt
(|   |---embedding_appearance.pt)
|---cfg_args
|---cameras.json
(|---input.ply)
```


## Contact

- Tao Lu: taolu@smail.nju.edu.cn
- Mulin Yu: yumulin@pjlab.org.cn

## Citation

If you find our work helpful, please consider citing:

```bibtex
@inproceedings{scaffoldgs,
  title={Scaffold-gs: Structured 3d gaussians for view-adaptive rendering},
  author={Lu, Tao and Yu, Mulin and Xu, Linning and Xiangli, Yuanbo and Wang, Limin and Lin, Dahua and Dai, Bo},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20654--20664},
  year={2024}
}
```

## LICENSE

Please follow the LICENSE of [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting).

## Acknowledgement

We thank all authors from [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting) for presenting such an excellent work.
