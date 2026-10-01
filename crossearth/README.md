# crossearth

[CrossEarth](https://github.com/VisionXLab/CrossEarth) (TPAMI 2025) is
a vision foundation model targeting Remote Sensing Domain
Generalization (RSDG): trained on a set of source domains and used
zero-shot on unseen target domains that differ in region, resolution,
spectral bands, climate, or combinations thereof. It pairs a frozen
DINOv2 ViT backbone with two domain-bridging mechanisms — a data-level
**Earth-Style Injection** pipeline and a model-level **Multi-Task
Training** scheme — and is benchmarked across a curated 32-scenario
RSDG suite spanning regions, sensors, and climates.

This container vendors the upstream `VisionXLab/CrossEarth` repo at a
pinned commit. There is no PyPI release; CrossEarth is a research
codebase that registers its models, heads, and segmentors with mmseg
via `from CrossEarth import *` side effects.

Its ENTRYPOINT is one runnable task, `crossearth-land-cover`: a
six-class land-cover map of an RGB orthophoto (below). The weights for
that task are baked into the image; every other checkpoint downloads
from the Hugging Face Hub on first call.

The land-cover task runs on a GPU when one is visible and on the CPU
otherwise. On the CPU the DINOv2 layers use standard PyTorch attention in
place of xformers, the fallback upstream's layers take when
`XFORMERS_DISABLED` is set; 16 CPU cores took about one second per
512-pixel window. `:v2` passed the lab's GPU probe on an H100
(`VERIFICATION.json`).

## Image tag

`ghcr.io/bradleylab/crossearth:v5` (also `:latest`, `:torch2.1-cu121`).
`:v4` is the same image without the PNG preview; `:v3` has a land-cover
task that needs a GPU; `:v2` has no land-cover task.
Do not use `:v1`: its CUDA 11.7 torch cannot target an H100 and hangs.

## Stack

- Base: `pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime` (amd64 only)
- Python 3.10 (from base)
- PyTorch 2.1.0 + CUDA 12.1
- `mmengine`, `mmcv ==2.1.0`, `mmsegmentation >=1.0.0,<1.3`,
  `mmdet >=3.0.0,<3.4` — mmengine and mmcv via `mim`
- `xformers ==0.0.22.post7`, the build paired with torch 2.1.0
- `numpy <2`, for the torch 2.1 ABI
- `rasterio ==1.4.3` (its wheel bundles GDAL 3.9.3) with `affine ==2.4.0`, and
  `safetensors ==0.8.0`, for the land-cover task
- Vendored CrossEarth at SHA `644a5a1b` (HEAD as of 2026-04-02)
- requirements.txt deps: numpy, ftfy, scipy, prettytable, matplotlib,
  regex, timm, einops

The Dockerfile header explains why the stack moved off upstream's torch
2.0.1 + CUDA 11.7. `PYTHONPATH=/opt/CrossEarth` is set in ENV so
`import CrossEarth` works from any cwd.

## Land-cover task

`crossearth-land-cover` implements the geospatial executor's entrypoint
contract v1 (`fossettlab/geospatial-executor`,
`docs/contracts/entrypoint-v1.md`): three options, the orthophoto staged
under `<input-dir>/primary/`, and every output plus a `run.json` manifest
written under `<output-dir>`. `crossearth_land_cover.cwl` describes the
same task as a CWL v1.2 tool, so a CWL engine that honors the image
ENTRYPOINT (cwltool with Docker, Podman or Apptainer; Toil) can run it.

- **Input:** one 8-bit RGB or RGBA GeoTIFF in a projected CRS in meters.
  Pixels its alpha band or nodata value masks are written as 0.
- **Model:** upstream's inference configuration
  (`configs/_base_/models/CrossEarth_dinov2_mask2former.py`) with the six
  ISPRS classes, and the `Potsdam(r)-source.pth` weights: the model trained
  on the ISPRS Potsdam RGB true orthophotos, whose cells are 5 cm. The raster
  is read in blocks, and each block runs upstream's sliding-window inference
  (512-pixel windows at a 341-pixel stride, overlaps averaged).
- **Parameter:** `resolution`, a cell size in meters (0 to 1) to resample
  the orthophoto onto first, bilinearly; 0 keeps its own cells.
- **Outputs:** `land_cover.tif`, a Cloud Optimized GeoTIFF of class codes
  1–6 (impervious surface, building, low vegetation, tree, car, clutter)
  with a color table; `land_cover_preview.png`, the same map as an RGBA PNG
  in the color table's colors with nodata transparent, at most 1024 pixels
  on its longer side (downsampled by nearest neighbor, never upsampled), for
  a chat interface to show inline; and `land_cover_report.json`, with each
  class's pixel count, area and share, the grid, the preview's size and
  downsampling factor, and the weights' provenance.

```bash
docker run --rm --gpus all \
  -v /path/to/job:/job \
  ghcr.io/bradleylab/crossearth:v5 \
  --input-dir /job/input \
  --output-dir /job/output \
  --params-json /job/params.json
```

with the orthophoto at `/path/to/job/input/primary/<name>.tif` and
`params.json` holding, for example, `{"resolution": "0"}`. Drop
`--gpus all` to run on the CPU.

The model has only been trained on Potsdam, so its classes are the
Potsdam benchmark's; on other scenes, forests, fields or water among
them, its accuracy is unmeasured. On a 50 m square of a 0.109 m drone
orthophoto of Tisch Park, Missouri, it labeled lawn as low vegetation,
trees as trees and paths as impervious surface, and most of a domed
roof as clutter rather than building.

## Weights

The land-cover weights are baked in at
`/opt/weights/crossearth_potsdam_rgb.safetensors`. At build time,
`scripts/convert_checkpoint.py` downloads `Potsdam(r)-source.pth` from
[`Cusyoung/CrossEarth`](https://huggingface.co/Cusyoung/CrossEarth) at a
pinned revision, checks its SHA-256, and writes the backbone and decode-head
tensors to safetensors. The published checkpoint is a pickle, which the
Hub's scanner flags, and besides its tensors it holds the training run's
log as numpy and mmengine objects. The converter never unpickles it
freely: tensors resolve only to the functions PyTorch's own `weights_only`
loader allows, the training log's numpy and mmengine entries become inert
placeholders, and any other global stops the build. Only the tensors of upstream's inference model are kept:
the backbone and the decode head. The checkpoint also holds three modules of
the unreleased training code, which are dropped: a style augmentor, an
auxiliary head, and a convolution stack over the input image stored inside the
backbone's Reins adapter. The released adapter takes the image but never uses
it, and the authors state that only the inference code is released
([issue #14](https://github.com/VisionXLab/CrossEarth/issues/14)). Upstream's
own `tools/test.py` loads checkpoints without strict key matching, so it drops
the same tensors without saying so. Whether the paper's scores were measured
with that image branch in use is not stated. The safetensors file records the
source, its checksum and the dropped modules in its metadata.

The other checkpoints download from the Hub on first call into
`HF_HOME=/opt/hf-cache`; bind-mount it for persistence. The Hub
organization is `Cusyoung`, not the author's GitHub handle `Cuzyoung`.

| Checkpoint | Size (Hub) | Use |
|---|---|---|
| `dinov2_converted.pth` | 1.22 GB | backbone initialization, 512×512 configs |
| `dinov2_converted_1024x1024.pth` | 1.23 GB | backbone initialization, 1024×1024 configs |
| `Potsdam(r)-source.pth` and the other per-benchmark files | 1.68–4.34 GB | trained segmentation models, one per benchmark |

For a shell with the upstream CLI, override the entrypoint:

```bash
docker run --rm -it --gpus all \
  -v /path/to/data:/work \
  -v /shared/hf-cache:/opt/hf-cache \
  --entrypoint bash \
  ghcr.io/bradleylab/crossearth:v5
```

Upstream's `tools/test.py` evaluates a config against a dataset; its
dataset paths and class counts are set in `configs/_base_/datasets/*.py`
and the model config.

## Why this container exists

CrossEarth fills a slot in the catalog that the other RS foundation
models (DOFA, DOFA-CLIP, Prithvi-EO, TerraMind, RemoteCLIP, GeoCLIP)
do not occupy:

- **Domain generalization first**: DOFA generalizes by learning
  spectral conditioning; CrossEarth generalizes by data augmentation
  (Earth-Style Injection) + multi-task heads. Different bet on what
  closes the domain gap.
- **DINOv2 backbone**: the only RS FM in the catalog that builds on a
  generalist self-supervised vision backbone (DINOv2) rather than an
  RS-pretrained backbone. Useful for tasks where the geometric
  features matter more than the spectral specificity.
- **Trained segmentation models**: sister containers ship encoders;
  CrossEarth's per-benchmark checkpoints include a trained Mask2Former
  head, so they segment without fine-tuning.

## License

Code: MIT (this Dockerfile, `VisionXLab/CrossEarth`).
Weights: the [`Cusyoung/CrossEarth`](https://huggingface.co/Cusyoung/CrossEarth)
repository is tagged MIT; its model card is empty. The backbone derives
from Meta's DINOv2 (Apache-2.0). The benchmark datasets the checkpoints
were trained on carry their own terms.

## References

- Gong et al. (2025) "CrossEarth: Geospatial Vision Foundation Model
  for Domain Generalizable Remote Sensing Semantic Segmentation",
  TPAMI 2025 [arXiv:2410.22629](https://arxiv.org/abs/2410.22629).
- Upstream code: [VisionXLab/CrossEarth](https://github.com/VisionXLab/CrossEarth)
  (configs, training scripts, benchmark collection).
- Project page:
  [cuzyoung.github.io/CrossEarth-Homepage/](https://cuzyoung.github.io/CrossEarth-Homepage/)
- ISPRS Potsdam benchmark:
  [isprs.org](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/2d-sem-label-potsdam.aspx)
