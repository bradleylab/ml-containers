# ml-containers

Custom ML Docker images for bradleylab research compute (WashU RIS Compute2, EC2).

For per-image task / sensor / paper / weights / license / lab-status,
see [`MODEL_CARDS.md`](MODEL_CARDS.md).

## Conventions

These rules govern what belongs in this repo and how images are built and
shipped. Full rationale lives in the operator-side rule file
(`~/.claude/rules/ml-containers.md`); this section is the public surface.

**One model per container.** Each container holds a single trained ML/DL
model (or a single classical pipeline that is invoked as if it were a
model — AMS3D, 3DFin). Kitchen-sink images that bundle multiple models
are not allowed by default; the exception clause requires a documented
reason in the image's README.

**The source recipe lives here.** Every container the lab runs must have
its `Dockerfile` (and any build-context files) committed under
`<image-name>/`, with a `README.md` describing the model, base image,
weights handling, and run command. Ad-hoc `docker build` invocations on
a host (Compute2, EC2, Mac) without a committed recipe are not
acceptable.

**GHCR is the publish target.** Push to
`ghcr.io/bradleylab/<image-name>:<tag>` via the per-image GitHub Actions
workflow at `.github/workflows/build-<image-name>.yml`. Never
`docker push` from a laptop or compute node.

**Compute2 `.sqsh` files are caches, not source.** They are produced by
`enroot import 'docker://ghcr.io#bradleylab/<image>:<tag>'` and live on
RIS storage. They can be deleted whenever space is tight; the canonical
recipe + image lives in this repo + GHCR.

**Tag scheme.** `:latest` tracks `main` and is identical to the most
recent stable `:vN`. `:v1`, `:v2`, ... are stable, immutable releases
(do not delete published tags). `:vN-<variant>` for same-base alternate
checkpoints or configs (e.g. `:v2-defaults`). `:deprecated` for images
intentionally being replaced.

**What does NOT belong here.** General-purpose libraries (lidR, GDAL,
PDAL, scikit-learn), generic compute environments ("R + GDAL", "Python
+ CUDA"), notebook shims, and tool wrappers without learned components
(PDAL CLI, gdal-tools). Test: "could a reviewer name a specific model
whose inference this container runs?" If no, use an upstream image
(`rocker/geospatial`, `pytorch/pytorch`, `osgeo/gdal`) or install
locally.

## Images

### Coverage

| GHCR image | Source dir | Status |
|------------|-----------|--------|
| `segment-any-tree-h100` | `segment-any-tree-h100/` | full recipe + 2 Dockerfiles (v2 + v2-defaults) |
| `ams3d-crownseg` | `ams3d-crownseg/` | full recipe |
| `fsct` | `fsct/` | full recipe |
| `sam2` | `sam2/` | full recipe |
| `treelearn` | `treelearn/` | full recipe |
| `pointstowood` | `pointstowood/` | full recipe |
| `3dfin` | `3dfin/` | full recipe |
| `backman-thermal-deer` | `backman-thermal-deer/` | full recipe (runtime-only; model bind-mounted) |
| `deepforest` | `deepforest/` | full recipe (NEON checkpoint via HF Hub) |
| `forainet` | `forainet/` | full recipe — **experimental** torch 2.2 / sm_90 port |
| `seisbench` | `seisbench/` | full recipe (CPU; weights via model zoo at runtime) |
| `neuralhydrology` | `neuralhydrology/` | full recipe (CPU; user-supplied checkpoint + forcings) |
| `remoteclip` | `remoteclip/` | full recipe (CPU; OpenCLIP arch + HF Hub weights at runtime) |
| `satlas` | `satlas/` | full recipe (GPU sm_90; SatlasPretrain backbones, runtime fetch from HF Hub) |
| `clay` | `clay/` | full recipe (GPU sm_90; ViT-MAE foundation model, runtime fetch from HF Hub) |
| `xrd-classifier` | `xrd-classifier/` | full recipe (CPU; autoXRD multi-phase ID; demo Li-Mn-Ti-O-F model baked in) |
| `prithvi-eo` | `prithvi-eo/` | full recipe (GPU sm_90; IBM/NASA HLS foundation model via TerraTorch, runtime fetch from HF Hub) |
| `multispec-species` | — | deleted (failed boundary test); see [`DEPRECATIONS.md`](DEPRECATIONS.md) |
| `tree-analysis` | — | deleted (kitchen-sink); see [`DEPRECATIONS.md`](DEPRECATIONS.md) |

### segment-any-tree-h100

SegmentAnyTree individual tree segmentation rebuilt for H100 GPUs (sm_90).

- Base: PyTorch 2.2.2 + CUDA 12.1
- MinkowskiEngine from [CiSong10/cuda12-installation](https://github.com/CiSong10/MinkowskiEngine/tree/cuda12-installation) fork
- Includes all patches for CUDA 12 compatibility (thrust namespace, NVTX3, std::to_address)

Two variants share the same CUDA / PyTorch / dependency stack — they
differ only in the checkpoint's `run_config` values:

| Tag | Dockerfile | Use case |
|-----|------------|---------|
| `:v2` (also `:latest`, `:cuda12.1-torch2.2`) | `Dockerfile` | Checkpoint patched with UAV-tuned clustering (`block_merge_th=0.3, cluster_radius_search=0.5, cluster_type=1, bandwidth=0.6`). |
| `:v2-defaults` (also `:cuda12.1-torch2.2-defaults`) | `Dockerfile.defaults` | Checkpoint kept exactly as shipped by SmartForest-no/SegmentAnyTree (Wielgosz et al. 2024 defaults). Use for new sensor modalities (e.g. TLS) or for "out-of-the-box DL baseline" comparisons. |

Pull:
```
ghcr.io/bradleylab/segment-any-tree-h100:v2
ghcr.io/bradleylab/segment-any-tree-h100:v2-defaults
```

Built automatically via GitHub Actions on push to
`segment-any-tree-h100/` (both workflows).

### ams3d-crownseg

AMS3D (Adaptive Mean-Shift 3D) via `crownsegmentr` on top of
`rocker/geospatial`. Classical method for UAV crown segmentation.

Pull: `ghcr.io/bradleylab/ams3d-crownseg:v1`

### fsct

Forest Structural Complexity Tool (Krisanski et al. 2021) — TLS / MLS
tree segmentation. CPU-only (FSCT pins torch 1.9 + CUDA 11.1 which
don't run on H100; porting is not worth the effort for our use case).

Pull: `ghcr.io/bradleylab/fsct:v1`

Run on Compute2 `general-cpu` partition.

### sam2

[SAM 2 / SAM 2.1](https://github.com/facebookresearch/sam2) — Meta's
Segment Anything Model 2 wrapped as a portable inference CLI. One
entrypoint takes any RGB image and produces a JSON file of COCO-RLE
masks; supports automatic mask generation, point prompts, and box
prompts. Weights download from HF Hub on first run and cache under
`$HF_HOME` (override-bindable for persistence).

- Base: PyTorch 2.5.1 + CUDA 12.1 (runtime variant)
- `SAM2_BUILD_CUDA=0` to skip the optional connected-components CUDA
  extension (needs `nvcc` only present in `-devel` images; pure-Python
  fallback is automatic and fine for our paths).
- `TORCH_CUDA_ARCH_LIST` includes `9.0` for H100.
- Default model: `facebook/sam2.1-hiera-large`. Override with
  `--model-id` for any of the other Hiera variants (tiny / small /
  base-plus / large).

Pull: `ghcr.io/bradleylab/sam2:latest`

Bundles `segment-geospatial` (`samgeo`) so the same image handles
geospatial inference end-to-end — read a GeoTIFF, run tiled SAM 2,
emit georeferenced polygons.

See `sam2/README.md` for full CLI docs and Compute2 / Apptainer usage examples.

### treelearn

TreeLearn (Henrich et al. 2024, *Ecol. Informatics*) — DL instance
segmentation of trees from ground-based lidar (TLS/MLS). PyTorch
2.0 + CUDA 11.8 + spconv-cu118; native sm_90.

Pull: `ghcr.io/bradleylab/treelearn:v1`

Weights are NOT baked — fetched at runtime via the bundled
`download_weights.sh` (Göttingen dataverse is too flaky for
build-time fetch). See `treelearn/README.md`.

### pointstowood

PointsToWood (Owen et al. 2025, *arXiv:2503.04420*) — DL semantic
leaf-wood segmentation of high-resolution TLS point clouds. PyTorch
2.5 + CUDA 12.1 + PyG ecosystem; native sm_90.

Pull: `ghcr.io/bradleylab/pointstowood:v1`

The `global.pth` checkpoint ships in-tree with the upstream
`version1.0-paper` branch and is verified loadable at build time.
See `pointstowood/README.md`.

### 3dfin

3DFin (Laino et al. 2024, *Forestry*) — deterministic TLS stem
detection + DBH estimation. CPU-only classical algorithm; runs on
Compute2 `general-cpu`.

Pull: `ghcr.io/bradleylab/3dfin:v1`

See `3dfin/README.md` for the wrapper script and run pattern.

### backman-thermal-deer

Backman et al. 2025 thermal animal detector — ONNX recurrent (LSTM)
model for frame-level animal detection in 640×512 thermal video
(DJI XT2). CPU runtime container; the upstream
`inferenceExample/` directory (model.onnx + generateVideoPredictions.py)
is bind-mounted at runtime rather than baked, since redistribution
rights are not established.

Pull: `ghcr.io/bradleylab/backman-thermal-deer:v1`

See `backman-thermal-deer/README.md` for the bind-mount pattern.

### deepforest

DeepForest 2.x (Weinstein et al.) — RetinaNet-style aerial RGB
tree-crown detector, NEON-pretrained via Hugging Face Hub. PyTorch
2.5 + CUDA 12.1; native sm_90.

Pull: `ghcr.io/bradleylab/deepforest:v1`

The `weecology/deepforest-tree` checkpoint downloads to
`$HF_HOME=/opt/hf-cache` on first call — bind-mount a persistent
host dir to avoid re-downloading per job.

### forainet

> **EXPERIMENTAL.** First end-to-end run scheduled 2026-05-01.

ForAINet (Xiang et al., ETH PRS) — panoptic segmentation of
airborne lidar via PointGroup-style architecture. Upstream targets
PyTorch 1.9 / CUDA 11.1 (no sm_90 support); this container ports
the stack to PyTorch 2.2 / CUDA 12.1 by reusing the H100-proven
recipe from `segment-any-tree-h100` (CiSong10 MinkowskiEngine fork
+ torchsparse 1.4 patches + torch_points3d PyG-2.x compat).

Pull: `ghcr.io/bradleylab/forainet:v1`

`PointGroup-PAPER.pt` distributed by upstream via Dropbox under
unclear license — bind-mount at runtime. See `forainet/README.md`
for the fetch command and the experimental-status caveat.

### seisbench

SeisBench (Woollam et al. 2022) — toolbox bundling maintained
PhaseNet (Zhu & Beroza 2019) and EQTransformer (Mousavi et al. 2020)
plus the public model zoo. Zero-shot on new stations and networks.

- Base: `python:3.11-slim`
- PyTorch 2.5.1 (CPU wheels)
- `seisbench >= 0.7`, `obspy >= 1.4`

Pull: `ghcr.io/bradleylab/seisbench:v1`

CPU-only by design — the model is fast on CPU for typical use; on
continental catalogs the bottleneck is I/O parallelism, not model
compute. A CUDA variant can be added later if needed.

Weights are NOT baked. The first call to `Model.from_pretrained(...)`
fetches from the SeisBench S3 model zoo into
`$SEISBENCH_CACHE_ROOT=/opt/seisbench-cache`; bind-mount a persistent
host dir there. See `seisbench/README.md`.

### neuralhydrology

NeuralHydrology (Kratzert, Klotz, Gauch et al.) — Python library for
training and running deep-learning hydrology models (LSTM
rainfall-runoff, streamflow prediction). Pretrained CAMELS LSTMs are
tiny; CPU inference is essentially instant.

- Base: `python:3.11-slim`
- PyTorch 2.5.1 (CPU wheels)
- `neuralhydrology >= 1.13`
- Console scripts: `nh-run`, `nh-schedule-runs`, `nh-results-ensemble`

Pull: `ghcr.io/bradleylab/neuralhydrology:v1`

CPU-only by design — this image targets *inference*. Training
(continental-scale, ensembles) is the GPU-bound case and would need
a separate CUDA variant; not on the roadmap until a use case lands.

Weights are NOT baked — users bind-mount a `run_dir/` containing the
trained checkpoint and `config.yml`. Pretrained CAMELS checkpoints
are linked from the [NeuralHydrology research blog](https://neuralhydrology.github.io/).
See `neuralhydrology/README.md`.

### remoteclip

RemoteCLIP (Liu, Chen et al. 2024, IEEE TGRS) — CLIP architecture
fine-tuned on a 12× larger remote-sensing pre-training corpus. Three
OpenCLIP-format checkpoints are distributed via Hugging Face Hub:
`RN50`, `ViT-B-32`, `ViT-L-14`.

- Base: `python:3.11-slim`
- PyTorch 2.5.1 + torchvision (CPU wheels)
- `open-clip-torch >= 2.20`, `huggingface_hub >= 0.25`

Pull: `ghcr.io/bradleylab/remoteclip:v1`

CPU-only by design — CLIP-sized models are very fast on CPU. For
batch embedding across large tile archives a CUDA variant would help;
not on the roadmap until that workload lands.

Weights are NOT baked — fetched at runtime via `hf_hub_download` from
[`chendelong/RemoteCLIP`](https://huggingface.co/chendelong/RemoteCLIP)
into `$HF_HOME=/opt/hf-cache`. See `remoteclip/README.md`.

### satlas

SatlasPretrain (Bastani et al., ICCV 2023) — Allen AI's pre-trained
foundation model backbones for satellite + aerial imagery. Sentinel-2
RGB+MS, Sentinel-1 SAR, Landsat 8/9 all-bands, and 0.5–2 m/px aerial
RGB; Swin-v2-{Base,Tiny} and ResNet{50,152} variants.

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11 + PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `satlaspretrain-models >= 0.3.1`

Pull: `ghcr.io/bradleylab/satlas:v1`

GPU-primary (H100 sm_90); CPU also works via `device='cpu'`.

Weights are NOT baked. `Weights().get_pretrained_model(checkpoint_id)`
fetches the `.pth` from `allenai/satlas-pretrain` on HF Hub at every
call — the upstream loader does **not** cache on disk. For repeated
jobs, pre-download checkpoints to a host directory; see
`satlas/README.md` for the wrapper pattern.

### clay

Clay Foundation Model — Vision-Transformer Masked Autoencoder
pretrained on multi-sensor Earth observation imagery (Sentinel-2,
Sentinel-1 SAR, Landsat, NAIP, MODIS). Outputs per-patch embeddings
usable for similarity search, clustering, or lightweight downstream
classification with minimal labels.

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11 + PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `claymodel==1.5.0` (installed from a pinned upstream git commit)
- Lightning, geopandas, timm, vit-pytorch, scikit-image, einops, etc.

Pull: `ghcr.io/bradleylab/clay:v1`

GPU-primary (H100 sm_90); single-tile embedding works on a laptop
GPU, but batch embedding across an archive of tiles is the killer
use case.

Weights are NOT baked. The v1.5 checkpoint (~3 GB) downloads from
[`made-with-clay/Clay`](https://huggingface.co/made-with-clay/Clay)
into `$HF_HOME=/opt/hf-cache` on first call to `hf_hub_download`.
Clay's input contract is non-trivial (multi-sensor datacubes with
time / lat-lon / GSD / wavelength metadata) — see `clay/README.md`
for the data-prep pointers and the upstream wall-to-wall tutorial.

### xrd-classifier

autoXRD (Szymanski et al. 2021, *Chem. Mater.*) — probabilistic
multi-phase identification from 1D powder XRD patterns. 1D CNN
trained on simulated patterns from a user-supplied CIF library with
physics-informed augmentation; multi-phase by design (default cap
of 3 phases per pattern).

- Base: `python:3.11-slim`
- TensorFlow >= 2.16 (CPU)
- `autoXRD >= 0.0.6` (PyPI)
- pymatgen, pyxtal (transitive)

Pull: `ghcr.io/bradleylab/xrd-classifier:v1`

CPU-only by design — autoXRD inference is ~10 s/pattern on CPU and
catalog-scale phase ID parallelises on `general-cpu` job arrays.

Bundled demo: the upstream Li-Mn-Ti-O-F battery-cathode model
(`Example/Model.h5`, ~73 MB) is baked in so a smoke run works on
first pull. **For arbitrary minerals or other chemistries you must
retrain** — drop CIFs into `Novel-Space/All_CIFs` and run the
`generate_References.py` → `generate_XRD.py` → `train_CNN.py`
pipeline (CPU-tractable for small phase libraries). See
`xrd-classifier/README.md` for the full retraining procedure.

### prithvi-eo

[Prithvi-EO](https://huggingface.co/ibm-nasa-geospatial) is the
IBM/NASA family of ViT-based geospatial foundation models pre-trained
on Harmonized Landsat-Sentinel-2 (HLS) imagery. Variants: 1.0-100M,
2.0-300M, 2.0-300M-TL, 2.0-600M, 2.0-600M-TL (TL = temporal +
locational embeddings).

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11 + PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `terratorch >= 1.2.5` (IBM toolkit; brings Lightning, torchgeo,
  segmentation-models-pytorch, diffusers, timm as transitive deps)

Pull: `ghcr.io/bradleylab/prithvi-eo:v1`

GPU-primary (H100 sm_90); 100M v1 fine-tuned variants run on a laptop
GPU with 8+ GB VRAM, but 300M / 600M v2 base models want H100.

Weights are NOT baked. The five Prithvi variants live at
`ibm-nasa-geospatial/Prithvi-EO-*` on HF Hub and download into
`$HF_HOME=/opt/hf-cache` on first call to TerraTorch's
`BACKBONE_REGISTRY.build(..., pretrained=True)`. See
`prithvi-eo/README.md` for inference + fine-tuning patterns and the
HLS data-prep references.
