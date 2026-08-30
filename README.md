# ml-containers

Custom ML Docker images for bradleylab research compute (WashU RIS Compute2, EC2).

For per-image task / sensor / paper / weights / license / lab-status,
see [`MODEL_CARDS.md`](MODEL_CARDS.md).

**Publishing a license anywhere else? Read it from
[`LICENSE_MANIFEST.json`](LICENSE_MANIFEST.json), not from a Dockerfile.**
That file is generated from the `org.opencontainers.image.licenses` and
`bradleylab.model.licence.note` labels by `scripts/license_manifest.py`, and CI
fails if it falls behind them. Every license error that has reached the lab
website so far was a label copied by hand and then left behind when the
Dockerfile moved on. Composite licenses (`Apache-2.0 AND CC-BY-NC-4.0`) are
recorded verbatim — publish them whole; dropping the second term drops the
binding constraint.

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
| `forestformer3d` | `forestformer3d/` | full recipe — **experimental** Plan B build (torch 1.13 + cu118 + sm_90); Plan A fallback documented |
| `seisbench` | `seisbench/` | full recipe (CPU; weights via model zoo at runtime) |
| `neuralhydrology` | `neuralhydrology/` | full recipe (CPU; user-supplied checkpoint + forcings) |
| `remoteclip` | `remoteclip/` | full recipe (CPU; OpenCLIP arch + HF Hub weights at runtime) |
| `satlas` | `satlas/` | full recipe (GPU sm_90; SatlasPretrain backbones, runtime fetch from HF Hub) |
| `clay` | `clay/` | full recipe (GPU sm_90; ViT-MAE foundation model, runtime fetch from HF Hub) |
| `dinov3-sat` | `dinov3-sat/` | full recipe (GPU sm_90; DINOv3 SAT-493M ViT-L/16 frozen dense-feature encoder for RGB aerial/satellite ortho; weights baked; DINOv3 License — not OSS) |
| `xrd-classifier` | `xrd-classifier/` | full recipe (CPU; autoXRD multi-phase ID; demo Li-Mn-Ti-O-F model baked in) |
| `prithvi-eo` | `prithvi-eo/` | full recipe (GPU sm_90; IBM/NASA HLS foundation model via TerraTorch, runtime fetch from HF Hub) |
| `treex` | `treex/` | full recipe (CPU; Burmeister et al. 2025 unsupervised tree-instance segmentation; classical, no weights) |
| `raman-classifier` | `raman-classifier/` | full recipe (CPU; ramanspy + RRUFF excellent_unoriented baked at build time; classical NN matcher, no learned weights) |
| `geoclip` | `geoclip/` | full recipe (CPU; Vivanco Cepeda et al. 2023 worldwide image geolocalization; CLIP-L/14 weights baked at build time, offline runtime) |
| `dofa` | `dofa/` | full recipe (CPU; Xiong et al. 2024 multispectral/SAR/optical foundation model; Base weights baked at build, Large lazy) |
| `dofa-clip` | `dofa-clip/` | full recipe (CPU; Xiong et al. 2025 multispectral CLIP via vendored open_clip fork; so400m-384-EO baked at build; **CC-BY-NC-4.0 — non-commercial only**) |
| `terramind` | `terramind/` | full recipe (GPU sm_90; Jakubik et al. 2025 IBM/ESA any-to-any generative EO foundation model — S1+S2+DEM+NDVI+LULC; tiny/small/base/large via TerraTorch + diffusers 0.30 pin; weights via HF Hub) |
| `timesfm` | `timesfm/` | full recipe (CPU multi-arch; Das et al. 2024 Google Research time-series foundation model — TimesFM 2.5 200M params, 16k context, continuous-quantile head; weights via HF Hub) |
| `crossearth` | `crossearth/` | full recipe (GPU; Gong et al. 2025 TPAMI vision FM for cross-domain RS semantic segmentation — frozen DINOv2 + Earth-Style Injection + Mask2Former head; mmcv 2.x + mmseg 1.x + xformers 0.0.20; vendored upstream at SHA `644a5a1b`; weights via HF Hub) |
| `croma` | `croma/` | full recipe (CPU multi-arch; Fuller et al. 2023 NeurIPS Sentinel-1/Sentinel-2-native radar-optical foundation model; SAR + optical + joint embeddings; upstream `use_croma.py` pinned; Base baked at build, Large lazy; MIT) |
| `evo2` | `evo2/` | full recipe — **experimental**, backfilled from the published image (GPU sm_90; Arc Institute DNA language model; NGC PyTorch 25.04; weights via HF Hub) |
| `esm` | `esm/` | full recipe — **experimental** (GPU sm_90; Chan Zuckerberg Biohub ESMC protein language model; NGC PyTorch 25.04; both git deps SHA-pinned; weights via HF Hub, MIT + ungated) |
| `clean` | `clean/` | full recipe — **experimental** (**CPU-only**; CLEAN enzyme EC-number prediction over ESM-1b embeddings; weights mounted at runtime, not baked; **research-use-only licence, not MIT**) |
| `saprot` | `saprot/` | full recipe — **experimental** (GPU sm_90; SaProt structure-aware protein language model; bundles the foldseek binary, so the image is **MIT AND GPL-3.0**; weights via HF Hub, MIT + ungated) |
| `boltz` | `boltz/` | full recipe — **experimental** (GPU sm_90; Boltz-2 biomolecular complex structure + binding affinity; MIT code *and* weights; weights via HF Hub / `model-gateway.boltz.bio`) |
| `chai-1` | `chai-1/` | full recipe — **experimental** (GPU sm_90, A100/H100 80 GB; Chai-1 co-folding, MSA-free by default; Apache-2.0 code *and* weights since Nov 2024; weights via `chaiassets.com`) |
| `dnabert-s` | `dnabert-s/` | full recipe — **experimental** (GPU sm_90; species-aware DNA embeddings for metagenomic binning; `transformers==4.27` pin dictates the whole stack; Apache-2.0 weights) |
| `ntv3` | `ntv3/` | full recipe — **experimental** (GPU sm_90; Nucleotide Transformer v3 — 1 Mb context, ~16k functional tracks; NGC PyTorch 25.04; **HF-gated, non-commercial weights**) |
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

### forestformer3d

> **EXPERIMENTAL.** Plan B build (lowest deviation from upstream).

ForestFormer3D (Xiang et al., ICCV 2025 Oral, [arXiv:2506.16991](https://arxiv.org/abs/2506.16991))
— transformer-panoptic 3D forest instance segmentation built on
OneFormer3D, fine-tuned on FOR-instanceV2 (extends FOR-instance with
TU_WIEN deciduous alluvial leaf-off + BlueCat broadleaf temperate).
Replaces PointGroup-style clustering with learned instance queries,
removing the post-hoc clustering parameters that complicate
SegmentAnyTree tuning.

- Base: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04` + pip torch 1.13.1+cu117
- MinkowskiEngine NVIDIA @ 02fc608, rebuilt with `TORCH_CUDA_ARCH_LIST="...9.0"` for native H100
- spconv-cu118 2.3.6 + cumm-cu118 0.4.11 (sm_90 PTX baked into wheel)
- mmengine 0.7.3 / mmcv 2.0.0 / mmdet 3.0.0 / mmsegmentation 1.0.0 / mmdet3d @ 22aaa47 — exact upstream pins
- `replace_mmdetection_files/` overlay (3,515 lines) applied at build time

Pull: `ghcr.io/bradleylab/forestformer3d:v1`

Weights are NOT baked. The pretrained `epoch_3000_fix.pth` ships
inside `clean_forestformer.zip` on
[Zenodo record 16742708](https://zenodo.org/records/16742708)
(~198 MB). License: CC BY-NC 4.0 (inherited from OneFormer3D base);
academic use OK. Use the bundled `download_weights.sh` for a
runtime fetch with retry/backoff and md5 verification — see
`forestformer3d/README.md`.

If Plan B's MinkowskiEngine fails to compile on sm_90 in CI, the
documented Plan A fallback bumps the entire mm-stack onto torch 2.2 +
cu121 (mirroring `segment-any-tree-h100`). See `forestformer3d/README.md`
§"Plan A fallback".

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

### treex

[treeX](https://doi.org/10.48550/arXiv.2509.03633) (Burmeister et al.
2025, arXiv:2509.03633) — re-engineered classical / unsupervised tree
instance segmentation in dense forest point clouds. Multi-platform
(TLS / PLS / ULS), evaluated on Wytham Woods + FOR-instance with a
reported ULS F1 of 0.58. Provides an alternative classical baseline
to `ams3d-crownseg` (crown-only) for UAV lidar, and to `3dfin`
(stem + DBH only) for TLS.

- Base: `python:3.11-slim`
- Upstream package: `pointtree` from
  [`ai4trees/pointtree`](https://github.com/ai4trees/pointtree)
- Compiled C++ extensions (pybind11 + scikit-build-core) build at
  install time; no GPU.
- **CPU-only**, classical, no shipped weights.

Pull: `ghcr.io/bradleylab/treex:v1`

Only the `TreeXAlgorithm` (unsupervised) path is supported. The
companion `CoarseToFineAlgorithm` from the same package needs torch +
torch-scatter and a learned semantic-segmentation checkpoint; that
would require a separate, much larger container variant. See
`treex/README.md` for the wrapper script, Compute2 enroot/pyxis
launch pattern, and known caveats (notably the modest ULS F1 in the
upstream evaluation).

### raman-classifier

Path A of the long-deferred Raman mineral classifier slot. RRUFF
nearest-neighbour matching via [ramanspy](https://doi.org/10.1021/acs.analchem.4c00383)
(Georgiev et al. 2024, *Anal. Chem.*) — no learned weights,
deterministic, defensible methodology. The `excellent_unoriented`
RRUFF archive is pulled at build time, preprocessed (Whitaker-Hayes
despike → SavGol denoise → ASLS baseline → vector L2-normalisation),
resampled onto a 100-1500 cm⁻¹ fingerprint grid, and baked as a
single ~30-50 MB numpy index at `/opt/rruff_index.npz` so runtime
matching is sub-second cosine over a small npz.

- Base: `python:3.11-slim`
- Stack: `numpy + scipy + ramanspy>=0.2`
- **CPU-only**, classical, no shipped weights.
- Reference data citation: Lafuente B, Downs RT, Yang H, Stone N
  (2015), *The power of databases: the RRUFF project*, in
  *Highlights in Mineralogical Crystallography*, De Gruyter, 1-30.

Pull: `ghcr.io/bradleylab/raman-classifier:v1`

Path B (Liu-2017-style 1D-CNN trained on RRUFF, weights deposited at
Zenodo + HF Hub under Apache-2) remains queued as a follow-up; this
container's preprocessing + index format is the planned inference
harness for those weights when they exist.

### geoclip

[GeoCLIP](https://arxiv.org/abs/2309.16020) (Vivanco Cepeda, Nayak,
Shah, NeurIPS 2023) — worldwide image geolocalization via CLIP-style
alignment between RGB photo embeddings and a learned location
encoder over MP-16 (~4.7M global geo-tagged photos). Given an image,
returns top-k predicted `(lat, lon)` locations and probabilities.

- Base: `python:3.11-slim`
- Stack: torch 2.5.1 CPU + torchvision 0.20.1 + `geoclip>=1.2`
- **CPU-only**, MIT licensed.
- **Weights baked at build time** (~900 MB CLIP-L/14 + location
  encoder + 100K-point GPS gallery). Different from `remoteclip` —
  pragmatic choice for one-shot photo QA where avoiding the
  first-run weight download matters.

Pull: `ghcr.io/bradleylab/geoclip:v1`

Use cases: geo-tagged dataset QA (detect implausible EXIF GPS),
locating photos with stripped EXIF, provenance / dedup. Not
designed for sub-meter accuracy — typical resolution is country /
continent at ~1 km tolerance for street-level scenes.

### dofa

[DOFA](https://arxiv.org/abs/2403.15356) (Dynamic One-For-All;
Xiong et al. 2024) — multispectral / SAR / optical / hyperspectral
foundation model with a wavelength-conditioning hypernetwork. A
single ViT backbone adaptable to arbitrary spectral configurations
via a per-band wavelength input. Trained with masked image modelling
on SatlasPretrain + Five-Billion-Pixels + HySpecNet-11k.

- Base: `python:3.11-slim`
- Stack: torch 2.5.1 CPU + torchvision 0.20.1 + `torchgeo>=0.6` + `timm>=1.0`
- **CPU-only**, CC-BY-4.0 weights.
- Base weights baked at build (445 MB, 768-D embeddings).
  Large (1.35 GB, 1024-D) is fetched lazily via `--variant large`.

Pull: `ghcr.io/bradleylab/dofa:v1`

Embedding-only — no task head. Downstream classification / change
detection / retrieval requires a small head trained on top. For the
text-aligned variant see `bradleylab/dofa-clip` (separate container,
CC-BY-NC-4.0).

### dofa-clip

[DOFA-CLIP](https://arxiv.org/abs/2503.06312) (Xiong et al. 2025) —
multispectral CLIP via DOFA's wavelength-conditioned image encoder
+ SigLIP-style text alignment. Pretrained on **GeoLangBind-2M**
(~2M EO image-caption pairs). The headline differentiator from
`bradleylab/remoteclip` is multispectral input — DOFA-CLIP accepts
arbitrary band counts via per-band wavelength tensors.

> **⚠ License: weights are CC-BY-NC-4.0.** The only NC-licensed
> image in the catalog. Use is restricted to non-commercial purposes;
> commercial users must obtain explicit permission from the upstream
> authors (`xiongzhitong@gmail.com`). Container code is Apache-2.0;
> only the trained checkpoint carries the NC restriction. See
> `dofa-clip/README.md` for the full compliance discussion.

- Base: `python:3.11-slim`
- Stack: torch 2.5.1 CPU + vendored open_clip from `xiong-zhitong/DOFA-CLIP` + `timm` + `einops` + `transformers<5`
- **CPU-only**, multispectral + RGB.
- so400m-384-EO weights baked at build (~1.7 GB).
- Output dim 1152, image resolution 384×384, SigLIP scoring
  (sigmoid per-prompt, not softmax across panel).

Pull: `ghcr.io/bradleylab/dofa-clip:v1`

**Path B build.** The HF transformers Path A
(`BiliSakura/DOFA-CLIP-{ViT-B-16,VIT-L-14}` mirrors) was evaluated
and is broken — text encoder self-attention stored as `in_proj.*`
is silently dropped by HF's `CLIPModel`, leaving every text
attention layer randomly initialized and text embeddings collapsed.
The vendored open_clip fork accepts the open_clip / SigLIP weight
naming directly. See `dofa-clip/README.md` for details.

For permissive multispectral embeddings (no text), use
`bradleylab/dofa` (CC-BY-4.0). For permissive RGB CLIP, use
`bradleylab/remoteclip` (Apache-2.0).

### dinov3-sat

[DINOv3](https://github.com/facebookresearch/dinov3) (Simeoni, Vo, Oquab et
al., 2025) in its **SAT-493M** variant — pretrained on 493 million 512x512
Maxar RGB ortho tiles at 0.6 m GSD, so it is an aerial/satellite prior rather
than a web-image one. Frozen encoder only; emits dense patch tokens for a
downstream head.

- Base: `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (matches `sam2`)
- Stack: `timm==1.0.28`; ViT-L/16, 1024-D
- amd64 only (the CUDA base has no arm64 build)
- Weights baked at build from the **ungated** `timm/` mirror, so enroot
  images are self-contained on offline compute nodes
  (`facebook/dinov3-*` is manually gated; same weights, same licence)

Pull: `ghcr.io/bradleylab/dinov3-sat:v1`

Three traps, all documented in `dinov3-sat/README.md`: the SAT weights use
their **own** normalization (not ImageNet) and wrong stats degrade features
silently; there are **5 prefix tokens** (CLS + 4 registers), so a 512 px input
returns 1029 tokens and a naive reshape scrambles the spatial grid; and the
0.6 m pretraining GSD may be far from yours, with no input size satisfying both
a fine patch grid and the pretraining scale.

**Licence: DINOv3 License, not open source.** Redistribution is permitted only
with a copy of the Agreement — shipped at `/opt/licenses/LICENSE.dinov3.md`,
inherited by anything built `FROM` this image; do not delete it. Publications
reporting results from this model **must acknowledge DINO Materials**.
Different from `remoteclip` (Apache-2.0) and `croma` (MIT).

### croma

[CROMA](https://arxiv.org/abs/2311.00566) (Contrastive Radar-Optical
Masked Autoencoders; Fuller, Millard & Green, NeurIPS 2023) — a
Sentinel-1/Sentinel-2-native foundation model with a SAR encoder, an
optical encoder, and a cross-modal joint encoder. Pretrained with
combined contrastive + masked-autoencoding objectives.

- Base: `python:3.11-slim`
- Stack: torch 2.5.1 CPU + `einops` (no torchvision); upstream
  `use_croma.py` pinned at commit `59505a6`
- **CPU multi-arch** (amd64 + arm64), MIT code + weights.
- Base weights baked at build (ViT-B, 768-D embeddings). Large
  (ViT-L, 1024-D) is fetched lazily via `--variant large`.

Pull: `ghcr.io/bradleylab/croma:v1`

Sentinel-1 (2-ch VV/VH) + Sentinel-2 (12-band, cirrus removed) only —
fixed channel layout, no wavelength conditioning (cf. `bradleylab/dofa`).
Embedding-only; train a downstream head (the `nc-helene-sar` Stage-3 use).

### terramind

[TerraMind 1.0](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
(IBM, ESA Φ-lab, FAST-EO; ICCV 2025) is the first any-to-any
generative foundation model for Earth Observation. Pretrained on
Sentinel-1 GRD, Sentinel-1 RTC, Sentinel-2 L2A, DEM, NDVI, and LULC.
Embeddings + segmentation + cross-modality generation
("Thinking-in-Modalities").

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Stack: torch 2.5.1 cu121 + `terratorch>=1.2.5` + `diffusers==0.30.0`
- **GPU primary**, Apache-2.0 across code + weights.
- Four scale variants on HF Hub: tiny / small / base / large
  (`ibm-esa-geospatial/TerraMind-1.0-{...}`). Weights NOT baked —
  TerraTorch fetches via HF Hub on first call.

Pull: `ghcr.io/bradleylab/terramind:v1`

Sister to `bradleylab/prithvi-eo` (also TerraTorch-fronted). Pick
TerraMind when the workflow needs S1+S2 fusion or any-to-any
modality translation; pick Prithvi-EO when it's HLS-only and the
existing 100M / 300M / 600M weights are appropriate.

### timesfm

[TimesFM 2.5](https://huggingface.co/google/timesfm-2.5-200m-pytorch)
(Das et al., ICML 2024) is Google Research's decoder-only time-series
foundation model. The 2.5 release (Sept 2025) is 200M parameters,
supports up to 16k context, and ships an optional 30M
continuous-quantile head for probabilistic forecasts.

- Base: `python:3.11-slim`
- Stack: torch 2.5.1 CPU + `timesfm` from upstream GitHub (pinned SHA)
- **CPU-primary**, multi-arch (`linux/amd64`, `linux/arm64`).
- Apache-2.0 across code AND weights.
- Weights baked: NO — pulled lazily from HF Hub on first call.

Pull: `ghcr.io/bradleylab/timesfm:v1`

Different modality from the rest of the catalog — operates on
univariate time-series (any 1D regularly-sampled signal), not
imagery. Lab use cases: streamflow / hydrology forecasting,
climate-reanalysis pixel-time-series, eddy-covariance and
soil-moisture gap-filling. Sister to
`bradleylab/neuralhydrology` for time-series workflows; TimesFM is
the zero-shot fallback when there isn't enough history to fine-tune
a CAMELS-style LSTM.

### crossearth

[CrossEarth](https://github.com/VisionXLab/CrossEarth) (Gong et al.,
TPAMI 2025) is a vision foundation model for Remote Sensing Domain
Generalization (RSDG): trained on a set of source domains and used
zero-shot on unseen target domains differing in region, resolution,
spectral bands, or climate. Pairs a frozen DINOv2 ViT backbone with
an Earth-Style Injection augmentation pipeline + multi-task training
over a 32-scenario RSDG benchmark.

- Base: `pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime` (amd64-only)
- Stack: torch 2.0.1 + cuda 11.7 + mmengine + mmcv 2.x + mmsegmentation 1.x + mmdet 3.x + xformers 0.0.20 (exact pin per upstream)
- **GPU-primary**, amd64-only.
- MIT (code); per HF model card (weights).
- Weights baked: NO — pulled from HF Hub `Cusyoung/CrossEarth` on first call.
- Vendored upstream at SHA `644a5a1b3c01b2e5531820b5291d4397597f75de`.

Pull: `ghcr.io/bradleylab/crossearth:v1`

Sister to `dofa`/`dofa-clip`/`terramind` in the RS foundation-model
cluster, but bets differently on the domain-generalization problem:
instead of spectral conditioning (DOFA) or multimodal pretraining
(TerraMind), CrossEarth uses data-level Earth-Style Injection +
multi-task training. Only RS FM in the catalog built on a generalist
self-supervised vision backbone (DINOv2). Ships encoder + Mask2Former
segmentation head, ready for inference without head fine-tuning.

### evo2

[Evo 2](https://github.com/ArcInstitute/evo2) (Arc Institute) — DNA
language model (StripedHyena 2) for variant-effect prediction, sequence
scoring (log-likelihoods), and genome generation across all domains of
life. Different modality from the imagery and point-cloud images: the
input is a nucleotide sequence, not a sensor product.

> **EXPERIMENTAL / backfilled recipe.** Reproduces a previously ad-hoc
> `ghcr.io/bradleylab/evo2` image that had no committed recipe,
> reconstructed from the published image's build history (NGC PyTorch
> 25.04 + `pip install evo2 biopython`). Not yet benchmarked on lab data.

- Base: `nvcr.io/nvidia/pytorch:25.04-py3` — torch 2.7.0a0, CUDA 12.9,
  Transformer Engine 2.2, flash-attn, Python 3.12
- Stack: `evo2` + `biopython` pip packages
- **GPU-only**, native sm_90 (base `TORCH_CUDA_ARCH_LIST` includes 9.0)
- Apache-2.0 code; weight license is per HF model card
- Large image (~12 GB) — the build workflow frees runner disk first

Pull: `ghcr.io/bradleylab/evo2:v1`

Weights are NOT baked — fetched from the `arcinstitute` HF org into
`$HF_HOME=/root/.cache/huggingface` on the first `Evo2('evo2_7b')` call;
bind-mount a scratch dir so the multi-GB checkpoint is fetched once. The
7B / 1B / `*_base` models run bf16 on the single H100 that C2
`general-gpu` allocates; `evo2_20b` / `evo2_40b` need FP8 via Transformer
Engine, and 40B needs multiple H100s — out of scope for a single-GPU job.

No CLI — upstream ships a Python API only, so inference goes through a
script under scratch. See `evo2/README.md` for the scoring / generation
snippets and the Transformer Engine version caveat (this image carries
NGC 25.04's TE 2.2; upstream suggests TE 2.3.0 for the FP8 path).

### esm

[ESM](https://github.com/Biohub/esm) (Chan Zuckerberg Biohub, formerly
EvolutionaryScale) — protein language model producing sequence embeddings
and per-residue logits from amino-acid sequence alone, no structure
required. ESMC is the current generation of Evolutionary Scale Modeling;
the open ESM3 checkpoint is reachable from the same image.

> **EXPERIMENTAL.** Not yet benchmarked on lab data. The default target
> is batch embedding with ESMC-600M.

- Base: `nvcr.io/nvidia/pytorch:25.04-py3` — torch 2.7.0a0, CUDA 12.9,
  **Python 3.12**. The Python version is a hard pin, not a preference:
  esm declares `requires-python = ">=3.12,<3.13"` and pip refuses to
  install on 3.11 or 3.13.
- Stack: `esm` from GitHub at a pinned commit SHA (upstream ships no
  PyPI release) plus Biohub's `transformers` fork, separately SHA-pinned.
  Both SHAs are recorded in the image labels and asserted by the build
  smoke test.
- **GPU-primary**, native sm_90
- MIT across code and weights

Pull: `ghcr.io/bradleylab/esm:v1`

Weights are NOT baked. The `biohub` checkpoints are MIT and **ungated**,
so an anonymous pull works and a Compute2 job needs no HF token.
`biohub/ESMC-600M` (575M params, 2.30 GB F32) is the workhorse for batch
embedding; `biohub/ESMC-300M`, `biohub/ESMC-6B` (6.35B, 25.41 GB F32 —
fits one H100 80 GB, bf16 halves it), and `biohub/esm3-sm-open-v1` (1.4B)
are also selectable. The large ESM3 checkpoints (7B, 98B) are API-only
and have never been downloadable, so they are out of scope here.

Local inference goes through the Biohub `transformers` fork, not an
`esm.*` model class — `esm.sdk` targets the hosted Biohub Platform API
and needs an API key. Do not add a pinned upstream `transformers` to this
image or merge it with an environment that has stock transformers: the
fork is what provides the ESMC model class. See `esm/README.md`.

### clean

[CLEAN](https://github.com/tttianhao/CLEAN) (Yu et al. 2023, *Science*
379:1358, [doi:10.1126/science.adf2465](https://doi.org/10.1126/science.adf2465))
— Contrastive Learning–Enabled Enzyme Annotation. Assigns Enzyme
Commission (EC) numbers to an amino-acid sequence: embed with ESM-1b,
project through a contrastively-trained network, assign by distance to
pre-computed EC cluster centres. One sequence can receive several EC
numbers.

> **EXPERIMENTAL.** Not yet benchmarked on lab data.

> **⚠ Licence: research use only, and not MIT.** GitHub's repository
> metadata advertises MIT, but the tree contains **no LICENSE file** —
> the only licence artifact upstream ships is a
> `NON-EXCLUSIVE RESEARCH USE LICENSE FOR CLEAN SOFTWARE.pdf`.
> Research-use-only and MIT cannot both be true, so the image is built
> and labelled under the research-use reading
> (`LicenseRef-CLEAN-Non-Exclusive-Research-Use`), deliberately **not**
> as MIT. Terms for the pretrained weights are stated nowhere at all;
> treat them as research-use-only by the same reading. University
> research is accepted as within terms; a commercial pipeline, a
> third-party service, or redistribution under a permissive licence is
> not established.

- Base: `python:3.10-slim` (upstream's manuscript environment was 3.10.4)
- Stack: torch 2.5.1+cpu, `fair-esm 1.0.2`, upstream's own
  `requirements.txt` pins (numpy 1.22.3, pandas 1.4.2, scipy 1.7.3,
  scikit-learn 1.2.0)
- **CPU-only** — run on Compute2 `general-cpu`. The documented resource
  floor is **>12 GB of system RAM**, not VRAM; the "7.3 GB" in upstream's
  README is the ESM-1b download size, not a memory requirement. CLEAN
  calls `torch.cuda.is_available()` and will happily occupy an H100 it
  does not need, so this image ships a CPU-only torch build and asserts
  that at build time.
- Both upstream repos SHA-pinned (`tttianhao/CLEAN` at `f2bf2a4f`,
  `facebookresearch/esm` at the `v1.0.2` tag); the smoke test fails the
  build if a pin has drifted.

Pull: `ghcr.io/bradleylab/clean:v1`

Weights are NOT baked, and staging them is the awkward part: two
independent downloads from two places, neither a package registry — the
CLEAN bundle (~141 MB, **Google Drive only**, and upstream cites two
conflicting Drive file IDs with no checksum for either) and ESM-1b
(~7.3 GB from `dl.fbaipublicfiles.com`). Mirror both to Storage3 with
`sha256` manifests before any analysis depends on them. `fair-esm 1.0.2`
is chosen over upstream's own `2.0.0` pin for manuscript fidelity —
predictions depend on the ESM-1b version, and the tradeoff is documented.
torch is pinned below 2.6 because 2.6's `weights_only=True` default
refuses the ESM-1b checkpoint. See `clean/README.md`.

### saprot

[SaProt](https://github.com/westlake-repl/SaProt) (Westlake University;
preprint [doi:10.1101/2023.10.01.560349](https://doi.org/10.1101/2023.10.01.560349),
journal version in *Nature Biotechnology*, 2025-10-24) — structure-aware
protein language model. Where ESM tokenizes one amino acid at a time,
SaProt tokenizes one *residue-state* at a time: each token pairs the amino
acid with that residue's foldseek 3Di structural state (`Aq`, `Md`, `Gp`).
The structural half can be masked (`A#`), which is how the 1.3B checkpoint
takes sequence-only input.

> **EXPERIMENTAL.** Not yet benchmarked on lab data. Default target is
> embedding structures with `SaProt_1.3B_AFDB_OMG_NCBI`.

- Base: `nvcr.io/nvidia/pytorch:25.04-py3` — torch 2.7.0a0, CUDA 12.9,
  Python 3.12 (same base as `esm` / `evo2` / `ntv3`)
- Stack: stock `transformers` 5.15.0 (the checkpoints are ordinary
  ESM-architecture HF repos, so `EsmTokenizer` / `EsmForMaskedLM` load
  them directly) + `biopython` + `accelerate`
- **GPU-primary**, native sm_90
- **Licence: `MIT AND GPL-3.0`.** SaProt's code and weights are MIT, but
  this image bundles the **foldseek** binary (release `10-941cd33`), which
  is **GPL-3.0**. The binary is an unmodified upstream release invoked as
  a subprocess — the arrangement foldseek itself documents — but the
  licence mix matters before this image is redistributed outside the lab.

Pull: `ghcr.io/bradleylab/saprot:v1`

**Bundling foldseek is the documented exception to one-model-per-container**,
and about as clear-cut as that exception gets: SaProt's input
representation does not exist until foldseek has produced it.
`get_struc_seq` shells out to `foldseek structureto3didescriptor` inside a
single function call and reads back the temp file it wrote, so there is no
intermediate artifact a separate container could hand over. foldseek is
also not pip-installable.

Weights are NOT baked — fetched from the `westlake-repl` HF org (MIT,
ungated) on first `from_pretrained`. Use `SaProt_1.3B_AFDB_OMG_NCBI`: the
35M and 650M checkpoints only produce usable frozen embeddings from
structure tokens, while the 1.3B ones also accept AA-only input, which
keeps embeddings comparable across a pipeline where some proteins have no
structure. **The bottleneck is foldseek on CPU, not the GPU** — size the
job by CPU count. Upstream's `torch==1.13.1` environment is deliberately
not used (it predates Hopper), so the repo's YAML-driven fine-tuning and
evaluation scripts are not available here; this image covers embedding and
scoring. Pass `plddt_mask=True` explicitly for any predicted structure —
the `"auto"` default only detects AFDB downloads. See `saprot/README.md`.

### boltz

[Boltz-2](https://github.com/jwohlwend/boltz) (Barzilay / Jaakkola group,
MIT), tag **v2.2.1** — biomolecular co-folding. Predicts the structure of
complexes containing proteins, RNA, DNA, and small molecules from sequence
and chemical-component identity, and — from a block in the same input
file — predicts binding affinity for a nominated ligand chain. Successor
to Boltz-1.

> **EXPERIMENTAL.** Not yet benchmarked on lab data. Read the silent
> out-of-memory note below before running anything unattended.

- Base: `pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime` — torch 2.8.0,
  CUDA 12.9, Python 3.11
- Stack: `boltz[cuda]==2.2.1` from PyPI, with cuEquivariance pinned to
  0.6.1 (unpinned, 0.11.1 promotes `torch>=2.11` to a hard runtime
  dependency and silently replaces the base image's torch and CUDA stack)
- **GPU-primary**, sm_90; upstream CLI `boltz predict`
- **Licence: MIT for the code *and* the weights** — academic and
  commercial use both permitted, no acceptance step, no gated download.

Pull: `ghcr.io/bradleylab/boltz:v1`

**`boltz predict` catches CUDA OOM per batch, warns, and still exits 0**
([upstream issue #167](https://github.com/jwohlwend/boltz/issues/167)), so
a Slurm job that produced no structures at all reports success. **Every
job must gate on output files existing, not on the exit status** — the
one-line gate is in `boltz/README.md`'s sbatch example. Reported VRAM is
~11 GB for structure and ~7–8 GB for affinity while NVIDIA's NIM support
matrix asks for ≥48 GB; the gap is headroom for large multimers, and a
single H100 80 GB is comfortable for ordinary work.

Weights are NOT baked — ~6.2 GB across three assets (`boltz2_conf.ckpt`,
`boltz2_aff.ckpt`, `mols.tar`) fetched into `$BOLTZ_CACHE=/opt/boltz-cache`,
which must be overridden with an **absolute** path. MSAs are required and
are not fetched for you (`--use_msa_server` defaults to False); the lab
default is precomputed `.a3m` files from our own mmseqs2, so the image
makes no MSA network calls. See `boltz/README.md`.

### chai-1

[Chai-1](https://github.com/chaidiscovery/chai-lab) (Chai Discovery,
release **v0.6.1**; bioRxiv 2024,
[doi:10.1101/2024.10.10.615955](https://doi.org/10.1101/2024.10.10.615955))
— co-folding model for complexes of proteins, RNA, DNA, and small
molecules. Unlike Boltz-2 and AlphaFold3 it reaches most of its accuracy
**without MSAs**, using a traced ESM-2 3B embedder in their place; local
MSAs still help and are supported.

> **EXPERIMENTAL.** Not yet benchmarked on lab data.

- Base: `pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime` — torch 2.6.0,
  CUDA 12.6, Python 3.11
- Stack: `chai_lab==0.6.1` from PyPI, pinned to the release tag
- **GPU-primary**, sm_90; upstream recommends A100/H100 80 GB or an
  L40S 48 GB. CLI: `chai-lab fold`
- **Licence: Apache-2.0 for the code *and* the weights.** Chai Discovery
  relicensed in **November 2024**; the original September-2024
  research-only terms no longer apply, and academic and commercial use are
  both permitted. Any older lab note describing Chai-1 as restrictively
  licensed is obsolete.

Pull: `ghcr.io/bradleylab/chai-1:v1`

**Request ≥64 GB of host `--mem`.** A measured PoseBench benchmark (A100
80 GB, 5 diffusion samples, `low_memory=True`) peaks at 56.2 GB VRAM and
**58.5 GB host RAM** — the second figure is the one people miss, and a
GPU-sized job with a default host allocation is the usual way this model
dies, with the OOM message pointing at the host rather than the GPU.

Weights are NOT baked — ~7.0 GB across 8 assets into
`$CHAI_DOWNLOADS_DIR=/opt/chai-downloads`, which is read at **import**
time and so must be set before Python starts. **The Hugging Face mirror is
incomplete**: it carries only the six `.pt` model components (~1.18 GB);
the 5.68 GB traced ESM-2 embedder and the 125 MB conformer pickle come
from `chaiassets.com` only, so pre-staging from HF alone leaves an
air-gapped job to fail at the ESM-embedding step. `--no-use-esm-embeddings`
avoids that download but changes what the model is given as input — a
methodological change, not a deployment convenience. This image and
`boltz` sit on deliberately different bases (2.6.0-cu126 vs 2.8.0-cu129);
the two dependency ceilings are incompatible, so do not harmonise them.
Chai-2 has no public weights or inference code and is not what this image
runs. See `chai-1/README.md`.

### dnabert-s

[DNABERT-S](https://github.com/MAGICS-LAB/DNABERT_S) (MAGICS-LAB) —
species-aware DNA language model built for metagenomic binning. Embeds a
nucleotide sequence into a 768-dimensional vector in which contigs from
the same species sit close together, so embeddings can be clustered into
bins without alignment or reference genomes. Architecture is DNABERT-2's
(MosaicBERT-style encoder, ALiBi position biases, 4096-token BPE
vocabulary, ~117M parameters) contrastively fine-tuned for species
separation.

> **EXPERIMENTAL.** Not yet benchmarked on lab data. Intended first
> target is batch embedding of assembled contigs.

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` + Python 3.11 +
  PyTorch 2.5.1 cu121 (the stack shared with `prithvi-eo`, `satlas`,
  `clay`, `terramind`)
- Stack: `transformers==4.27.*` — upstream's own pin — with
  `huggingface_hub<0.26`, `einops`, `numpy<2` held around it
- **GPU-primary**, native sm_90. The model is small enough that a GPU buys
  throughput, not capability
- **Triton is deliberately uninstalled**, on upstream's advice for
  non-A100 GPUs; `torch.compile` / TorchInductor are therefore unavailable
  in this image

Pull: `ghcr.io/bradleylab/dnabert-s:v1`

Weights are NOT baked — the single `zhihan1996/DNABERT-S` checkpoint
(468 MB fp32) downloads on first `from_pretrained`, which requires
`trust_remote_code=True` because the architecture ships as Python inside
the HF repo. **Weights are Apache-2.0 and ungated, but the upstream GitHub
repo has no LICENSE file** (formally all rights reserved); this image
vendors nothing from that repo, so it stays Apache-2.0 clean — the gap
only matters if someone later copies training or evaluation scripts out of
it. **Mean pooling over the attention mask is the documented embedding
method** for this model, not `[CLS]`.

**This model cannot share a container with `ntv3`** — DNABERT-S requires
`transformers==4.27`, NTv3 requires `>=4.55`, and no version satisfies
both. The two DNA images are separate by necessity, not only by the
one-model-per-container convention. See `dnabert-s/README.md`.

### ntv3

[Nucleotide Transformer v3](https://github.com/instadeepai/nucleotide-transformer)
(InstaDeep, released December 2025) — genomic language model that reads up
to **1 Mb of sequence at nucleotide resolution**. Beyond embeddings, the
post-trained checkpoints predict roughly 16,000 functional genomic tracks
across 24 species — the signal set you would otherwise get from BigWig
files — plus base-resolution annotation suitable for writing out as BED.

> **EXPERIMENTAL.** Not yet benchmarked on lab data. Intended first
> target is embeddings and track prediction at 131 kb windows.

> **⚠ Licence: non-commercial, and the weights are HF-gated.** The weights
> carry the **InstaDeep NTv3 non-commercial licence** (no commercial use;
> no training a competing model on this model's outputs) and upstream code
> is **CC BY-NC-SA 4.0**. An HF account must accept the terms on the model
> page before any download works. University research at WashU is in
> scope; commercial work is not. The second non-commercial image in the
> catalog, after `dofa-clip`.

- Base: `nvcr.io/nvidia/pytorch:25.04-py3` — torch 2.7.0a0, CUDA 12.9,
  flash-attn, Python 3.12 (same base as `esm`, `evo2`, `saprot`)
- Stack: `transformers>=4.55,<5` — upstream's floor for the custom
  `ntv3_posttrained` architecture, reached via `trust_remote_code=True`
- **GPU-primary**, native sm_90; bf16 recommended on H100
- **The HF token is never baked.** It is supplied from the environment
  only while weights are staged on a login node, and jobs then run offline
  (`HF_HUB_OFFLINE=1`). The build-time smoke test fails the build if
  `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is set during the build, so a
  token cannot be published to GHCR by accident.

Pull: `ghcr.io/bradleylab/ntv3:v1`

Weights are NOT baked. `InstaDeepAI/NTv3_650M_post` (650M params, 2.72 GB
fp32, embedding dim 1536) is the recommended checkpoint; 100M, `*_131kb`,
pre-trained-only, and generative variants also exist. **There is no
official VRAM figure and none is invented here** — the weights are the
easy part, and activation memory is what scales with context length. Start
at 131 kb windows, measure with `torch.cuda.max_memory_allocated()`, and
scale up empirically.

Two input rules produce wrong output rather than an error: **input length
must be a multiple of 128 bp**, and **padding is the character `N`, not
the tokenizer's `[PAD]`** — appended to the string before tokenization.
For the post-trained track heads, outputs are cropped to the **middle
62.5%** of the window, so tiling a chromosome steps by 81,920 bp of a
131,072 bp window, not by the full window.

**This model cannot share a container with `dnabert-s`** (see above).
Note also that the GitHub repo is a JAX codebase; this image uses the
PyTorch checkpoints on the Hub instead, so upstream's JAX install
instructions do not apply. See `ntv3/README.md`.
