# ml-containers

Custom ML Docker images for bradleylab research compute (WashU RIS Compute2, EC2).

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
| `treelearn` | (backfill in progress) | recipe pending |
| `pointstowood` | (backfill in progress) | recipe pending |
| `3dfin` | (backfill in progress) | recipe pending |
| `multispec-species` | (backfill in progress) | recipe pending |
| `tree-analysis` | — | kitchen-sink (deprecated; replaced by per-model containers) |

Planned: `forainet`, `deepforest` (extracted from the deprecated
`tree-analysis` kitchen sink).

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
