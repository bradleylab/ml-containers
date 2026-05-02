# clay

Clay Foundation Model — Vision-Transformer Masked Autoencoder
pretrained on multi-sensor Earth observation imagery (Sentinel-2,
Sentinel-1 SAR, Landsat, NAIP, MODIS). Outputs per-patch embeddings
usable for similarity search, clustering, or lightweight downstream
classification with minimal labels.

GPU-primary (H100 sm_90). Single-tile embedding works on a laptop
GPU; batch embedding across an archive of tiles is the killer
use case where H100 throughput matters.

## Image tag

`ghcr.io/bradleylab/clay:v1` (also `:latest`, `:torch2.5-cu121`)

## Stack

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11
- PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `claymodel==1.5.0` (installed from pinned upstream commit; see
  Dockerfile `CLAY_GIT_SHA` ARG)
- huggingface_hub, geopandas, scikit-image, scikit-learn, timm,
  vit-pytorch, lightning, einops, jsonargparse (transitive)
- GDAL system libs (for geopandas / rasterio downstream use)

`WANDB_MODE=offline` is set in ENV so accidental wandb auth attempts
don't block runs.

## Weights

Not baked. The `made-with-clay/Clay` HF Hub repo carries:

| Variant | File | Approx. size |
|---|---|---|
| Clay v1.5 | `v1.5/clay-v1.5.ckpt` | ~3 GB |

The first call to
`hf_hub_download("made-with-clay/Clay", "v1.5/clay-v1.5.ckpt")`
downloads into `$HF_HOME=/opt/hf-cache`. Bind-mount a persistent host
dir so the checkpoint only downloads once per host:

```bash
docker run --rm -it --gpus all \
  -v "$PWD/hf-cache:/opt/hf-cache" \
  -v "$PWD/data:/data" \
  ghcr.io/bradleylab/clay:v1
```

## Inference

```python
import torch
from huggingface_hub import hf_hub_download
from claymodel.module import ClayMAEModule

ckpt = hf_hub_download("made-with-clay/Clay", "v1.5/clay-v1.5.ckpt")

# Load into eval mode for embedding extraction.
model = ClayMAEModule.load_from_checkpoint(ckpt, map_location="cuda")
model.eval()

# Construct a batch matching Clay's input contract — see the upstream
# tutorials at https://clay-foundation.github.io/model for the
# data-prep recipe (datacube layout, time, latlon, gsd, waves).
# Then call `model.encoder(...)` for embeddings.
```

The Clay project's
[wall-to-wall tutorial](https://clay-foundation.github.io/model/clay-v1/tutorials/wall-to-wall.html)
is the canonical "load + embed + downstream" walkthrough; once
checkpoints + tutorial datacubes are bind-mounted into `/data`, the
notebook runs unchanged inside this container.

## Inputs

Clay expects multi-sensor input "datacubes" with paired metadata
(time, lat/lon, ground sample distance, wavelength bands). Building
these from raw STAC items is the bulk of the work; the upstream
[`stacchip`](https://github.com/Clay-foundation/stacchip) package is
the recommended tooling but is not bundled here (out of scope for
v1; add later if needed).

For zero-prep "just embed an arbitrary image" usage, downstream
adapters from the Clay community wrap the contract — start there.

## Run on Compute2

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-gpu \
       --gpus=1 \
       --mem=64G \
       --time=04:00:00 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+clay+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/opt/hf-cache,/scratch2/fs1/alexander.s.bradley/datacubes:/data,/scratch2/fs1/alexander.s.bradley/embeddings:/outputs \
         bash -lc "export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/clay_embed.py"'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — see
`~/.claude/rules/research-infrastructure.md`. Memory: 64 GB is
generous for a v1.5 checkpoint plus a moderate datacube batch;
shrink for short jobs.

## Limitations

- Clay's input contract is non-trivial — datacubes with time, lat/lon,
  GSD, and wavelength metadata are required, not bare images. Plan
  to spend more time on data prep than on embedding.
- The image does not bundle `stacchip` / Clay's tutorial datacube
  builders — out of scope for v1. Add a downstream container or a
  scripts/ directory if the data-prep gets repeated.
- v1 pins claymodel to a specific upstream commit (see Dockerfile
  ARG); when the upstream releases a new model variant, bump the
  SHA and the image version together.
