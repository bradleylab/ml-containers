# clay

Clay Foundation Model — Vision-Transformer Masked Autoencoder
pretrained on multi-sensor Earth observation imagery (Sentinel-2,
Sentinel-1 SAR, Landsat, NAIP, MODIS). Outputs per-patch embeddings
usable for similarity search, clustering, or lightweight downstream
classification with minimal labels.

Its ENTRYPOINT is one runnable task, `clay-embed`: Clay v1.5
embeddings of a 4-band PlanetScope surface-reflectance scene, one per
square chip (below). The encoder's weights are baked into the image.
The task uses a GPU when one is visible and the CPU otherwise; a whole
6778 × 2612-pixel PlanetScope scene took 52 s on 16 CPU cores.

## Image tag

`ghcr.io/bradleylab/clay:v3` (also `:latest`, `:torch2.5-cu121`).
`:v2` is the same stack without the embedding task. Both bundle the
upstream `configs/metadata.yaml` (pinned to `CLAY_GIT_SHA`) at
`/work/configs/metadata.yaml` so `ClayMAEModule.load_from_checkpoint`
works without callers having to pass `metadata_path=...`. Users
bind-mounting their own data should mount it at `/data`, not `/work`,
or the bundled metadata gets shadowed.

## Stack

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11
- PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `claymodel==1.5.0` (installed from pinned upstream commit; see
  Dockerfile `CLAY_GIT_SHA` ARG)
- huggingface_hub, geopandas, scikit-image, scikit-learn, timm,
  vit-pytorch, lightning, einops, jsonargparse (transitive)
- `rasterio ==1.4.3` (its wheel bundles GDAL 3.9.3) and
  `safetensors ==0.7.0`, for the embedding task
- GDAL system libs

`WANDB_MODE=offline` is set in ENV so accidental wandb auth attempts
don't block runs.

## Embedding task

`clay-embed` implements the geospatial executor's entrypoint contract
v1 (`fossettlab/geospatial-executor`, `docs/contracts/entrypoint-v1.md`):
three options, the scene staged under `<input-dir>/primary/`, and every
output plus a `run.json` manifest written under `<output-dir>`.
`clay_embed.cwl` describes the same task as a CWL v1.2 tool.

- **Input:** one 4-band PlanetScope surface-reflectance GeoTIFF (blue,
  green, red, NIR, reflectance scaled by 10,000, 16-bit integers) in a
  projected CRS in meters with square cells.
- **Model:** the v1.5 encoder as upstream's `Embedder` runs it
  (`claymodel/finetune/embedder/factory.py`): no masking, the class
  token as each chip's embedding (1,024 numbers).
- **Datacube:** built as upstream's wall-to-wall tutorial builds it.
  Bands are normalized with the mean and standard deviation of the
  matching bands of the `planetscope-sr` entry in `metadata.yaml`, which
  also gives their wavelengths (0.490, 0.565, 0.665 and 0.865 µm).
  Latitude and longitude are the sines and cosines of each chip's
  center; the cell size is the ground sample distance. The acquisition
  time is not read from the GeoTIFF and is encoded as zeros, the value
  upstream's fine-tuning docs give for an unknown date.
- **Parameter:** `chip_size`, 64, 128 or 256 pixels (default 256, the
  chip size Clay v1.5 was trained on).
- **Chips:** a grid anchored at the scene's upper-left corner. A chip
  holding any masked (nodata) pixel is not embedded, and a partial chip
  at the right or bottom edge is left out.
- **Outputs:** `embeddings.tif`, a Cloud Optimized GeoTIFF with one cell
  per chip and one float32 band per embedding dimension, NaN where no
  embedding was made; and `embeddings_report.json`, with the chip counts
  (total, embedded, masked), the band constants used and the weights'
  provenance.

```bash
docker run --rm --gpus all \
  -v /path/to/job:/job \
  ghcr.io/bradleylab/clay:v3 \
  --input-dir /job/input \
  --output-dir /job/output \
  --params-json /job/params.json
```

with the scene at `/path/to/job/input/primary/<name>.tif` and
`params.json` holding, for example, `{"chip_size": "256"}`. Drop
`--gpus all` to run on the CPU.

On a 2026-05-03 PlanetScope scene of St. Louis, 64-pixel chips separated
the river, the urban west bank and the fields east of it in the first
three principal components of the embeddings. Embeddings are features
for comparing and grouping places, not labels; nothing here measures how
well they separate any particular class.

## Weights

The embedding task's weights are baked in at
`/opt/weights/clay_v1.5_encoder.safetensors`: at build time,
`scripts/convert_checkpoint.py` downloads `v1.5/clay-v1.5.ckpt` from
[`made-with-clay/Clay`](https://huggingface.co/made-with-clay/Clay) at a
pinned revision, checks its SHA-256, and keeps the `model.encoder.*`
tensors (311,442,944 parameters). The checkpoint is read with PyTorch's
`weights_only` loader, whose allowlist covers everything its pickle
names.

The full checkpoint is 5.16 GB (the Hub's file listing). For other uses
it downloads on first call to
`hf_hub_download("made-with-clay/Clay", "v1.5/clay-v1.5.ckpt")` into
`$HF_HOME=/opt/hf-cache`; bind-mount a persistent host dir there so it
downloads once per host:

```bash
docker run --rm -it --gpus all \
  -v "$PWD/hf-cache:/opt/hf-cache" \
  -v "$PWD/data:/data" \
  --entrypoint bash \
  ghcr.io/bradleylab/clay:v3
```

## Other inference

```python
import torch
from huggingface_hub import hf_hub_download
from claymodel.module import ClayMAEModule

ckpt = hf_hub_download("made-with-clay/Clay", "v1.5/clay-v1.5.ckpt")

# As upstream's wall-to-wall tutorial loads it; the module's
# own default mask_ratio would mask 75% of patches.
model = ClayMAEModule.load_from_checkpoint(
    ckpt,
    model_size="large",
    metadata_path="/work/configs/metadata.yaml",
    dolls=[16, 32, 64, 128, 256, 768, 1024],
    doll_weights=[1, 1, 1, 1, 1, 1, 1],
    mask_ratio=0.0,
    shuffle=False,
)
model.eval()

# Construct a batch matching Clay's input contract — see the upstream
# tutorials at https://clay-foundation.github.io/model for the
# data-prep recipe (datacube layout, time, latlon, gsd, waves).
# Then call `model.model.encoder(...)` for embeddings.
```

The Clay project's
[wall-to-wall tutorial](https://clay-foundation.github.io/model/clay-v1/tutorials/wall-to-wall.html)
is the canonical "load + embed + downstream" walkthrough; once
checkpoints + tutorial datacubes are bind-mounted into `/data`, the
notebook runs unchanged inside this container.

## Inputs beyond PlanetScope

Clay expects multi-sensor input "datacubes" with paired metadata
(time, lat/lon, ground sample distance, wavelength bands). Building
these from raw STAC items is the bulk of the work; the upstream
[`stacchip`](https://github.com/Clay-foundation/stacchip) package is
the recommended tooling but is not bundled here.

## Run on Compute2

The embedding task on a CPU node, run the way the geospatial executor
runs contract images under pyxis (the entrypoint's path as the command):

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=8 \
       --mem=16G \
       --time=01:00:00 \
       --wrap='srun \
         --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+clay+v3.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
         --export=ALL,PYTHONNOUSERSITE=1,NVIDIA_VISIBLE_DEVICES=void \
         /usr/local/bin/clay-embed \
           --input-dir /scratch2/fs1/alexander.s.bradley/clay-job/input \
           --output-dir /scratch2/fs1/alexander.s.bradley/clay-job/output \
           --params-json /scratch2/fs1/alexander.s.bradley/clay-job/params.json'
```

`PYTHONNOUSERSITE=1` is required on Compute2: enroot mounts `$HOME`,
and a user site-packages directory there would otherwise shadow the
image's packages. `NVIDIA_VISIBLE_DEVICES=void` is for CPU nodes only:
without it, enroot's GPU hook stops a CUDA image from starting where
there is no GPU driver. The whole St. Louis scene peaked at 4.1 GB of
memory.

## Limitations

- Clay's input contract is non-trivial — datacubes with time, lat/lon,
  GSD, and wavelength metadata are required, not bare images. The
  embedding task covers one sensor, PlanetScope 4-band surface
  reflectance.
- The image does not bundle `stacchip` / Clay's tutorial datacube
  builders.
- The image pins claymodel to a specific upstream commit (see
  Dockerfile ARG); when the upstream releases a new model variant, bump
  the SHA and the image version together.
