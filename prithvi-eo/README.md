# prithvi-eo

[Prithvi-EO](https://huggingface.co/ibm-nasa-geospatial) is the
IBM/NASA family of ViT-based geospatial foundation models pre-trained
on Harmonized Landsat–Sentinel-2 (HLS) imagery. Three variants:

- **Prithvi-EO-1.0-100M** — original v1 base model.
- **Prithvi-EO-2.0-300M / 600M** — v2 base; `-TL` variants add
  temporal + locational embeddings.
- **Pre-fine-tuned heads** for burn-scar mapping, flood detection,
  and multi-temporal crop classification (separate HF Hub repos).

This container ships [TerraTorch](https://github.com/IBM/terratorch),
the IBM-supported fine-tuning toolkit that wraps Prithvi (and other
geospatial foundation models) behind the `BACKBONE_REGISTRY` +
Lightning task scaffolding. Backbone weights are NOT baked — they
download from HF Hub on first use.

Its ENTRYPOINT is one runnable task, `prithvi-burn-scars`: a burn-scar
map of one HLS scene from the `Prithvi-EO-2.0-300M-BurnScars`
fine-tune (below), whose checkpoint and config are baked into the
image.

GPU-primary (H100 sm_90). The 100M v1 fine-tuned variants run on a
laptop GPU with 8+ GB VRAM; the 300M and 600M v2 base models want
H100 for training, and inference benefits from GPU even if it's not
strictly required.

## Image tag

`ghcr.io/bradleylab/prithvi-eo:v3` (also `:latest`,
`:torch2.5-cu121`). `:v2` is the same task without the PNG preview, and
`:v1` the same stack without the burn-scar task.

## Stack

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11
- PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `terratorch ==1.2.7` (the Dockerfile explains the pin)
- Lightning, segmentation-models-pytorch, torchgeo, timm, diffusers,
  geopandas, rasterio, albumentations (transitive — see
  `terratorch`'s pyproject for the full list)

`WANDB_MODE=offline` and `NO_ALBUMENTATIONS_UPDATE=true` in ENV so
neither service blocks runs with a network probe at import time.

## Burn-scar task

`prithvi-burn-scars` implements the geospatial executor's entrypoint
contract v1 (`fossettlab/geospatial-executor`,
`docs/contracts/entrypoint-v1.md`): three options, the scene staged under
`<input-dir>/primary/`, and every output plus a `run.json` manifest
written under `<output-dir>`. `prithvi_burn_scars.cwl` describes the same
task as a CWL v1.2 tool.

- **Input:** one 6-band HLS scene, int16 reflectance × 10000 with fill
  -9999, bands blue, green, red, narrow NIR, SWIR 1, SWIR 2, in a
  projected CRS in meters. The `hls` product of the satellite fetch task
  (`bradleylab/geo-pipelines`, `satellite-fetch`) writes one.
- **Model:** `ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars` at revision
  `a3f2c410`, loaded as its own `inference.py` loads it: TerraTorch's
  `LightningInferenceModel` from `burn_scars_config.yaml` and the
  checkpoint. The baked config differs from the published one in one line,
  `backbone_pretrained: false`, so building the model fetches nothing; the
  checkpoint holds every weight, and TerraTorch loads it with every key
  matched.
- **Inference:** as that `inference.py` runs it: reflectance divided by
  10,000, the scene reflect-padded to whole 512-pixel windows, each window
  normalized by the datamodule's transforms, the argmax of the two classes.
  Pixels with any band at fill go into the model as 0, as the training
  config's `no_data_replace` does, and come out as nodata.
- **Outputs:** `burn_scars.tif`, a Cloud Optimized GeoTIFF of 0 (not
  burned), 1 (burn scar) and 255 (nodata) with a color table;
  `burn_scars_preview.png`, an RGBA picture of that map for a chat
  interface to show inline, in the same colors with nodata transparent,
  shrunk by nearest neighbor to at most 1024 pixels on its longer side
  (never enlarged) and not georeferenced; and `burn_scars_report.json`,
  with the burned area in hectares and share and the preview's size in
  pixels and downsampling factor. `run.json` lists the preview with role
  `preview` and format `PNG`.
- **Parameters:** none.

```bash
docker run --rm \
  -v /path/to/job:/job \
  ghcr.io/bradleylab/prithvi-eo:v3 \
  --input-dir /job/input \
  --output-dir /job/output \
  --params-json /job/params.json
```

with the scene at `/path/to/job/input/primary/<name>.tif` and `{}` in
`params.json`. It uses a GPU when one is visible and the CPU otherwise:
on 16 CPU cores a 512 × 512 scene took 10 s and peaked at 3.5 GB.

On the model's three published example scenes (HLS v1.4 subsets over
California, converted to this layout), the burned areas it mapped
coincide with the burns visible in a SWIR 2 / NIR / red composite. The
model does not mask clouds: it was trained with clouds labeled as no
data, so cloud edges may be mapped as burn scars. The satellite fetch
task's quality layer shows where the scene is cloudy.

## Weights

Backbones are not baked; pull them from HF Hub on first call:

| Variant | HF Hub repo | Approx. size |
|---|---|---|
| 1.0 100M | [`ibm-nasa-geospatial/Prithvi-EO-1.0-100M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M) | ~400 MB |
| 2.0 300M | [`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) | ~1.2 GB |
| 2.0 300M-TL | [`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL) | ~1.2 GB |
| 2.0 600M | [`ibm-nasa-geospatial/Prithvi-EO-2.0-600M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M) | ~2.5 GB |
| 2.0 600M-TL | [`ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL) | ~2.5 GB |
| 2.0 300M-TL Sen1Floods11 (flood fine-tune) | [`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11) | ~1.2 GB |
| 2.0 300M BurnScars (burn-scar fine-tune, baked since v2) | [`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-BurnScars) | 1.30 GB |

Bind-mount a persistent host directory at `/opt/hf-cache` so each
variant only downloads once per host.

```bash
docker run --rm -it --gpus all \
  -v "$PWD/hf-cache:/opt/hf-cache" \
  -v "$PWD/data:/data" \
  --entrypoint bash \
  ghcr.io/bradleylab/prithvi-eo:v3
```

## Inference

TerraTorch's preferred pattern is to instantiate a backbone via
`BACKBONE_REGISTRY` and either run as embedding extractor or attach a
task head and fine-tune. Minimal embedding example:

```python
import torch
from terratorch import BACKBONE_REGISTRY

# Backbone names follow `prithvi_eo_v2_300m`, `prithvi_eo_v2_300m_tl`, etc.
backbone = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_300m_tl",
    pretrained=True,         # downloads from HF Hub on first call
    num_frames=4,            # multi-temporal input
)
backbone = backbone.cuda().eval()

# Input: (B, C=6, T=4, H=224, W=224) — HLS bands B2 B3 B4 B5 B6 B7,
# four time steps, 224x224 patch.
x = torch.randn(1, 6, 4, 224, 224, device="cuda")
with torch.no_grad():
    out = backbone(x)
print(type(out), [t.shape for t in out] if isinstance(out, (list, tuple)) else out.shape)
```

For end-to-end fine-tuning (downstream classification / segmentation
heads), use TerraTorch's LightningCLI: `terratorch fit --config
configs/...`. See the upstream
[TerraTorch tutorials](https://ibm.github.io/terratorch/) for the
canonical fine-tuning recipes (burn-scar, flood, multi-temporal crop).

### Pre-fine-tuned flood head (Sen1Floods11)

The flood fine-tune ships its own `config.yaml` + checkpoint and loads
through TerraTorch's `LightningInferenceModel`, not `BACKBONE_REGISTRY`.
Its `requirements.txt` pins `terratorch==0.99.8`; this image ships
terratorch >= 1.2.5, against which the load path below was verified
(pinned revision, CPU, 224x224 chip, 2026-06-10):

```python
from huggingface_hub import hf_hub_download
from terratorch.cli_tools import LightningInferenceModel

REPO = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL-Sen1Floods11"
REV = "918b9f140bb1783716664a2421ea3253d806017d"

config = hf_hub_download(REPO, "config.yaml", revision=REV)
ckpt = hf_hub_download(
    REPO, "Prithvi-EO-V2-300M-TL-Sen1Floods11.pt", revision=REV
)
model = LightningInferenceModel.from_config(config, ckpt)
```

Inputs, from the fine-tune's `config.yaml`: six S2 bands
BLUE/GREEN/RED/NIR_NARROW/SWIR_1/SWIR_2 (B02 B03 B04 B8A B11 B12),
224x224 chips, `constant_scale: 0.0001` (DN -> reflectance). Output is
two classes (water / not-water).

## Inputs

- 6-band HLS (Harmonized Landsat-Sentinel-2) imagery: B2 (Blue),
  B3 (Green), B4 (Red), B5 (NIR-narrow), B6 (SWIR-1), B7 (SWIR-2).
- Multi-temporal stacking: T = 1 (single image) up to T = 4
  (recommended for v2-TL).
- Spatial input: 224×224 patches typical; configurable via the
  `image_size` argument.

Input data prep (downloading HLS via NASA Earthdata, building
multi-temporal stacks, reprojection) is **not** in this image —
`stackstac`, `pystac-client`, or NASA's HLS DAAC scripts handle that
upstream.

## Run on Compute2

The burn-scar task on a CPU node, run the way the geospatial executor runs
contract images under pyxis (the entrypoint's path as the command):

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=8 \
       --mem=16G \
       --time=01:00:00 \
       --wrap='srun \
         --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+prithvi-eo+v3.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
         --export=ALL,PYTHONNOUSERSITE=1,NVIDIA_VISIBLE_DEVICES=void \
         /usr/local/bin/prithvi-burn-scars \
           --input-dir /scratch2/fs1/alexander.s.bradley/burn-job/input \
           --output-dir /scratch2/fs1/alexander.s.bradley/burn-job/output \
           --params-json /scratch2/fs1/alexander.s.bradley/burn-job/params.json'
```

`PYTHONNOUSERSITE=1` keeps a user site-packages directory in the mounted
`$HOME` from shadowing the image's packages. `NVIDIA_VISIBLE_DEVICES=void`
is for CPU nodes: without it, enroot's GPU hook stops this CUDA image from
starting where there is no GPU driver. Inside the container `HF_HOME`
(`/opt/hf-cache`) is read-only, so anything that downloads from the Hub
there needs `HF_HOME` exported to a writable path in the job's shell.

## Limitations

- **HLS is the only fully supported input.** TerraTorch and Prithvi
  expect the 6-band HLS layout. Other multi-spectral sources need
  band-mapping / resampling.
- **TerraTorch installs a heavy transitive stack** (Lightning,
  diffusers, segmentation-models-pytorch, torchgeo). Image is ~6-7 GB
  on disk. Cold pulls take a while.
- **No batch-data-prep tooling baked in.** Building HLS multi-temporal
  stacks from raw STAC items requires `stackstac` / `pystac-client` /
  `rio-stac`; not bundled here. Add a downstream image or a `scripts/`
  directory if you find yourself repeating the prep.
