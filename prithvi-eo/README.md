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
Lightning task scaffolding. Prithvi weights are NOT baked — they
download from HF Hub on first use.

GPU-primary (H100 sm_90). The 100M v1 fine-tuned variants run on a
laptop GPU with 8+ GB VRAM; the 300M and 600M v2 base models want
H100 for training, and inference benefits from GPU even if it's not
strictly required.

## Image tag

`ghcr.io/bradleylab/prithvi-eo:v1` (also `:latest`,
`:torch2.5-cu121`)

## Stack

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11
- PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `terratorch >= 1.2.5`
- Lightning, segmentation-models-pytorch, torchgeo, timm, diffusers,
  geopandas, rasterio, albumentations (transitive — see
  `terratorch`'s pyproject for the full list)

`WANDB_MODE=offline` and `NO_ALBUMENTATIONS_UPDATE=true` in ENV so
neither service blocks runs with a network probe at import time.

## Weights

Not baked. Pull from HF Hub on first call:

| Variant | HF Hub repo | Approx. size |
|---|---|---|
| 1.0 100M | [`ibm-nasa-geospatial/Prithvi-EO-1.0-100M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M) | ~400 MB |
| 2.0 300M | [`ibm-nasa-geospatial/Prithvi-EO-2.0-300M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) | ~1.2 GB |
| 2.0 300M-TL | [`ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL) | ~1.2 GB |
| 2.0 600M | [`ibm-nasa-geospatial/Prithvi-EO-2.0-600M`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M) | ~2.5 GB |
| 2.0 600M-TL | [`ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL`](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-600M-TL) | ~2.5 GB |

Bind-mount a persistent host directory at `/opt/hf-cache` so each
variant only downloads once per host.

```bash
docker run --rm -it --gpus all \
  -v "$PWD/hf-cache:/opt/hf-cache" \
  -v "$PWD/data:/data" \
  ghcr.io/bradleylab/prithvi-eo:v1
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

Inference / fine-tuning on `general-gpu`:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-gpu \
       --gpus=1 \
       --mem=64G \
       --time=04:00:00 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+prithvi-eo+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/opt/hf-cache,/scratch2/fs1/alexander.s.bradley/hls-stacks:/data,/scratch2/fs1/alexander.s.bradley/prithvi-out:/outputs \
         bash -lc "export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/prithvi_embed.py"'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — see
`~/.claude/rules/research-infrastructure.md`. Memory: 64 GB
accommodates 600M-TL plus a moderate batch; shrink for 100M / 300M.

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
