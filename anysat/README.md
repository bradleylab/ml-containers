# anysat

AnySat (Astruc, Gonthier, Mallet and Landrieu; CVPR 2025 Highlight) is one
Earth-observation encoder for many resolutions, scales and sensors. It is a
ViT-style transformer of 125 M parameters, pretrained self-supervised with a
scale-adaptive joint-embedding predictive architecture (JEPA) on GeoPlex, five
multimodal datasets that together span 11 sensors. Any combination of those
sensors over one extent goes in at once, so very-high-resolution imagery and a
Sentinel-2 time series of the same ground are fused inside the model rather
than after it.

Paper: [arXiv:2412.14123](https://arxiv.org/abs/2412.14123) ·
Code: [gastruc/AnySat](https://github.com/gastruc/AnySat) @ `5f6f475` ·
Weights: [g-astruc/AnySat](https://huggingface.co/g-astruc/AnySat) @ `63f2521`

Pull: `ghcr.io/bradleylab/anysat:v1`

## Features, not labels

AnySat returns embeddings. Nothing in the release maps them to classes, so
every use needs a head trained on labeled examples: a linear probe, a small
MLP, or a fine-tune. Upstream's own evaluations use both linear probing and
fine-tuning. Where no labels exist, a public product can stand in as weak
labels; [`SMOKE.md`](SMOKE.md) does this with NLCD over Tyson Research Center.

## Loading

Upstream loads from GitHub through `torch.hub`. In this image the same
entrypoint loads offline from a pinned clone, with the checkpoint baked:

```python
import torch
model = torch.hub.load("/opt/anysat", "anysat", source="local",
                       pretrained=True, flash_attn=False).cuda().eval()
features = model(data, patch_size=30, output="patch")
```

`flash_attn=False` is required here. flash-attn is optional upstream, where it
changes only memory use and speed. PyPI carries it as source only, and
building it needs the CUDA toolkit, which this `python:3.12-slim` image does
not have; without it the encoder uses its own softmax attention.

## Input keys

`data` is a dict. Any subset of these keys may be present:

| Key | Type | Tensor | Channels | Resolution |
|---|---|---|---|---|
| `aerial` | single date | B×4×H×W | R, G, B, NIR | 0.2 m |
| `aerial-flair` | single date | B×5×H×W | R, G, B, NIR, elevation | 0.2 m |
| `spot` | single date | B×3×H×W | R, G, B | 1 m |
| `naip` | single date | B×4×H×W | R, G, B, NIR | 1.25 m |
| `s2` | time series | B×T×10×H×W | B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 | 10 m |
| `s1-asc` | time series | B×T×2×H×W | VV, VH | 10 m |
| `s1` | time series | B×T×3×H×W | VV, VH, ratio | 10 m |
| `alos` | time series | B×T×3×H×W | HH, HV, ratio | 30 m |
| `l7` | time series | B×T×6×H×W | B1, B2, B3, B4, B5, B7 | 30 m |
| `l8` | time series | B×T×11×H×W | B8, B1, B2, B3, B4, B5, B6, B7, B9, B10, B11 | 10 m |
| `modis` | time series | B×T×7×H×W | B1–B7 | 250 m |

The table is upstream's; the resolutions are the ones `hubconf.py` uses.
The `aerial-flair` elevation channel is FLAIR's normalized digital surface
model. `spot` is nominally 1 m, although the paper gives PASTIS-HD's SPOT 6
imagery as 1.5 m. The `l8` bands are at 10 m because S2NAIP-Urban's Landsat
8/9 data were rescaled to 10 m.

**Dates.** Each time series needs a `<key>_dates` tensor of shape B×T holding
the day of year of each acquisition. Upstream's README says 1 January is 0;
its dataset loaders, which produced the pretraining inputs, use Python's
`tm_yday`, where 1 January is 1 (PASTIS-HD's loader instead counts days from
2018-09-01). The helper script follows the majority, `tm_yday`. A
`<key>_mask` tensor may accompany a time series to mark invalid observations;
none of the five pretraining datasets supplies one.

**Normalization.** Every channel is standardized, `(x − mean) / std`. Upstream
computes the statistics per dataset from the dataset itself and ships none of
them (the demo notebook hard-codes TreeSatAI-TS values). For new data,
compute them over the whole region being processed, never per tile.

## One extent, square tiles

Every modality in one sample must cover the same ground: pixel count times
resolution has to be equal across keys, or `hubconf.py` raises `ValueError`
(MODIS is exempt from the check). Images must be square. A 960 m tile is
therefore NAIP 768×768, Sentinel-2 96×96, and SPOT 960×960.

`patch_size` is in meters and must be a multiple of 10 that divides the tile.
Internally the projectors work on 10 m cells: 50 pixels of `aerial`, 8 of
`naip`, 10 of `spot`, 1 of `s2`, so VHR tile edges must be multiples of those
pixel counts times `patch_size / 10`. Upstream advises at most 1024 patches per
tile; 960 m tiles with 30 m patches are exactly 32×32.

## Outputs

| `output=` | Shape | Content |
|---|---|---|
| `tile` | B×768 | one vector per tile (the class token) |
| `patch` | B×n×n×768 | one vector per patch, n = tile / patch_size |
| `dense` | B×S×S×1536 | per sub-patch of `output_modality`: the patch vector concatenated with the sub-patch vector |
| `all` | B×(1+n²)×768 | class token first, then the patches |

`patch` and `dense` are channels-last. Upstream's README comments show
`patch` as B×D×n×n; the code and the demo notebook return B×n×n×D. For
`dense`, a sub-patch is one pixel of a time series and one projector cell of a
VHR image (10×10 pixels of `aerial`, 2 m on the ground), and `output_modality`
names which input sets that grid.

**Per-location work belongs on `dense`.** The fused tokens that `patch`
returns come out of a final cross-attention layer, and those of one tile are
close to identical to each other, already on upstream's demo sample; most of
their variation is between tiles. The second half of each `dense` vector is
the sub-patch feature from before that fusion and keeps the local detail,
which is why upstream sends segmentation through `dense`. `tile` suits
tile-level questions.

## Pretraining data (GeoPlex)

| Dataset | Region | Modalities | Tiles |
|---|---|---|---|
| TreeSatAI-TS | Germany | aerial 0.2 m RGBN, Sentinel-2, Sentinel-1 | 50,381 |
| PASTIS-HD | metropolitan France (cropland) | SPOT 6 1.5 m, Sentinel-2, Sentinel-1 ascending | 2,433 |
| FLAIR | metropolitan France | aerial 0.2 m RGBN + nDSM, Sentinel-2 | 77,762 |
| PLANTED | planted forests worldwide | Sentinel-2, Landsat 7, MODIS, Sentinel-1, ALOS-2 | 1,346,662 |
| S2NAIP-Urban | US urban areas (NAIP) | NAIP 1.25 m, Sentinel-2, Sentinel-1, Landsat 8/9 | 515,270 |

Regions and counts are from the paper, except PASTIS's, which is from the
PASTIS benchmark repository. Three of the five datasets are European, all
the 0.2 m aerial imagery AnySat has seen is from Germany and France, and its
NAIP is urban. Temperate hardwood forest under NAIP or drone imagery, as at
Tyson, is out of distribution for the VHR keys; the Sentinel-2 path saw
forests worldwide through PLANTED. That is a reason to judge any use against
a simple baseline, as `SMOKE.md` does, and not a reason to expect failure.

## Drone orthomosaics

There is no drone key. Drone imagery goes in through one of the VHR keys,
resampled to that key's resolution and aligned to the same extent as the
other inputs:

- **Multispectral with NIR** (R, G, B, NIR): resample to 0.2 m and use
  `aerial`. When the orthomosaic is finer than 0.2 m, downsample by area
  averaging (`gdalwarp -r average -tr 0.2 0.2`).
- **With a canopy or surface height model**: add normalized surface height
  (height above ground, from lidar or photogrammetry) as a fifth band and use
  `aerial-flair`, whose fifth channel is FLAIR's nDSM.
- **RGB only**: no 0.2 m key takes three bands. Either resample to 1 m and
  use `spot`, giving up the fine detail, or leave the VHR key out and use
  Sentinel-2 alone. Filling a NIR channel with zeros or a copy of red feeds
  the `aerial` projector values it never saw.

Band order follows the table, whatever order the camera writes. At 0.2 m a
60 m tile is 300×300 pixels, the size upstream's demo uses; a 960 m tile would
be 4800×4800 per sample.

## Helper scripts

On `PATH` in the image:

- [`anysat_features.py`](scripts/anysat_features.py) — builds the input dict
  from a VHR GeoTIFF and a date-major Sentinel-2 stack with a dates file,
  checks bands, resolution and extent, normalizes over the whole extent, runs
  the model tile by tile and writes the reassembled `tile`, `patch` and, with
  `--dense <key>`, `dense` grids to `.npz`. Importable as a module (`read_scene`,
  `channel_stats`, `extract`).
- [`fetch_aoi.py`](scripts/fetch_aoi.py) — writes NAIP, a Sentinel-2 L2A time
  series and NLCD 2021 over a square AOI on one UTM grid, from Planetary
  Computer and the MRLC web coverage service. The default AOI is Tyson.
- [`nlcd_probe.py`](scripts/nlcd_probe.py) — the linear probe against NLCD:
  fused patch, dense and sub-patch features each against a band-mean
  baseline, with spatial cross-validation.

## Stack

- Base: `python:3.12-slim-bookworm`.
- `torch==2.12.1` from the CUDA 12.9 index. Upstream asks for torch ≥ 2.0;
  2.12.1 on CUDA 12.9 is the build `nasa-ibm-lunar-fm` runs on Compute2's H100
  nodes, whose driver (580.105.08) supports it.
- AnySat cloned at `5f6f475` to `/opt/anysat` and loaded with
  `torch.hub.load(..., source="local")`. Its inference path imports only torch
  and numpy. Upstream's training stack (Lightning, Hydra, wandb, the GeoPlex
  loaders) is not installed, so `src/train.py` does not run here.
- `models/AnySat.pth` (504 MB, SHA-256 `ce124652…3d4f`) baked at
  `/opt/torch-hub/hub/checkpoints/`, with `TORCH_HOME=/opt/torch-hub`, which
  is where the hub entrypoint's `load_state_dict_from_url` looks before
  downloading. The Hugging Face repository also holds `models/AnySat_full.pth`
  (1.1 GB); the hub entrypoint does not use it and it is not baked.
- [`constraints.txt`](constraints.txt) freezes the full dependency set,
  resolved 2026-09-28 for the helper scripts' libraries (rasterio, pyproj,
  scikit-learn, pystac-client, planetary-computer) against that torch.

The build test runs with the network cut: it imports torch and the helper
libraries separately under faulthandler, asserts the CUDA 12.9 build, loads
the model from `/opt/anysat` with the baked weights, and forwards a synthetic
320 m sample (`naip` 1×4×256×256, `s2` 1×6×10×32×32 with `s2_dates`) at 40 m
patches, asserting a 1×768 tile vector, a 1×8×8×768 patch grid, finite values,
and that `output="all"` agrees with both. [`SMOKE.md`](SMOKE.md) is the
Compute2 test on real imagery.

## Related images

- `prithvi-eo`, `terramind`, `clay`, `dofa`, `croma` and `satlas` are other
  Earth-observation encoders in this catalog.

## License

MIT for the code (the repository's `LICENSE`, whose copyright line reads
Damien Robert, 2023) and MIT for the weights (the Hugging Face model card).
The GeoPlex datasets are not redistributed here and carry their own terms.
