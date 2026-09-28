# ssl4eo-landsat

SSL4EO-L encoders: ResNet-18, ResNet-50 and ViT-S/16 backbones pretrained
self-supervised on Landsat imagery, one set per sensor and processing level,
covering Landsat 4–5 TM, Landsat 7 ETM+ and Landsat 8–9 OLI/TIRS. Stewart et
al. built the SSL4EO-L corpus from 250K locations, each imaged in four seasons,
264 × 264 px at 30 m (1M patches per sensor/product), then pretrained each
backbone from ImageNet initialization with MoCo v2 and SimCLR, using
two-season views of the same location as the positive pairs. TorchGeo exposes
the result as weight enums that load into timm models.

The TM and ETM+ sets are the reason for this image. The Landsat data behind the
lab's other Landsat-capable encoders, `clay` and `prithvi-eo`, is Landsat 8–9
only; these weights reach back to Landsat 4–5 and Landsat 7, the route to
change detection from the 1980s onward.

Paper: Stewart et al. (2023), SSL4EO-L: Datasets and Foundation Models for
Landsat Imagery, NeurIPS Datasets and Benchmarks,
[arXiv:2306.09424](https://arxiv.org/abs/2306.09424) ·
Code: [TorchGeo](https://github.com/microsoft/torchgeo) `v0.10.0` ·
Weights: [torchgeo/ssl4eo_landsat](https://huggingface.co/torchgeo/ssl4eo_landsat) @ `1c88bb5`

Pull: `ghcr.io/bradleylab/ssl4eo-landsat:v1`

## Sensors and band order

Each checkpoint expects its sensor's bands in the order below, the Google Earth
Engine Collection 2 band names recorded in each enum's `meta["bands"]`. The
benchmark year is the NLCD/CDL year of the SSL4EO-L benchmark for that sensor;
the pretraining imagery is from other years (TM 2009–2010, ETM+ 2001–2002,
OLI/TIRS 2021–2022).

| Sensor, level | Enum prefix | in_chans | Band order | Benchmark year |
|---|---|---|---|---|
| Landsat 4–5 TM, TOA | `LANDSAT_TM_TOA_` | 7 | B1 B2 B3 B4 B5 B6 B7 | 2011 |
| Landsat 7 ETM+, TOA | `LANDSAT_ETM_TOA_` | 9 | B1 B2 B3 B4 B5 B6_VCID_1 B6_VCID_2 B7 B8 | 2019 |
| Landsat 7 ETM+, SR | `LANDSAT_ETM_SR_` | 6 | SR_B1 SR_B2 SR_B3 SR_B4 SR_B5 SR_B7 | 2019 |
| Landsat 8–9 OLI/TIRS, TOA | `LANDSAT_OLI_TIRS_TOA_` | 11 | B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 | 2019 |
| Landsat 8–9 OLI, SR | `LANDSAT_OLI_SR_` | 7 | SR_B1 SR_B2 SR_B3 SR_B4 SR_B5 SR_B6 SR_B7 | 2019 |

There is no TM SR set: the paper left it out because TM and ETM+ use the same
sensor for the SR bands, which makes the ETM+ SR set the nearest match for
Landsat 4–5 SR. The OLI/TIRS weights cover Landsat 9 (OLI-2/TIRS-2) as well.

**Inputs are 8-bit.** The pretraining patches were converted from float32 to
uint8 by a per-band linear stretch, and each enum's `transforms` then center-crops
to 224 px and divides by 255. For TM TOA the stretch is reflectance 0–0.4 and
brightness temperature 248.15–318.15 K
([`compress_tm_toa.sh`](https://github.com/microsoft/torchgeo/blob/releases/v0.5/experiments/ssl4eo/landsat/compress_tm_toa.sh));
the other sensors' ranges are in the matching `compress_*.sh` beside it. Data
prepared any other way is out of distribution for these encoders.

## Checkpoints

All 30 are baked. Scores are overall accuracy (%) / mIoU from Stewart et al.
(2023) Table 2, for a U-Net decoder trained on the frozen backbone over the
SSL4EO-L NLCD and CDL benchmarks; TorchGeo's `docs/api/weights/landsat.csv`
carries the same values. The mIoU is micro-averaged, which makes it a function of
the accuracy ([`SMOKE.md`](SMOKE.md) shows why).

| Weight enum member | Architecture | Method | in_chans | File | NLCD acc / mIoU | CDL acc / mIoU |
|---|---|---|---|---|---|---|
| `ResNet18_Weights.LANDSAT_TM_TOA_MOCO` | ResNet-18 | MoCo v2 | 7 | `resnet18_landsat_tm_toa_moco-1c691b4f.pth` | 67.65 / 51.11 | 68.70 / 52.32 |
| `ResNet18_Weights.LANDSAT_TM_TOA_SIMCLR` | ResNet-18 | SimCLR | 7 | `resnet18_landsat_tm_toa_simclr-d2d38ace.pth` | 60.86 / 43.74 | 61.94 / 44.86 |
| `ResNet50_Weights.LANDSAT_TM_TOA_MOCO` | ResNet-50 | MoCo v2 | 7 | `resnet50_landsat_tm_toa_moco-ba1ce753.pth` | 68.75 / 53.28 | 69.45 / 53.20 |
| `ResNet50_Weights.LANDSAT_TM_TOA_SIMCLR` | ResNet-50 | SimCLR | 7 | `resnet50_landsat_tm_toa_simclr-a1c93432.pth` | 62.05 / 44.98 | 62.80 / 45.77 |
| `ViTSmall16_Weights.LANDSAT_TM_TOA_MOCO` | ViT-S/16 | MoCo v2 | 7 | `vits16_landsat_tm_toa_moco-a1c967d8.pth` | 67.17 / 50.57 | 67.60 / 51.07 |
| `ViTSmall16_Weights.LANDSAT_TM_TOA_SIMCLR` | ViT-S/16 | SimCLR | 7 | `vits16_landsat_tm_toa_simclr-7c2d9799.pth` | 66.82 / 50.17 | 66.92 / 50.28 |
| `ResNet18_Weights.LANDSAT_ETM_TOA_MOCO` | ResNet-18 | MoCo v2 | 9 | `resnet18_landsat_etm_toa_moco-bb88689c.pth` | 65.22 / 48.39 | 62.84 / 45.81 |
| `ResNet18_Weights.LANDSAT_ETM_TOA_SIMCLR` | ResNet-18 | SimCLR | 9 | `resnet18_landsat_etm_toa_simclr-4d813f79.pth` | 58.76 / 41.60 | 56.47 / 39.34 |
| `ResNet50_Weights.LANDSAT_ETM_TOA_MOCO` | ResNet-50 | MoCo v2 | 9 | `resnet50_landsat_etm_toa_moco-e9a84d5a.pth` | 66.60 / 49.92 | 64.12 / 47.19 |
| `ResNet50_Weights.LANDSAT_ETM_TOA_SIMCLR` | ResNet-50 | SimCLR | 9 | `resnet50_landsat_etm_toa_simclr-70b5575f.pth` | 57.17 / 40.02 | 54.95 / 37.88 |
| `ViTSmall16_Weights.LANDSAT_ETM_TOA_MOCO` | ViT-S/16 | MoCo v2 | 9 | `vits16_landsat_etm_toa_moco-26d19bcf.pth` | 63.75 / 46.79 | 60.88 / 43.70 |
| `ViTSmall16_Weights.LANDSAT_ETM_TOA_SIMCLR` | ViT-S/16 | SimCLR | 9 | `vits16_landsat_etm_toa_simclr-34fb12cb.pth` | 63.33 / 46.34 | 59.06 / 41.91 |
| `ResNet18_Weights.LANDSAT_ETM_SR_MOCO` | ResNet-18 | MoCo v2 | 6 | `resnet18_landsat_etm_sr_moco-4f078acd.pth` | 64.18 / 47.25 | 67.30 / 50.71 |
| `ResNet18_Weights.LANDSAT_ETM_SR_SIMCLR` | ResNet-18 | SimCLR | 6 | `resnet18_landsat_etm_sr_simclr-8e8543b4.pth` | 57.26 / 40.11 | 54.42 / 37.48 |
| `ResNet50_Weights.LANDSAT_ETM_SR_MOCO` | ResNet-50 | MoCo v2 | 6 | `resnet50_landsat_etm_sr_moco-1266cde3.pth` | 64.37 / 47.46 | 62.35 / 45.30 |
| `ResNet50_Weights.LANDSAT_ETM_SR_SIMCLR` | ResNet-50 | SimCLR | 6 | `resnet50_landsat_etm_sr_simclr-e5d185d7.pth` | 57.79 / 40.64 | 55.69 / 38.59 |
| `ViTSmall16_Weights.LANDSAT_ETM_SR_MOCO` | ViT-S/16 | MoCo v2 | 6 | `vits16_landsat_etm_sr_moco-eaa4674e.pth` | 64.09 / 47.21 | 52.37 / 35.48 |
| `ViTSmall16_Weights.LANDSAT_ETM_SR_SIMCLR` | ViT-S/16 | SimCLR | 6 | `vits16_landsat_etm_sr_simclr-a14c466a.pth` | 63.99 / 47.05 | 53.17 / 36.21 |
| `ResNet18_Weights.LANDSAT_OLI_TIRS_TOA_MOCO` | ResNet-18 | MoCo v2 | 11 | `resnet18_landsat_oli_tirs_toa_moco-a3002f51.pth` | 67.82 / 51.30 | 65.74 / 48.96 |
| `ResNet18_Weights.LANDSAT_OLI_TIRS_TOA_SIMCLR` | ResNet-18 | SimCLR | 11 | `resnet18_landsat_oli_tirs_toa_simclr-b0635cc6.pth` | 62.14 / 45.08 | 60.01 / 42.86 |
| `ResNet50_Weights.LANDSAT_OLI_TIRS_TOA_MOCO` | ResNet-50 | MoCo v2 | 11 | `resnet50_landsat_oli_tirs_toa_moco-de7f5e0f.pth` | 69.17 / 52.87 | 67.29 / 50.70 |
| `ResNet50_Weights.LANDSAT_OLI_TIRS_TOA_SIMCLR` | ResNet-50 | SimCLR | 11 | `resnet50_landsat_oli_tirs_toa_simclr-030cebfe.pth` | 64.66 / 47.78 | 62.08 / 45.01 |
| `ViTSmall16_Weights.LANDSAT_OLI_TIRS_TOA_MOCO` | ViT-S/16 | MoCo v2 | 11 | `vits16_landsat_oli_tirs_toa_moco-c7c2cceb.pth` | 67.11 / 50.49 | 64.62 / 47.73 |
| `ViTSmall16_Weights.LANDSAT_OLI_TIRS_TOA_SIMCLR` | ViT-S/16 | SimCLR | 11 | `vits16_landsat_oli_tirs_toa_simclr-ad43e9a4.pth` | 66.12 / 49.39 | 63.88 / 46.94 |
| `ResNet18_Weights.LANDSAT_OLI_SR_MOCO` | ResNet-18 | MoCo v2 | 7 | `resnet18_landsat_oli_sr_moco-660e82ed.pth` | 67.01 / 50.39 | 68.05 / 51.57 |
| `ResNet18_Weights.LANDSAT_OLI_SR_SIMCLR` | ResNet-18 | SimCLR | 7 | `resnet18_landsat_oli_sr_simclr-7bced5be.pth` | 59.93 / 42.79 | 57.44 / 40.30 |
| `ResNet50_Weights.LANDSAT_OLI_SR_MOCO` | ResNet-50 | MoCo v2 | 7 | `resnet50_landsat_oli_sr_moco-ff580dad.pth` | 67.44 / 50.88 | 65.96 / 49.21 |
| `ResNet50_Weights.LANDSAT_OLI_SR_SIMCLR` | ResNet-50 | SimCLR | 7 | `resnet50_landsat_oli_sr_simclr-94f78913.pth` | 63.65 / 46.68 | 60.01 / 43.17 |
| `ViTSmall16_Weights.LANDSAT_OLI_SR_MOCO` | ViT-S/16 | MoCo v2 | 7 | `vits16_landsat_oli_sr_moco-c9b8898d.pth` | 66.81 / 50.16 | 64.17 / 47.24 |
| `ViTSmall16_Weights.LANDSAT_OLI_SR_SIMCLR` | ViT-S/16 | SimCLR | 7 | `vits16_landsat_oli_sr_simclr-4e8f6102.pth` | 65.04 / 48.20 | 62.61 / 45.46 |

`ResNet50_Weights.LANDSAT_ETM_SR_MOCO` and `LANDSAT_ETM_SR_SIMCLR` record
`"model": "resnet18"` in their metadata; the file names, the file sizes and
`torchgeo.models.resnet50`, which ignores that field, all make them ResNet-50.

## What is baked

The 30 checkpoints above, 2,279,333,658 bytes (2.28 GB), in
`/opt/torch-cache/hub/checkpoints/`. `TORCH_HOME` is set to `/opt/torch-cache`,
so TorchGeo's loader finds them there and never reaches for the network. They
were fetched through that same loader at build time, which checks each file
against the SHA-256 prefix in its name.

The Hugging Face repository holds 48 `.pth` files. The 18 not baked are 12
files of about 600 bytes whose `state_dict` is empty and 6 alternate ViT-S/16
files (531 MB together) that no TorchGeo 0.10.0 enum references, so nothing in
TorchGeo can load them.

## Loading a checkpoint

```python
import torch
from torchgeo.models import ResNet50_Weights, resnet50

w = ResNet50_Weights.LANDSAT_TM_TOA_MOCO          # Landsat 4–5 TM, 7 bands
model = resnet50(weights=w, num_classes=0).eval()  # 2048-D pooled embedding

x = torch.randint(0, 256, (1, 7, 264, 264)).float()  # uint8-scaled bands, meta["bands"] order
emb = model(w.transforms(x))
```

`resnet18`, `resnet50` and `vit_small_patch16_224` take the matching enum.
Without `num_classes=0` the timm model keeps a 1000-class head that SSL4EO-L
does not ship, initialized at random. `features_only=True` returns the feature
pyramid for a segmentation decoder. For fine-tuning, `torchgeo.tasks`
(`SemanticSegmentation`, `Classification`) accepts the enum, or its string
name, as `weights=` and can freeze the backbone, as the paper's benchmarks did.

`ssl4eo-check [cpu|cuda]` loads all 30 checkpoints and runs each on synthetic
input, printing versions and output shapes.

## Stack

- Base: `python:3.12-slim-bookworm` (TorchGeo 0.10.0 requires Python ≥ 3.12).
- `torch==2.13.0` and `torchvision==0.28.0` from the CUDA 12.9 index. TorchGeo
  sets no upper bound on torch; this was the newest pair on PyPI when TorchGeo
  0.10.0 was released (2026-08-14), and Compute2's driver (580.105.08) supports
  CUDA 12.9.
- `torchgeo==0.10.0` with its required dependencies (timm, Lightning,
  segmentation-models-pytorch, kornia, rasterio), no optional extras.
- [`constraints.txt`](constraints.txt) freezes the resolution, 96 pins resolved
  2026-09-28 with nothing newer than 2026-08-15, because TorchGeo pins only
  lower bounds.

The build test runs with no network: it imports torch and TorchGeo separately
under faulthandler, asserts the CUDA 12.9 torch build survived the install, and
runs `ssl4eo-check cpu` over every baked checkpoint. [`SMOKE.md`](SMOKE.md) is
the Compute2 test, which checks the GPU path and reruns one published SSL4EO-L
benchmark score.

## Related images

- `clay` — Clay v1.5 MAE encoder over Sentinel-2, Sentinel-1, Landsat 8–9, NAIP,
  LINZ and MODIS: broader sensor coverage, but its Landsat input is Landsat 8–9.
- `prithvi-eo` — Prithvi-EO on Harmonized Landsat–Sentinel-2 (HLS), six bands,
  multi-temporal, run through TerraTorch.
- `dofa` — a single wavelength-conditioned encoder for arbitrary band sets,
  also loaded through TorchGeo; it takes Landsat bands by wavelength rather than
  through a sensor-specific checkpoint.

## License

TorchGeo is MIT. The SSL4EO-L weights and the SSL4EO-L benchmark datasets on
Hugging Face are CC0-1.0, so the weights are baked and may be redistributed.
CC0 requires no attribution; TorchGeo asks users of the SSL4EO-L data to cite
Stewart et al. (2023).
