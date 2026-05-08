# terramind

[TerraMind 1.0](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
(IBM, ESA Φ-lab, FAST-EO; ICCV 2025) is the first any-to-any
generative foundation model for Earth Observation. A single
multimodal backbone is pretrained on Sentinel-1 GRD, Sentinel-1 RTC,
Sentinel-2 L2A, DEM, NDVI, and LULC. Use cases: shared embedding
extraction across sensors, cross-sensor fine-tuning (S1 + S2 fusion),
generation of one modality from another (e.g. infer NDVI from S1
when S2 is cloud-blocked), and "Thinking-in-Modalities" fine-tuning
where the model first generates a missing modality as an intermediate
step before predicting the downstream task.

This container ships [TerraTorch](https://github.com/terrastackai/terratorch),
the IBM-supported fine-tuning toolkit that wraps TerraMind (and
Prithvi-EO) behind the `BACKBONE_REGISTRY` + Lightning task
scaffolding, plus the `diffusers==0.30.0` pin TerraMind requires for
its any-to-any generation pipeline. Weights are NOT baked — they
download from HF Hub on first call.

GPU-primary (H100 sm_90 by stack). The `tiny` and `small` variants
run on a laptop GPU with 8+ GB VRAM; `base` and `large` want
H100-class for fine-tuning, and inference benefits from GPU even
when not strictly required.

## Image tag

`ghcr.io/bradleylab/terramind:v1` (also `:latest`,
`:torch2.5-cu121`)

## Stack

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11
- PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `terratorch >= 1.2.5` (TerraMind backbones registered through TerraTorch)
- `diffusers == 0.30.0` — exact pin per upstream `IBM/terramind`
  README; newer diffusers break the any-to-any generation pipeline
- Lightning, segmentation-models-pytorch, torchgeo, timm, geopandas,
  rasterio, albumentations (transitive — see `terratorch`'s pyproject)

`WANDB_MODE=offline` and `NO_ALBUMENTATIONS_UPDATE=true` in ENV so
neither service blocks runs with a network probe at import time.

## Weights

Not baked. Pull from HF Hub on first call:

| Variant | HF Hub repo | Approx. size |
|---|---|---|
| Tiny | [`ibm-esa-geospatial/TerraMind-1.0-tiny`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-tiny) | ~50 MB |
| Small | [`ibm-esa-geospatial/TerraMind-1.0-small`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-small) | ~250 MB |
| Base | [`ibm-esa-geospatial/TerraMind-1.0-base`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base) | ~700 MB |
| Large | [`ibm-esa-geospatial/TerraMind-1.0-large`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-large) | ~1.7 GB |

Plus six tokenizers under the same HF org (S1GRD, S1RTC, S2L2A,
DEM, NDVI, LULC) for any-to-any generation.

`HF_HOME=/opt/hf-cache` — bind-mount that path for cross-run weight
persistence and so cluster nodes can share a cache.

## Backbone names

Registered in TerraTorch's `BACKBONE_REGISTRY`:

```
terramind_v1_tiny       terramind_v1_tiny_tim
terramind_v1_small      terramind_v1_small_tim
terramind_v1_base       terramind_v1_base_tim
terramind_v1_large      terramind_v1_large_tim
```

The `_tim` suffix selects the Thinking-in-Modalities variant — the
backbone first generates additional modalities (default LULC) before
producing its embedding.

## Quickstart

```bash
docker run --rm --gpus all \
  -v /path/to/data:/work \
  -v /shared/hf-cache:/opt/hf-cache \
  ghcr.io/bradleylab/terramind:v1 bash
```

Inside the container:

```python
import torch
from terratorch import BACKBONE_REGISTRY

# 12-band Sentinel-2 L2A input, B x C x H x W float32
x = torch.randn(1, 12, 224, 224, device="cuda")
backbone = BACKBONE_REGISTRY.build(
    "terramind_v1_base", pretrained=True, modalities=["S2L2A"]
).cuda().eval()
with torch.no_grad():
    embeddings = backbone({"S2L2A": x})
print({k: v.shape for k, v in embeddings.items()})
```

For fine-tuning via the `terratorch` CLI, see the upstream
[config examples](https://github.com/IBM/terramind/tree/main/configs)
(Sen1Floods11, HLS Burn Scars, Multitemporal Crop).

## Why this container exists

TerraMind is a natural pair to `bradleylab/prithvi-eo` —
both are TerraTorch-fronted and serve as backbones for downstream
geospatial tasks. TerraMind earns a separate container because:

- The four scale variants ship with the *same* TerraTorch entry
  points but different config + tokenizer dependencies.
- The `diffusers==0.30.0` pin TerraMind requires conflicts with
  newer diffusers releases; isolating it inside its own image
  prevents unrelated containers from being constrained by it.
- Multimodal pretraining (S1 + S2 + DEM + NDVI + LULC) covers a
  superset of Prithvi-EO's HLS-only training — for any project
  that wants S1+S2 fusion, TerraMind is the right starting point.

## License

Code: Apache-2.0 (this Dockerfile, `IBM/terramind`,
`terrastackai/terratorch`).
Weights: Apache-2.0 (per
[`ibm-esa-geospatial/TerraMind-1.0-base`](https://huggingface.co/ibm-esa-geospatial/TerraMind-1.0-base)
model card; verify per checkpoint before redistribution).

## References

- Jakubik et al. (2025) "TerraMind: Large-Scale Generative
  Multimodality for Earth Observation", ICCV 2025
  [arXiv:2504.11171](https://arxiv.org/abs/2504.11171).
- Upstream code: [IBM/terramind](https://github.com/IBM/terramind)
  (config examples, fine-tuning notebooks).
- TerraTorch toolkit:
  [terrastackai/terratorch](https://github.com/terrastackai/terratorch)
  (backbone registry + Lightning task heads).
