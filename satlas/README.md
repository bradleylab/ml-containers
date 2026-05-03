# satlas

Pre-trained foundation models for satellite and aerial imagery from
Allen AI's SatlasPretrain (Bastani et al., ICCV 2023). Backbones are
pre-trained on a large remote-sensing corpus and exposed via the
`satlaspretrain_models` loader package.

GPU-primary (H100 sm_90 via cu121 wheels) since the canonical use is
batch backbone-feature extraction across many tiles. CPU inference
works too — pass `device='cpu'` to `get_pretrained_model`.

## Image tag

`ghcr.io/bradleylab/satlas:v1` (also `:latest`,
`:torch2.5-cu121`)

## Stack

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11
- PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `satlaspretrain-models >= 0.3.1`
- Pillow, requests

## Available pretrained variants

| Sensor | Backbones | Variants |
|---|---|---|
| Sentinel-2 RGB | Swin-v2-Base, Swin-v2-Tiny, ResNet50, ResNet152 | SI, MI |
| Sentinel-2 9-band MS | Swin-v2-Base, Swin-v2-Tiny, ResNet50, ResNet152 | SI, MI |
| Sentinel-1 (VH+VV) | Swin-v2-Base | SI, MI |
| Landsat 8/9 (all bands) | Swin-v2-Base | SI, MI |
| Aerial (0.5–2 m/px RGB) | Swin-v2-Base | SI, MI |

`SI` = single-image, `MI` = multi-image (temporal-max-pool over a
short stack). Checkpoint IDs follow `<Sensor>_<Backbone>_<SI|MI>_<Modality>`,
e.g. `Sentinel2_SwinB_SI_RGB`. Full list and direct download links at
[allenai/satlaspretrain_models](https://github.com/allenai/satlaspretrain_models).

## Weights

Not baked. `Weights().get_pretrained_model(checkpoint_id)` fetches
the `.pth` from `allenai/satlas-pretrain` on HF Hub each call. **The
upstream loader does NOT cache on disk** — it streams the weight file
into RAM via `requests.get` + `BytesIO` and loads from there. For
repeated jobs across the same checkpoint, either:

1. **Pre-download once** to a host directory and point a wrapper at
   the local file (`torch.load(local_path)` then `Model(...)` directly), or
2. **Bind-mount** a host directory and write a small wrapper that
   checks for a cached copy before delegating to `Weights()`.

Direct download (variant `Sentinel2_SwinB_SI_RGB`):

```bash
wget -O /opt/satlas-cache/sentinel2_swinb_si_rgb.pth \
  https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_rgb.pth
```

Checkpoint license: [ODC-BY](https://github.com/allenai/satlas/blob/main/DataLicense).

## Inference

```python
import torch, satlaspretrain_models

w = satlaspretrain_models.Weights()
model = w.get_pretrained_model(
    "Sentinel2_SwinB_SI_RGB",   # checkpoint id
    fpn=True,                    # also load the FPN
    device="cuda",               # or "cpu"
)
model.eval()

# Output is the multi-scale feature map. Wire your downstream head
# (classification / detection / segmentation / regression) onto it.
feature_maps = model(torch.randn(1, 3, 512, 512, device="cuda"))
```

Sentinel-2 pixel normalization, band ordering, and tile size
conventions are documented in the upstream
[Normalization.md](https://github.com/allenai/satlas/blob/main/Normalization.md).

## Run on Compute2

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-gpu \
       --gpus=1 \
       --mem=32G \
       --time=02:00:00 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+satlas+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/satlas-cache:/opt/satlas-cache,/scratch2/fs1/alexander.s.bradley/inputs:/inputs,/scratch2/fs1/alexander.s.bradley/outputs:/outputs \
         bash -lc "export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/satlas_embed.py"'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — see
`~/.claude/rules/research-infrastructure.md`.

## Limitations

- The upstream loader's lack of on-disk caching is a real pain for
  repeated jobs. Always pre-download or wrap the loader.
- These are **backbones**, not task heads — fine-tuning for a
  specific downstream task (which buildings, which infrastructure,
  etc.) requires labelled data and a training step. Any framing of
  this image as "infrastructure detection out of the box" actually
  refers to the separate downstream models hosted at
  [allenai/satlas](https://github.com/allenai/satlas), which is a
  different package (not yet containerized here).
- `device='cuda'` is the default — pass `device='cpu'` explicitly on
  CPU-only systems.
