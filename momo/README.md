# momo

[kerner-lab MOMO](https://github.com/kerner-lab/MOMO) — Mars Orbital
foundation model. Vision Transformer pre-trained on ~12M samples
across three Martian orbital sensors (HiRISE 0.25 m/px, CTX 5 m/px,
THEMIS 100 m/px), distributed via
[`Mirali33/MOMO`](https://huggingface.co/Mirali33/MOMO) on Hugging
Face Hub. Ships as a single multi-sensor checkpoint plus three
sensor-specific checkpoints, in ViT-Small / ViT-Base / ViT-Large
variants.

**Mars-Bench** is the separate benchmark MOMO is evaluated on (paper
arXiv 2510.24010; task datasets under
[`Mirali33/mars-bench-*`](https://huggingface.co/Mirali33) on HF) —
9 downstream tasks (4 classification + 5 segmentation): crater
segmentation, boulder detection, dust devils, S5Mars rover surface,
DoMars16k landmark classification, etc. The `kerner-lab/MOMO` repo (and
this container) ships the fine-tuning engine + data loaders to run
those tasks, not the benchmark datasets themselves.

GPU-primary (H100 sm_90 via cu121 wheels). ViT-Base is small enough
to run on a laptop GPU for single-image demos, but the canonical use
is batch processing across HiRISE / CTX / THEMIS image strips where
H100 throughput matters.

## Image tag

`ghcr.io/bradleylab/momo:v1` (also `:latest`, `:torch2.5-cu121`)

## Stack

- Base: `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`
- Python 3.11
- PyTorch 2.5.1 + torchvision 0.20.1 (cu121, sm_90)
- `kerner-lab/MOMO` at pinned commit `a837ab5` (full
  `requirement.txt` baked: pytorch-lightning, hydra-core,
  segmentation-models-pytorch, albumentations, rasterio, shapely,
  scikit-image, scikit-learn, imbalanced-learn, timm 0.6.12, einops,
  lpips, lxml, ...)

`WANDB_MODE=offline` is set so accidental wandb auth attempts don't
block runs.

## Available pretrained variants

| Variant | File on HF Hub | Approx. size |
|---|---|---|
| ViT-Base 16, multi-sensor (recommended) | `vit-b-16/momo.pth` | ~340 MB |
| ViT-Base 16, HiRISE-only | `vit-b-16/hirise.pth` | ~340 MB |
| ViT-Base 16, CTX-only | `vit-b-16/ctx.pth` | ~340 MB |
| ViT-Base 16, THEMIS-only | `vit-b-16/themis.pth` | ~340 MB |
| ViT-Small 16 | `vit-s-16/<variant>.pth` | smaller |
| ViT-Large 16 | `vit-l-16/<variant>.pth` | larger |

Default recommendation: `vit-b-16/momo.pth` — the sensor-merged
backbone is what Mars-Bench evaluates against in the upstream paper.

## Weights

Not baked. The first call to
`hf_hub_download("Mirali33/MOMO", "vit-b-16/momo.pth")` downloads into
`$HF_HOME=/opt/hf-cache`. Bind-mount a persistent host dir so each
variant only downloads once per host:

```bash
docker run --rm -it --gpus all \
  -v "$PWD/hf-cache:/opt/hf-cache" \
  -v "$PWD/data:/data" \
  ghcr.io/bradleylab/momo:v1
```

Code license: MIT (per upstream `pyproject.toml`). Weights license:
CC-BY-4.0 (per the Hugging Face model card).

## Inference

```python
import torch
from huggingface_hub import hf_hub_download
from models import models_vit  # MOMO's ViT factory module

# 1. Pull the multi-sensor ViT-Base checkpoint
ckpt_path = hf_hub_download(
    repo_id="Mirali33/MOMO",
    filename="vit-b-16/momo.pth",
)

# 2. Load. Upstream stores raw torch state dicts via torch.save —
# weights_only=False is required because the file may include
# non-tensor metadata.
state_dict = torch.load(ckpt_path, map_location="cuda", weights_only=False)

# 3. Build a matching ViT and load
model = models_vit.vit_base_patch16(num_classes=0).cuda().eval()
model.load_state_dict(state_dict, strict=False)

# 4. Run on an input tile (3, H, W) -- any RGB image, sized to ViT's
# expected resolution; consult upstream for sensor-specific
# normalization.
import torch
x = torch.randn(1, 3, 224, 224, device="cuda")
with torch.no_grad():
    feats = model.forward_features(x)
print("feat shape:", feats.shape)
```

For the Mars-Bench downstream task heads, follow the upstream's
`datasets_finetune` + Hydra config pattern at
[kerner-lab/MOMO](https://github.com/kerner-lab/MOMO).

## Inputs

- RGB image arrays sized to ViT's expected resolution (default
  `224x224` for `vit_b_16`). Upstream provides sensor-specific
  preprocessing in `utils/`.
- For full Mars-Bench evaluation, follow the upstream config-driven
  pipeline (Hydra `+experiment=...`).

## Run on Compute2

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-gpu \
       --gpus=1 \
       --mem=32G \
       --time=02:00:00 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+momo+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/opt/hf-cache,/scratch2/fs1/alexander.s.bradley/mars-tiles:/data \
         bash -lc "export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/momo_embed.py"'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — see
`~/.claude/rules/research-infrastructure.md`.

## Limitations

- **First-mover territory.** MOMO is recent (referenced as Oct 2025 /
  Apr 2026 on the [model card](https://huggingface.co/Mirali33/MOMO)).
  HF disables download tracking on this model, so we have no public
  adoption signal — "likes" sit at 0 as of 2026-05-03. Expect to
  discover sharp edges around dataset preprocessing, normalization
  conventions, and Mars-Bench config quirks.
- **Generic top-level packages.** Upstream installs `models`, `utils`,
  `datasets_finetune` directly into site-packages without a `momo/`
  namespace. Inside this container that's fine; if you ever pip
  install it on a host alongside other tools, watch for collisions
  with packages of the same name.
- **Inference vs Mars-Bench evaluation.** The container is shipped as
  the inference / fine-tuning environment, not as a turnkey "run all
  9 Mars-Bench tasks." For the latter, follow the upstream Hydra
  recipes.
- **Code vs weights licensing differ.** Code MIT, weights CC-BY-4.0.
  Cite the model card if you publish results.
