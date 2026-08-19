# insar-unwrap — learned InSAR phase unwrapping

Unwraps a Sentinel-1 interferogram with a U-Net instead of a network-flow
solver. Input is a 128×128 patch of wrapped phase (as sin/cos), coherence,
and the three line-of-sight unit-vector components; output is unwrapped
line-of-sight displacement over the same patch. No residues, no branch
cuts, no congruence step.

From Singh & Singh, *When Less Is More: Simplicity Beats Complexity for
Physics-Constrained InSAR Phase Unwrapping* (oral, ML4RS @ ICLR 2026).
The paper's finding is that the plainest of the four architectures wins:
a 7.76 M-parameter vanilla U-Net beats a 17.21 M-parameter multi-scale
hybrid with ASPP and attention gates on every error metric, at roughly
2.5× the speed. Convolutional locality suits smooth deformation fields;
global attention does not buy anything here.

Why the lab might care: we already unwrap with SNAP/SNAPHU. This is the
learned alternative on the same inputs, and the obvious use is as a
cross-check on frames where SNAPHU produces unwrapping errors — the
failure mode is different, so disagreement between the two is
informative.

> **Status: experimental.** Not benchmarked on lab data. The published
> errors are against a held-out split of the authors' own LiCSAR
> training distribution, not against an independent validation set, and
> not on any frame we care about.

## Image tag

`ghcr.io/bradleylab/insar-unwrap:v1` (also `:latest`, `:torch2.5-cu121`)

## Contents

- `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (Python 3.11).
- Upstream source unpacked at a pinned commit under
  `/opt/insar-unwrap/src` (`train/`, `visualize/`, `data/`, `results/`),
  on `PYTHONPATH`. The pinned SHA is also written to
  `/opt/insar-unwrap/UPSTREAM_SHA` and carried in the image labels.
- `rasterio`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `tqdm`,
  `huggingface_hub` — all version-pinned in the Dockerfile.
- `HF_HOME=/opt/hf-cache` — override at runtime to a mounted scratch dir
  so the checkpoints persist across jobs.
- One added compatibility file, `src/train/base_config.py`. It is
  required; see "Caveats".
- No wrapper CLI. Upstream ships training scripts and figure scripts, not
  an inference command, so inference runs from a short Python script (see
  below). Nothing here invents a preprocessing path upstream does not
  define.

## Checkpoints and resources

Eight checkpoints, 1.07 GB total, in two families on one Hugging Face
repo. Both families contain the same four architectures.

| Architecture | Class | Params | Checkpoint | RMSE (cm) | Latency (ms) |
|---|---|---|---|---|---|
| **Vanilla U-Net** (recommended) | `VanillaInSAR_UNet` | 7.76 M | `vanilla_unet_model.pth`, 0.093 GB | **1.070** | **2.74 ± 0.04** |
| Enhanced U-Net (SE blocks) | `EnhancedInSAR_UNet` | 8.29 M | `enhanced_unet_model.pth`, 0.100 GB | 1.325 | 5.97 ± 0.04 |
| Attention U-Net | `AttentionInSAR_UNet` | 11.37 M | `attention_unet_model.pth`, 0.137 GB | 1.439 | 6.81 ± 0.04 |
| Hybrid multi-scale (ASPP) | `HybridMultiScaleUNet` | 17.21 M | `hybrid_model.pth`, 0.207 GB | 1.639 | 6.74 ± 0.05 |

RMSE and latency above are the `standardized/` numbers, from
`results/standardized/{performance,efficiency}_metrics.txt` at the pinned
commit. Latency is per 128×128 patch on the authors' GH200.

**Take the `standardized/` family, not `mixed_precision/`.** The two
prefixes are two training runs, not two precisions of one run.
`mixed_precision/` is the workshop-paper run, in which the four models
were trained under *different* protocols — AMP on for attention/hybrid
and off for vanilla/enhanced, dropout varying 0.0–0.20, two different
weight decays. The reviewer-driven revision retrained all four under one
protocol (FP32 throughout, dropout 0.15, weight decay 1e-4, identical
loss and schedule); that is the `standardized/` family, and it is the
only one where an architecture comparison means anything. The vanilla
U-Net wins under both.

The Hugging Face model card's results table is the `mixed_precision` run
(its latencies match that file exactly), so its numbers will not match
the table above. Its vanilla RMSE of 1.009 cm matches neither results
file in the repo; treat the repo's `results/` files as authoritative.

**GPU is optional for a patch, not for a frame.** A single 128×128 patch
runs in milliseconds on anything, CPU included. A LiCSAR frame tiled at
`PATCH_SIZE=128` / `STRIDE=64` is tens of thousands of patches per
interferogram, and a multi-year stack multiplies that by the number of
dates — which is why the image is CUDA-capable and why the Compute2
recipe below exists.

## Weights are NOT baked

The image ships no checkpoints. Fetch them at run time into
`HF_HOME=/opt/hf-cache`:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Prabhjotschugh/InSAR-Phase-Unwrapping-Models",
    filename="standardized/vanilla_unet_model.pth",
)
```

The repo is public and ungated, so no token is needed.

### Pre-staging the weights on Storage3

Compute nodes should not each re-download 93 MB, and a node without
outbound network cannot. Pull once on a login node into a scratch cache
and mount that cache into the job:

```bash
mkdir -p /scratch2/fs1/alexander.s.bradley/hf-cache
HF_HOME=/scratch2/fs1/alexander.s.bradley/hf-cache \
  python -c 'from huggingface_hub import snapshot_download; \
snapshot_download("Prabhjotschugh/InSAR-Phase-Unwrapping-Models", \
allow_patterns="standardized/*")'
```

Then set `HF_HUB_OFFLINE=1` in the job so a cache miss fails loudly
instead of hanging on a network call the node cannot make.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node, pointing enroot's scratch dirs off the
50 GB home:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+insar-unwrap+v1.sqsh \
    'docker://ghcr.io#bradleylab/insar-unwrap:v1'
```

Then submit a single-GPU job:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+insar-unwrap+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/opt/hf-cache,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; export HF_HUB_OFFLINE=1; \
       python /scratch2/fs1/alexander.s.bradley/scripts/unwrap_frame.py'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
into the container, so a stray `pip install --user` on the login node
would otherwise shadow the container's site-packages. See
`~/.claude/rules/research-infrastructure.md`.

## Minimal inference

```python
import torch
from huggingface_hub import hf_hub_download
from train.standardized.train_vanilla_unet import VanillaInSAR_UNet

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = hf_hub_download(
    repo_id="Prabhjotschugh/InSAR-Phase-Unwrapping-Models",
    filename="standardized/vanilla_unet_model.pth",
)
checkpoint = torch.load(ckpt_path, map_location=device)

model = VanillaInSAR_UNet(in_channels=6, out_channels=1, base_channels=32, dropout=0.0)
model.load_state_dict(checkpoint["model"])
model.to(device).eval()

# Normalization statistics travel with the checkpoint. Reusing them is not
# optional — the model was trained on inputs standardized by exactly these
# values, and recomputing them from your own scene changes the input
# distribution the network sees.
stats = checkpoint["stats"]          # X_mean/X_std: (1, 6, 1, 1); y_mean/y_std: scalars

# patch: (N, 6, 128, 128) float32, channels in this order —
#   0 sin(wrapped phase)   1 cos(wrapped phase)   2 coherence
#   3 LOS east   4 LOS north   5 LOS up
patch = torch.zeros(1, 6, 128, 128)
patch = (patch - stats["X_mean"]) / stats["X_std"]

with torch.inference_mode():
    pred = model(patch.to(device))

los_metres = pred.cpu() * stats["y_std"] + stats["y_mean"]   # (1, 1, 128, 128)
```

Swap the class and filename for another architecture:
`EnhancedInSAR_UNet` / `enhanced_unet_model.pth`,
`AttentionInSAR_UNet` / `attention_unet_model.pth`,
`HybridMultiScaleUNet` / `hybrid_model.pth`, each from
`train.standardized.train_<name>`.

Going from an interferogram to patches, and from patches back to a frame,
is not covered by upstream's published code in a reusable form — the
patch extraction lives inside `prepare_datasets()` in each training
script and is tied to the authors' LiCSAR directory layout. Read
`train/standardized/train_vanilla_unet.py` (`extract_all_patches_with_metadata`,
`prepare_datasets`) before writing a frame-level driver, and match its
coherence mask (`MIN_COHERENCE = 0.5`) and stride, or the model sees
inputs it was not trained on.

## Licensing

- **Code: MIT** (`prabhjotschugh/When-Less-is-More-InSAR-Phase-Unwrapping`).
  The upstream `LICENSE` is preserved at
  `/opt/insar-unwrap/src/LICENSE` inside the image.
- **Weights: CC-BY-4.0**, ungated
  (`Prabhjotschugh/InSAR-Phase-Unwrapping-Models`). Attribution is
  required for any published result — cite the ML4RS/ICLR 2026 paper.
- The two differ, so the image label is `MIT AND CC-BY-4.0`. Note that
  CC-BY applies to the weights only, which is why they are fetched rather
  than redistributed inside the image.

```bibtex
@inproceedings{singh2026when,
  title     = {When Less Is More: Simplicity Beats Complexity for
               Physics-Constrained {InSAR} Phase Unwrapping},
  author    = {Singh, Prabhjot and Singh, Manmeet},
  booktitle = {4th ICLR Workshop on Machine Learning for Remote Sensing},
  year      = {2026},
  url       = {https://openreview.net/forum?id=liJldeR5ZX}
}
```

## Caveats

- **Upstream's own imports are broken at the pinned commit, and this
  image patches them.** Every `train/standardized/train_*.py` does
  `sys.path.insert(0, <repo>/train)` then
  `from train.base_config import BaseConfig, print_protocol_banner`, but
  `base_config.py` sits in `train/standardized/`. The 2026-07-29
  reorganization moved the file without updating the import, so importing
  any of the four model classes fails with `ModuleNotFoundError: No
  module named 'train.base_config'`. The Dockerfile adds a one-statement
  `train/base_config.py` that re-exports from `standardized.base_config`
  — a module path, not a copy of the hyperparameters, so upstream edits
  to the config still take effect. Drop that layer when upstream fixes
  the import.

- **`visualize/result_*.py` does not run here, and is not worth fixing.**
  Those scripts import `train.train_vanilla_unet` (also a stale path) and
  then call `prepare_datasets()`, which needs the authors' 350-frame
  LiCSAR training corpus. They regenerate the paper's figures from the
  training data; they are not an inference entrypoint. The shim above
  deliberately does not cover them.

- **The Hugging Face model card's usage snippet is stale in two places.**
  It downloads `filename="vanilla_unet_model.pth"` — there is no file at
  the repo root; the real paths are `standardized/vanilla_unet_model.pth`
  and `mixed_precision/vanilla_unet_model.pth`. It also does
  `sys.path.append('./train')` + `from train_vanilla_unet import ...`,
  which predates the reorganization. Follow the inference block above
  instead.

- **`torch.load` and `weights_only`.** Upstream calls `torch.load` with
  no `weights_only` argument, which torch 2.6 flipped to `True` by
  default — normally a breaking change for `.pth` checkpoints. Checked
  rather than assumed: `standardized/vanilla_unet_model.pth` loads
  cleanly under the `weights_only=True` default on torch 2.10, because
  its payload (`epoch`, `model`, `optimizer`, `stats`, `config`,
  `best_val_loss`) is tensors and plain Python types only. The torch 2.5.1
  pin is for ABI coherence with the lab's other GPU images, not to dodge
  this.

- **`cartopy` is in upstream's `requirements.txt` and is not installed.**
  Its only importer is `figures/map_plot.py`, which redraws the paper's
  map of frame locations; it is on no inference path and pulls GEOS/PROJ.
  That script will not run in this image.

- **Trained on LiCSAR C-band Sentinel-1 only** — 350 interferograms,
  20 frames, 2020–2025, `WAVELENGTH = 0.056`. Nothing establishes that it
  transfers to L-band (NISAR, ALOS), to X-band, or to deformation regimes
  outside the training frames. Treat outputs on a new frame as a
  hypothesis to check against SNAPHU, not as a replacement for it.

- **The published errors are in-distribution.** RMSE ≈ 1 cm is against a
  held-out split of the same LiCSAR corpus. There is no independent
  validation set in the repo.
