# geolg-3dfaultnet — Compute2 smoke test

The smallest run that proves the image works on an H100: **one 96³ tile, one
forward pass, plus a random-initialisation control.** Under a minute of GPU
time. It uses the 8 MB sample volume that ships inside the image, so it needs
no data staging and no network beyond the checkpoint.

This is deliberately not the full `inference.py` run. That sweeps 27 tile
positions with 4-way test-time augmentation — 108 forward passes — and proves
nothing the single tile does not.

## What it proves, and what it does not

Proves:

1. The checkpoint downloads (or is present) and matches the recorded sha256.
2. The checkpoint **fully populates** the model — zero missing and zero
   unexpected keys after the `lgfe.`→`lg.` / `fcom.`→`aco.` prefix rewrite.
   This is the load that upstream performs with `strict=False`, where a
   mismatch is silent; see the *Provenance* section of `README.md`.
3. The forward pass runs on the GPU and returns the expected shape.
4. The loaded weights are doing work: the trained model's Dice against the
   shipped ground truth is far above the same model with random weights,
   measured in the same run on the same tile.

Does not prove: that the model is correct, that it generalises off this
synthetic patch, or that the manuscript's numbers are real. Those need the
external benchmark described in `README.md`.

## Prerequisites

Import the image once on a login node:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+geolg-3dfaultnet+v1.sqsh \
    'docker://ghcr.io#bradleylab/geolg-3dfaultnet:v1'
```

Stage the checkpoint on the login node, where there is outbound network:

```bash
mkdir -p /storage3/fs1/alexander.s.bradley/Active/model_weights/geolg-3dfaultnet
curl -L \
  -o /storage3/fs1/alexander.s.bradley/Active/model_weights/geolg-3dfaultnet/best_model.pth \
  https://github.com/letsfly27/GeoLG-3DFaultNet/releases/download/v1.0/best_model.pth
```

The job then verifies that file rather than re-downloading it. If the compute
nodes do have egress, skip this step — `fetch_weights.sh` will fetch it.

## The test script

Write this to `/scratch2/fs1/alexander.s.bradley/geolg/smoke_geolg.py`:

```python
"""One-tile smoke test for geolg-3dfaultnet. Expects ./best_model.pth."""

import os
import time

import numpy as np
import torch

from model import GeoLG3DFaultNet  # PYTHONPATH is set in the image

GEOLG_HOME = os.environ["GEOLG_HOME"]
SHAPE = (128, 128, 128)
TILE = 96  # upstream's default chunk_size

assert torch.cuda.is_available(), "no CUDA device visible to the container"
device = torch.device("cuda")

seis = np.fromfile(f"{GEOLG_HOME}/data/seis/2.dat", dtype=np.float32).reshape(SHAPE)
truth = np.fromfile(f"{GEOLG_HOME}/data/fault/2.dat", dtype=np.float32).reshape(SHAPE)

# Upstream normalises by whole-volume mean/std before tiling; match that.
seis_n = (seis - seis.mean()) / (seis.std() + 1e-6)
tile = (slice(0, TILE),) * 3
x = torch.from_numpy(seis_n[tile]).float()[None, None].to(device)
y_true = truth[tile]


def build(checkpoint=None):
    model = GeoLG3DFaultNet(in_channels=1, num_classes=2).to(device).eval()
    if checkpoint is None:
        return model  # random-initialisation control
    state = checkpoint.get("model_state_dict", checkpoint)
    state = {k.replace("lgfe.", "lg.").replace("fcom.", "aco."): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing keys      : {len(missing)} {missing[:5]}")
    print(f"unexpected keys   : {len(unexpected)} {unexpected[:5]}")
    assert not missing and not unexpected, "checkpoint did not fully populate the model"
    return model


def dice(pred, ref, eps=1e-6):
    return float((2 * (pred * ref).sum() + eps) / (pred.sum() + ref.sum() + eps))


ckpt = torch.load("best_model.pth", map_location=device)
print(f"checkpoint entries: {list(ckpt.keys())[:8]}")

torch.cuda.reset_peak_memory_stats()
model = build(ckpt)
with torch.no_grad():
    start = time.time()
    logits = model(x)
    torch.cuda.synchronize()
    elapsed = time.time() - start
pred = (torch.softmax(logits, dim=1)[0, 1].cpu().numpy() > 0.5).astype(np.float32)

with torch.no_grad():
    control_logits = build(None)(x)
pred_control = (torch.softmax(control_logits, dim=1)[0, 1].cpu().numpy() > 0.5).astype(np.float32)

print(f"device            : {logits.device}")
print(f"output shape      : {tuple(logits.shape)}")
print(f"forward ({TILE}^3)   : {elapsed:.2f} s")
print(f"peak GPU alloc    : {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")
print(
    "fault fraction    : "
    f"truth {y_true.mean():.4f} | trained {pred.mean():.4f} | random {pred_control.mean():.4f}"
)
print(
    "Dice vs truth     : "
    f"trained {dice(pred, y_true):.4f} | random {dice(pred_control, y_true):.4f}"
)
assert tuple(logits.shape) == (1, 2, TILE, TILE, TILE), logits.shape
print("SMOKE OK")
```

## The job

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH -J geolg-smoke
#SBATCH -o geolg-smoke-%j.out

SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+geolg-3dfaultnet+v1.sqsh
WEIGHTS=/storage3/fs1/alexander.s.bradley/Active/model_weights/geolg-3dfaultnet
WORK=/scratch2/fs1/alexander.s.bradley/geolg

srun --container-image="$SQSH" \
     --container-mounts="$WEIGHTS":/opt/geolg-weights,"$WORK":/workspace \
     --container-workdir=/workspace \
     bash -lc 'export PYTHONNOUSERSITE=1; \
               fetch_weights.sh && python /workspace/smoke_geolg.py'
```

15 minutes of walltime is generous — the GPU work is a couple of forward
passes. Most of the wall time is container start and the 24 MB weight load.
`--mem=24G` is host RAM, sized for the numpy volumes and the checkpoint load,
not for the model.

## Reading the output

Pass requires all four:

| Line | Pass condition |
|---|---|
| `[weights] sha256 verified: 67cb47ab…` | present, digest matches |
| `missing keys` / `unexpected keys` | both `0` |
| `device` / `output shape` | `cuda:0` and `(1, 2, 96, 96, 96)` |
| `Dice vs truth` | trained value far above the random value on the same line |

The Dice comparison is the substantive check, and it is written as a
within-run control precisely so that no threshold has to be invented here.
Record both numbers the first time this runs — the trained value on this tile
becomes the regression baseline for future rebuilds, and a rebuild that moves
it has changed something real.

Also record `forward (96^3)` and `peak GPU alloc`: the memory table in
`README.md` is a CPU-RSS proxy pending exactly these two numbers.

## Failure modes

- **`sha256` mismatch.** The release asset changed, or the staged copy is
  truncated. Do not proceed and do not relax the check — re-verify upstream
  first. See *Weights* in `README.md`.
- **Non-zero missing or unexpected keys.** Upstream's key-prefix rewrite no
  longer covers the checkpoint. This is the silent-failure case the test
  exists for; the printed key names say which layers would have run
  randomly initialised.
- **`UnpicklingError` / `WeightsUnpickler error` on `torch.load`.** The
  `weights_only=True` default introduced in torch 2.6. Re-run the job with
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` exported inside the `bash -lc` line,
  but only because `fetch_weights.sh` has just verified the file's digest.
  See *Build notes* in `README.md`.
- **`assert torch.cuda.is_available()` fires.** The job got no GPU, or `srun`
  was given no `--gpus`. Nothing about the image.
- **`ModuleNotFoundError: model`.** `PYTHONPATH` was overridden. The image
  sets it to `/opt/GeoLG-3DFaultNet`.
