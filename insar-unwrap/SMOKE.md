# insar-unwrap — Compute2 smoke test

The smallest run that proves the image works on a compute node. It does
not prove the science; it proves the container, the checkpoint, and the
GPU are all reachable from a job.

Two stages: a no-weights architecture check that needs no network and no
GPU, then a single-patch forward pass with the real checkpoint on an
H100.

## Prerequisites

Import the image once on a login node (see README for the enroot env
vars) and pre-stage the weights:

```bash
mkdir -p /scratch2/fs1/alexander.s.bradley/hf-cache
HF_HOME=/scratch2/fs1/alexander.s.bradley/hf-cache \
  python -c 'from huggingface_hub import snapshot_download; \
snapshot_download("Prabhjotschugh/InSAR-Phase-Unwrapping-Models", \
allow_patterns="standardized/vanilla_unet_model.pth")'
```

That is one 93 MB file. Everything below then runs offline.

## Stage 1 — architecture check, no weights, no GPU

**Input:** none.

```bash
srun -A compute2-alexander.s.bradley -p general-cpu \
     --cpus-per-task=2 --mem=8G --time=00:10:00 \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+insar-unwrap+v1.sqsh \
     bash -lc 'export PYTHONNOUSERSITE=1; export HF_HUB_OFFLINE=1; \
       python /opt/insar-unwrap/smoke_test.py'
```

**Output proving success** — the last line is `SMOKE OK`, preceded by
four lines with these exact parameter counts:

```
VanillaInSAR_UNet        params= 7,763,905  out=(1, 1, 128, 128)
EnhancedInSAR_UNet       params= 8,287,088  out=(1, 1, 128, 128)
AttentionInSAR_UNet      params=11,372,820  out=(1, 1, 128, 128)
HybridMultiScaleUNet     params=17,206,128  out=(1, 1, 128, 128)
SMOKE OK
```

**Expected runtime:** under 1 minute once the job starts (this is the
same check the build already ran, so a failure here means the node
environment, not the image).

## Stage 2 — one patch through the real checkpoint, on a GPU

**Input:** the pre-staged `standardized/vanilla_unet_model.pth` and a
synthetic 6×128×128 patch of zeros. Zeros are the point: the test is
whether the weights load and the graph runs on an H100, not whether the
prediction is meaningful.

```bash
srun -A compute2-alexander.s.bradley -p general-gpu \
     --gpus=1 --cpus-per-task=4 --mem=16G --time=00:15:00 \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+insar-unwrap+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/opt/hf-cache \
     bash -lc 'export PYTHONNOUSERSITE=1; export HF_HUB_OFFLINE=1; python - <<PY
import torch
from huggingface_hub import hf_hub_download
from train.standardized.train_vanilla_unet import VanillaInSAR_UNet

assert torch.cuda.is_available(), "no GPU visible to the job"
print("gpu:", torch.cuda.get_device_name(0))

path = hf_hub_download(
    repo_id="Prabhjotschugh/InSAR-Phase-Unwrapping-Models",
    filename="standardized/vanilla_unet_model.pth",
)
ckpt = torch.load(path, map_location="cuda")
model = VanillaInSAR_UNet(6, 1, base_channels=32, dropout=0.0)
print("load_state_dict:", model.load_state_dict(ckpt["model"], strict=True))
model.cuda().eval()

stats = ckpt["stats"]
x = torch.zeros(1, 6, 128, 128)
x = ((x - stats["X_mean"]) / stats["X_std"]).cuda()
with torch.inference_mode():
    pred = model(x)
los_cm = (pred.cpu() * stats["y_std"] + stats["y_mean"]) * 100

print("trained epochs:", ckpt["epoch"] + 1)
print("out:", tuple(pred.shape), "| LOS cm range:",
      float(los_cm.min()), float(los_cm.max()))
assert tuple(pred.shape) == (1, 1, 128, 128)
print("SMOKE OK")
PY'
```

**Output proving success:**

- `gpu: NVIDIA H100 ...`
- `load_state_dict: <All keys matched successfully>` — this is the line
  that matters. It means the checkpoint's state dict and the pinned
  source's architecture agree.
- `trained epochs: 534` for `standardized/vanilla_unet_model.pth` at the
  current HF revision.
- `out: (1, 1, 128, 128)` and a finite LOS range.
- `SMOKE OK`

**Expected runtime:** 1–2 minutes of job time, most of it CUDA context
setup and the 93 MB checkpoint read. The forward pass itself is
milliseconds.

## Failure modes worth recognising

| Symptom | Cause |
|---|---|
| `ModuleNotFoundError: No module named 'train.base_config'` | the compat shim layer was lost — the image is broken, rebuild |
| `LocalEntryNotFoundError` under `HF_HUB_OFFLINE=1` | weights were not pre-staged, or `--container-mounts` did not land on `/opt/hf-cache` |
| `load_state_dict` reports missing/unexpected keys | the pinned source SHA and the checkpoint revision have drifted apart |
| smoke test passes on the login node but not in the job | almost always `PYTHONNOUSERSITE=1` missing, letting `$HOME/.local` shadow site-packages |
