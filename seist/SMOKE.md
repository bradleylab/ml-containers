# seist — Compute2 smoke test

> [!warning] CPU-partition runs need `NVIDIA_VISIBLE_DEVICES=void`
> Every CUDA-base image in this repo sets `NVIDIA_VISIBLE_DEVICES`, and enroot's
> NVIDIA hook then tries to inject a driver on whatever node it lands on. On
> `general-cpu` there is none, so the container fails to *start*:
> `nvidia-container-cli: initialization error: nvml error: driver not loaded`.
>
> Add `--export=ALL,NVIDIA_VISIBLE_DEVICES=void` to any `srun` on `general-cpu`.
> `void` tells libnvidia-container to skip the hook. Verified 2026-08-21.

The smallest run that proves the image works on a compute node. Because
the checkpoints ship inside the image, there is nothing to pre-stage and
nothing to download — one CPU job is the whole test.

## Prerequisites

Import the image once on a login node (see README for the enroot env
vars). No weight staging, no `HF_HOME` mount, no network on the node.

## Stage 1 — the smoke test (CPU, no network)

**Input:** none. The script builds five models from the in-image
checkpoints and pushes a zero waveform through each. Zeros are the point:
the test is whether the checkpoints load and the graphs run, not whether
the predictions mean anything.

```bash
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     -A compute2-alexander.s.bradley -p general-cpu \
     --cpus-per-task=2 --mem=8G --time=00:10:00 \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+seist+v1.sqsh \
     bash -lc 'export PYTHONNOUSERSITE=1; python /opt/seist/smoke_test.py'
```

**Output proving success:**

```
torch 2.5.1 | cuda-build 12.1
timm 0.9.2 | obspy 1.4.0 | h5py 3.12.1 | numpy 1.26.4 | pandas 2.2.3 | matplotlib 3.9.2
checkpoints: 18 files, 29.3 MB in /opt/seist/pretrained
seist_m_dpk    <- seist_m_dpk_diting.pth     params= 380,805 out=(1, 3, 8192)
seist_m_pmp    <- seist_m_pmp_diting.pth     params= 312,140 out=(1, 2)
seist_m_emg    <- seist_m_emg_diting.pth     params= 312,043 out=(1, 1)
seist_m_baz    <- seist_m_baz_diting.pth     params= 312,043 out=(1, 1)
seist_m_dis    <- seist_m_dis_diting.pth     params= 312,043 out=(1, 1)
SMOKE OK
```

The three claims that matter, in order:

1. `checkpoints: 18 files, 29.3 MB` — the in-repo weights survived the
   image build and the `.sqsh` import.
2. Each `load_state_dict` is implicit and strict; a line printing at all
   means that checkpoint matched its architecture exactly.
3. `SMOKE OK` — all four task families (picking, polarity, magnitude,
   back-azimuth, distance) are reachable.

**Expected runtime:** under 1 minute of job time; the models are
0.3–0.4 M parameters and there is no I/O.

## Stage 2 — GPU check (only if you intend to fine-tune)

Inference does not need a GPU, so skip this unless you are about to
submit a training job.

```bash
srun -A compute2-alexander.s.bradley -p general-gpu \
     --gpus=1 --cpus-per-task=4 --mem=16G --time=00:10:00 \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+seist+v1.sqsh \
     bash -lc 'export PYTHONNOUSERSITE=1; python - <<PY
import torch
from models import create_model, load_checkpoint

assert torch.cuda.is_available(), "no GPU visible to the job"
print("gpu:", torch.cuda.get_device_name(0))

device = torch.device("cuda")
model = create_model("seist_m_emg", in_channels=3)
ckpt = load_checkpoint("/opt/seist/pretrained/seist_m_emg_diting.pth", device=device)
print("load_state_dict:", model.load_state_dict(
    ckpt["model_dict"] if "model_dict" in ckpt else ckpt, strict=True))
model.to(device).eval()

with torch.inference_mode():
    out = model(torch.zeros(1, 3, 8192, device=device))
print("out:", tuple(out.shape), "| finite:", bool(torch.isfinite(out).all()))
assert tuple(out.shape) == (1, 1)
print("SMOKE OK")
PY'
```

**Output proving success:** `gpu: NVIDIA H100 ...`, then
`load_state_dict: <All keys matched successfully>`, `out: (1, 1) |
finite: True`, and `SMOKE OK`.

**Expected runtime:** 1–2 minutes, nearly all of it CUDA context setup.

## Failure modes worth recognising

| Symptom | Cause |
|---|---|
| `AssertionError` on the checkpoint count | the `pretrained/` directory did not survive the build or the `.sqsh` import — rebuild, do not work around it |
| `ModuleNotFoundError: No module named 'models'` | `PYTHONPATH=/opt/seist` was overridden by the job environment |
| `load_state_dict` reports missing/unexpected keys | the pinned source SHA and the checkpoints have drifted apart — impossible unless someone edited the image |
| smoke test passes on the login node but not in the job | almost always `PYTHONNOUSERSITE=1` missing, letting `$HOME/.local` shadow site-packages |
