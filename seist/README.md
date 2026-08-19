# seist — multi-task seismogram transformer

One backbone, six tasks, from a single three-component waveform window:
event detection, P/S phase picking, first-motion polarity, magnitude,
back-azimuth, and epicentral distance. From Li et al., *SeisT: A
Foundational Deep-Learning Model for Earthquake Monitoring Tasks* (IEEE
TGRS 62, 2024).

## Read this before pulling it: how it relates to `seisbench`

The lab already runs `seisbench` (PhaseNet, EQTransformer) and that is
the right tool for detection and phase picking. **SeisT is not a second
picker.** It is here for the four things SeisBench does not do:

| Task | Covered by `seisbench`? | SeisT abbreviation |
|---|---|---|
| Event detection | yes | `dpk` |
| P/S phase picking | yes | `dpk` |
| **First-motion polarity** | no | `pmp` |
| **Magnitude** | no | `emg` |
| **Back-azimuth** | no | `baz` |
| **Epicentral distance** | no | `dis` |

The intended pairing is: pick with SeisBench, then run SeisT's `pmp` /
`emg` / `baz` / `dis` heads on the same windows to get polarity (for
focal mechanisms), a per-station magnitude, and a single-station location
constraint. SeisT's `dpk` head exists and is competitive in the paper,
but there is no reason to switch pickers on that basis alone.

> **Status: experimental.** Not benchmarked on lab data. The shipped
> weights are trained on DiTing (China) and PNW; see "Caveats" for why
> magnitude and back-azimuth in particular should not be trusted on a new
> region without retraining.

## Image tag

`ghcr.io/bradleylab/seist:v1` (also `:latest`, `:torch2.5-cu121`)

## Contents

- `pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime` (Python 3.11).
- Upstream source unpacked at a pinned commit under `/opt/seist`, on
  `PYTHONPATH`. The SHA is written to `/opt/seist/UPSTREAM_SHA` and
  carried in the image labels.
- **All 18 pretrained checkpoints, in-image at `/opt/seist/pretrained/`**
  (29.3 MB total). Nothing is downloaded at run time; the image works on
  a node with no outbound network.
- `timm` (one symbol: `DropPath`), `obspy` (one symbol: `trigger_onset`),
  `h5py`, `numpy`, `pandas`, `matplotlib`, `tensorboard`, `GPUtil` — all
  version-pinned in the Dockerfile.
- Upstream's `demo_predict.py` (a real inference entrypoint) and `main.py`
  (train / fine-tune / test driver, single-GPU and `torchrun` DDP).
- `smoke_test.py`, the build-time check, re-runnable on a node — see
  `SMOKE.md`.

## Weights: baked, and why that is the right call here

Most images in this repo fetch weights at run time and say so loudly. This
one ships them, deliberately.

The checkpoints are **committed inside the upstream Git repository**,
under `pretrained/`. They are not a separate artifact that happens to be
convenient to bake — they are files in the source tree, and the SHA-pinned
source checkout is what puts them in the image. Excluding them would mean
deleting files out of a pinned checkout and then re-hosting them
ourselves, because there is no Hugging Face repo, no release asset, and no
other out-of-band source for SeisT weights. That trades a self-contained
29.3 MB image for a lab-maintained mirror that can drift from the pinned
commit. Not worth it.

The two things the no-baked-weights convention exists to prevent do not
apply: the payload is 29.3 MB, about 1% of the image, so it neither
bloats the `.sqsh` nor makes the build slow; and there is no gated or
restrictively-licensed artifact being redistributed — everything is MIT,
code and weights alike. Baking also buys a real build-time guarantee: the
smoke test asserts all 18 checkpoints are present and that one per task
family loads `strict=True` into its model, which is only testable because
they are in the image.

### The 18 checkpoints

Three backbone sizes (S / M / L) × five task heads, plus a second
magnitude model trained on PNW instead of DiTing.

| Task | Head | Train set | Files |
|---|---|---|---|
| Detection & phase picking | `dpk` | DiTing | `seist_{s,m,l}_dpk_diting.pth` |
| First-motion polarity | `pmp` | DiTing | `seist_{s,m,l}_pmp_diting.pth` |
| Magnitude | `emg` | DiTing | `seist_{s,m,l}_emg_diting.pth` |
| Magnitude | `emg` | PNW | `seist_{s,m,l}_emg_pnw.pth` |
| Back-azimuth | `baz` | DiTing | `seist_{s,m,l}_baz_diting.pth` |
| Epicentral distance | `dis` | DiTing | `seist_{s,m,l}_dis_diting.pth` |

## Resources

| Backbone | Params (`dpk`) | Checkpoint | Window | GPU |
|---|---|---|---|---|
| SeisT-S | 0.13 M | 0.6–0.7 MB | 3 × 8192 | not needed |
| **SeisT-M** (default) | 0.38 M | 1.6–1.8 MB | 3 × 8192 | not needed |
| SeisT-L | 0.66 M | 2.5–3.1 MB | 3 × 8192 | not needed |

Parameter counts are for the `dpk` head, which is the largest; the other
four heads are slightly smaller on the same backbone (0.31 M each at M).
These are the counts the smoke test asserts.

**GPU is genuinely optional for inference.** At 0.13–0.66 M parameters
this is three orders of magnitude smaller than anything else in this
repo; a laptop CPU runs it comfortably, and for a catalog the bottleneck
is waveform I/O, not the model — exactly as for `seisbench`.

The image is CUDA-capable anyway, for **fine-tuning**, which is the
realistic path (see "Caveats"). Upstream ships the full training engine —
`main.py` with a `--checkpoint` argument for warm starts, plus
`torchrun`-based DDP — and that is where an H100 pays for itself.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node, pointing enroot's scratch dirs off the
50 GB home:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+seist+v1.sqsh \
    'docker://ghcr.io#bradleylab/seist:v1'
```

Inference over a catalog needs no GPU — use `general-cpu` and an array,
the same shape as the `seisbench` job:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=4 \
       --mem=8G \
       --time=04:00:00 \
       --array=0-99 \
       --wrap='srun \
         --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+seist+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
         bash -lc "export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/seist_attributes.py"'
```

Fine-tuning wants the GPU partition:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+seist+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; cd /opt/seist; \
       python main.py \
         --seed 0 \
         --mode "train_test" \
         --model-name "seist_m_emg" \
         --checkpoint /opt/seist/pretrained/seist_m_emg_pnw.pth \
         --log-base /scratch2/fs1/alexander.s.bradley/seist-logs \
         --device "cuda:0" \
         --data /scratch2/fs1/alexander.s.bradley/waveforms \
         --dataset-name "pnw" \
         --in-samples 8192 \
         --batch-size 256'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
into the container, so a stray `pip install --user` on the login node
would otherwise shadow the container's site-packages. See
`~/.claude/rules/research-infrastructure.md`.

Fine-tuning on your own waveforms means writing a dataset module: add
`datasets/<yours>.py`, decorate it with `@register_dataset`, and pass
`--dataset-name`. `datasets/diting.py` and `datasets/pnw.py` are the
worked examples. That file has to live somewhere writable, so copy
`/opt/seist` into scratch first and run from there.

## Minimal inference

```python
import numpy as np
import torch
from models import create_model, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 'emg' = magnitude. Swap for seist_m_pmp (polarity), seist_m_baz
# (back-azimuth), seist_m_dis (distance), seist_m_dpk (detection+picks).
model = create_model("seist_m_emg", in_channels=3)
ckpt = load_checkpoint("/opt/seist/pretrained/seist_m_emg_diting.pth", device=device)
model.load_state_dict(ckpt["model_dict"] if "model_dict" in ckpt else ckpt)
model.to(device).eval()

# waveform: (3, 8192) float32, channel order Z, N, E.
waveform = np.zeros((3, 8192), dtype=np.float32)

# Per-channel demean then divide by per-channel std — this is upstream's
# normalize(mode="std") from demo_predict.py, and the training-time
# preprocessing. Do not substitute a different normalization.
waveform -= waveform.mean(axis=1, keepdims=True)
std = waveform.std(axis=1, keepdims=True)
std[std == 0] = 1
waveform /= std

with torch.inference_mode():
    out = model(torch.from_numpy(waveform).reshape(1, 3, -1).to(device))

print(out.shape)   # emg/baz/dis -> (1, 1); pmp -> (1, 2); dpk -> (1, 3, 8192)
```

Output shapes, verified against the shipped checkpoints:

| Head | Output | Meaning |
|---|---|---|
| `dpk` | `(N, 3, 8192)` | per-sample detection, P, S probability traces |
| `pmp` | `(N, 2)` | up/down first-motion logits |
| `emg` | `(N, 1)` | magnitude |
| `baz` | `(N, 1)` | back-azimuth |
| `dis` | `(N, 1)` | epicentral distance |

For `dpk`, converting probability traces to picks is
`training/postprocess.py:process_outputs`, which wraps ObsPy's
`trigger_onset`. Upstream's `demo_predict.py` shows the full path from an
HDF5 trace to a plotted prediction; it reads DiTing-format HDF5, so
rewrite its `load_data` for your own waveform source.

## Licensing

**MIT**, for both code and the shipped weights — the checkpoints are part
of the MIT-licensed repository and carry no separate terms. The upstream
`LICENSE` is preserved at `/opt/seist/LICENSE` inside the image.

```bibtex
@article{li2024seist,
  author  = {Li, Sen and Yang, Xu and Cao, Anye and Wang, Changbin and
             Liu, Yaoqi and Liu, Yapeng and Niu, Qiang},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  title   = {{SeisT}: A Foundational Deep-Learning Model for Earthquake
             Monitoring Tasks},
  year    = {2024},
  volume  = {62},
  doi     = {10.1109/TGRS.2024.3371503}
}
```

## Caveats

- **The shipped weights are benchmark artifacts, and upstream says so.**
  The repo README states plainly that the checkpoints exist to reproduce
  the paper's comparison, and that practical use requires retraining on
  larger, task-appropriate data. Take that seriously for `emg` and `baz`
  above all: magnitude scales are regionally calibrated and back-azimuth
  depends on local structure and instrument orientation, so a model
  trained on DiTing (China) or PNW has no claim on the New Madrid seismic
  zone or any other target. Polarity (`pmp`) is the most likely to
  transfer; magnitude is the least.

- **Windows are fixed at 8192 samples, three components.** Sampling rate
  is a property of the training set, not of the model: `datasets/diting.py`
  declares 50 Hz and `datasets/pnw.py` 100 Hz, so the same 8192-sample
  window is 163.8 s for a DiTing-trained checkpoint and 81.9 s for a PNW
  one. Resampling and windowing are the caller's job and must match the
  checkpoint's training set. There is no automatic resampling the way
  SeisBench does it.

- **The image also carries five baseline models, and that is not a
  bundling violation.** `models/` includes upstream's reimplementations
  of PhaseNet, EQTransformer, MagNet, DitingMotion, BAZ-Network, and
  distPT-Network. They are the paper's comparison baselines, ship no
  weights, and are inseparable from the pinned source tree. The container
  runs one model — SeisT — and only SeisT checkpoints are present. If you
  want PhaseNet or EQTransformer with real weights, use the `seisbench`
  image.

- **Generic top-level module names on `PYTHONPATH`.** The image puts
  `/opt/seist` on `PYTHONPATH`, which exposes `models`, `datasets`,
  `utils`, `training`, and `config` as top-level packages. That is fine
  inside this container, but it will collide with anything else of the
  same name — do not merge this environment with another Python
  installation. (Same situation as `momo`.)

- **Upstream's dependency pins do not install on Python 3.11.**
  `requirements.txt` asks for `h5py==3.1.0` (2020), which publishes no
  cp311 wheel and falls back to an sdist needing an HDF5 toolchain. The
  Dockerfile therefore date-matches the whole dependency block to the
  pinned commit (2025-06) rather than to the paper's 2023 environment;
  each deviation is justified inline. `timm`, `obspy`, and `GPUtil` keep
  upstream's exact versions.

- **`torch.load` and `weights_only`.** `models/_factory.py:load_checkpoint`
  calls `torch.load` with no `weights_only` argument, which torch 2.6
  flipped to `True` by default. Checked rather than assumed:
  `pretrained/seist_m_dpk_diting.pth` loads cleanly under the
  `weights_only=True` default on torch 2.10, because the released
  checkpoints are bare `OrderedDict` state dicts of tensors. The torch
  2.5.1 pin is for ABI coherence with the lab's other GPU images, not to
  dodge this.

- **`onnx`/`onnxruntime` are pinned upstream but never imported**, so
  they are not installed. If you want to export a SeisT model to ONNX,
  add them; the model code was written with export in mind (its
  `_auto_pad_1d` helper exists to avoid `padding='same'`, which
  `torch.onnx` does not support).
