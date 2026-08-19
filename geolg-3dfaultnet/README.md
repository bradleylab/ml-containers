# geolg-3dfaultnet — 3D seismic fault segmentation

Voxelwise fault / non-fault segmentation of a 3D seismic amplitude volume.
The network is a 3D U-Net (32/64/128/256 channels) with the two blocks the
method is named for: a local-global feature-enhancement block that sums a
plain and a dilated 3D convolution branch, and an attention-guided continuity
block applied after a windowed 3D self-attention layer. 5,956,226 parameters
(23.8 MB fp32).

- Upstream: https://github.com/letsfly27/GeoLG-3DFaultNet — MIT, PyTorch.
- Weights: GitHub Release tag `v1.0`, asset `best_model.pth`.
- Manuscript: "GeoLG-3DFaultNet: The 3D Fault Segmentation Method Based on
  Local-Global Feature Enhancement and Geometric Constraints". Publication
  status unconfirmed — see below.

This image closes the lab's long-deferred fault-detection item. FaultSeg3D was
deferred because using it meant rebuilding a TensorFlow 1.x environment;
GeoLG-3DFaultNet is PyTorch 2.8, MIT-licensed, and ships a checkpoint, so that
blocker is gone. What replaced it is a provenance problem, not a packaging one.

## Provenance — read this before trusting any output

**Nothing this container produces is a validated fault interpretation.** The
image exists because the licence and the weights are clean, not because the
method has been checked.

What is verified (GitHub API, 2026-08-18):

- The repository is MIT-licensed, with a `LICENSE` file, last pushed
  2026-03-30.
- The release asset exists and its digest is recorded (see *Weights* below).
- The code runs, and the architecture matches the checkpoint's shape.

What is not:

- **Zero stars, one contributor, no forks, no issues, created and last touched
  on the same day.** There is no user community that would have found an
  obvious error.
- **The manuscript could not be located.** The repository describes itself as
  the official implementation of a paper, but the lab could not confirm that
  the paper is published, or peer-reviewed, or that the reported metrics were
  ever scrutinised. Treat every performance claim as unreviewed.
- **No independent benchmark.** The only public data in the repository is one
  128³ synthetic patch with its ground truth, which is also (as far as anyone
  outside can tell) drawn from the same distribution the model was trained on.
  Scoring well on it demonstrates that the weights loaded, not that the model
  generalises.

The obvious comparison is `xinwucwp/faultSegPlus` — the direct FaultSeg3D
successor from Xinming Wu's own group, published in *Geophysics* 89(5), 2024.
Its provenance is far better than this one's. It is unusable here for a
different reason: **the repository has no `LICENSE` file at all** (confirmed by
API, 2026-08-18), so there is no grant of rights and lab rules rule it out. The
honest summary is that the two available options fail on opposite axes, and
this one fails on the axis that can be repaired by doing work.

**The required work, before any GeoLG-3DFaultNet result is used in an
analysis, a figure, or a manuscript:** benchmark it against either
faultSegPlus predictions or a published fault interpretation of a public
survey (F3 Block or Kerry-3D), on data that is not the shipped sample patch,
and record the comparison. Until that exists, results from this image are
exploratory only.

One code-level hazard reinforces this. Upstream's `inference.py` loads the
checkpoint with `load_state_dict(..., strict=False)` after rewriting two key
prefixes (`lgfe.` → `lg.`, `fcom.` → `aco.`). Under `strict=False`, a key that
fails to match is skipped in silence: the affected layers keep their random
initialisation and the model still emits a plausible-looking fault volume. Any
run whose output will be used for anything must assert that the load consumed
every key — `SMOKE.md` shows how, and the build-time smoke test pins the
parameter and key counts so upstream drift fails the build instead.

## Image tag

`ghcr.io/bradleylab/geolg-3dfaultnet:v1` (also `:latest`, `:torch2.8-cu128`).

## Contents

- `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04`, Python 3.12.
- `torch==2.8.0+cu128` from the PyPI cu128 index — upstream's exact tested
  build.
- `numpy==1.26.4`, `scikit-image==0.24.0`, `matplotlib==3.9.4`,
  `tqdm==4.67.1` — upstream's pins, but only the packages the repository
  actually imports (see *Build notes*).
- Upstream checked out at commit `fe9bdb001ab9086c23ac77491ee2c03e35a38267`
  under `/opt/GeoLG-3DFaultNet` (`$GEOLG_HOME`), including the two 8 MB sample
  volumes. The SHA is also written to `$GEOLG_HOME/.upstream-sha` and carried
  in the `bradleylab.build.geolg_ref` image label.
- `fetch_weights.sh` on `PATH`.
- `PYTHONPATH=/opt/GeoLG-3DFaultNet`, so upstream's bare-name imports
  (`from model import GeoLG3DFaultNet`) work from any working directory.
- No wrapper CLI. Upstream ships scripts with hardcoded relative paths; run
  them from a writable working directory, or import the model yourself.

## Weights

Not baked. One checkpoint exists:

| Field | Value |
|---|---|
| Release tag | `v1.0` ("Pre-trained Model Weights", published 2026-03-30) |
| Asset | `best_model.pth` |
| Size | 71,617,015 bytes |
| sha256 | `67cb47ab5ebfd7da50eb874011c36ba1c35473c4a94d4fd8da9a0a560c76229a` |

The size and digest were read from the GitHub Releases API on 2026-08-18 and
are asserted by `fetch_weights.sh`, which **fails closed** on a mismatch.
GitHub allows an asset to be deleted and re-uploaded under the same tag, so
the tag alone is not an identity; the digest is. If verification ever fails,
the asset changed — re-verify upstream and update the constants in the script
deliberately, in a commit, rather than relaxing the check.

The 71.6 MB is not all model. It is a training checkpoint: `model_state_dict`
(23.8 MB) plus the Adam optimizer state (two moment tensors, 47.7 MB) plus the
`GradScaler` state, the epoch, and the best validation Dice. Only the first
key is used at inference.

```bash
# Online: download, verify, link into the working directory.
fetch_weights.sh

# Pre-staged: same command, no network — verifies what is already there.
fetch_weights.sh /staged/geolg-weights
```

To pre-stage for Compute2, run once on a login node:

```bash
mkdir -p /storage3/fs1/alexander.s.bradley/Active/model_weights/geolg-3dfaultnet
curl -L \
  -o /storage3/fs1/alexander.s.bradley/Active/model_weights/geolg-3dfaultnet/best_model.pth \
  https://github.com/letsfly27/GeoLG-3DFaultNet/releases/download/v1.0/best_model.pth
```

then bind-mount that directory at `/opt/geolg-weights` in the job and let
`fetch_weights.sh` verify it.

## Input format

**Raw binary, not SEG-Y.** Upstream reads volumes with
`np.fromfile(path, dtype=np.float32).reshape((128, 128, 128))`:

- little-endian float32, C-order, no header, no trailer;
- exactly 128 × 128 × 128 samples — 8,388,608 bytes per file;
- extension `.dat` (the file listing filters on it); `.npy` is mentioned in
  upstream's training instructions but no code path reads it;
- ground-truth labels, when present, are the same format and shape, with
  values 0 and 1 in a float32 array, in a parallel directory.

Amplitudes need no pre-scaling — `infer()` z-scores each volume by its own
mean and standard deviation before tiling.

Nothing in this image reads SEG-Y. A SEG-Y reader is deliberately not
installed: it would be a second tool in a one-model container, and the
conversion belongs in the analysis repo where the survey geometry, byte
locations and inline/crossline ordering are already known. Convert to raw
float32 there, and record the axis order you chose — the model was trained on
synthetic volumes with no geographic convention attached, so an axis
permutation between training and inference is a silent error, not a crash.

## GPU and memory

**This model is tile-size bound, not parameter bound.** At 5.96M parameters the
weights occupy 23.8 MB; the working set is one to two orders of magnitude
larger and grows steeply with the edge length of the tile:

| Tile edge | Voxels vs 64³ | Peak resident memory |
|---|---|---|
| 64³ | 1× | 3.3 GB |
| 96³ (upstream default `chunk_size`) | 3.4× | 5.6 GB |
| 128³ (whole sample volume in one pass) | 8× | 10.8 GB |

Measured as peak RSS of a single `torch.no_grad()` forward pass, fp32, batch
1, on CPU (torch 2.10, Apple Silicon). CPU RSS is a proxy, not a GPU
measurement — the CUDA caching allocator will report different numbers — but
the scaling is a property of the graph, not the device, and the ceiling it
implies is the one that matters when choosing a tile size. Record the real
`torch.cuda.max_memory_allocated()` on the first H100 run (`SMOKE.md` prints
it) and correct this table.

Two things drive the growth. The U-Net keeps full-resolution 32-channel
activations at the top of both the encoder and the decoder, so that cost is
linear in tile volume. On top of it, `WindowAttention3D` calls
`nn.MultiheadAttention` without `need_weights=False`, so the attention weight
matrix for each 512-window chunk is materialised and then discarded — a large
transient that scales with the number of windows, i.e. again with tile volume.

Practical consequences:

- A single H100 (80 GB) is far more than this needs. Any 16 GB GPU runs the
  96³ default comfortably; the model will also run on CPU, slowly.
- Throughput, not capacity, is the reason to use Compute2 — a production
  survey is thousands of tiles.
- Test-time augmentation costs 4× the forward passes: `predict_with_tta` runs
  the tile unflipped and flipped along each of the three axes, then averages.
  With `chunk_size=96` and a stride of 48, one 128³ volume is 27 tile
  positions × 4 passes = **108 forward passes**. For reference, one 96³
  forward took 15.9 s on the CPU used for the table above; the H100 figure is
  the one to record.

## Minimal inference

The shipped script, on the shipped sample:

```bash
docker run --rm --gpus all \
  -v "$PWD/geolg-weights:/opt/geolg-weights" \
  -v "$PWD/work:/workspace" \
  ghcr.io/bradleylab/geolg-3dfaultnet:v1 \
  bash -lc 'cp -r "$GEOLG_HOME/data" . && fetch_weights.sh && python "$GEOLG_HOME/inference.py"'
```

It writes `predict_images/2_result.png` — a 3×3 panel of orthogonal slices
through the seismic volume, the ground truth and the prediction. That is a
visual check, not a metric; `SMOKE.md` scores it.

On your own volume, skip upstream's script — it hardcodes 128³ and its own
directory layout — and call the two functions directly:

```python
import numpy as np
import torch
from model import GeoLG3DFaultNet          # PYTHONPATH is set in the image

SHAPE = (128, 128, 128)                    # must match the file's true shape
CHUNK = 96                                 # tile edge; see the memory table

device = torch.device("cuda")
model = GeoLG3DFaultNet(in_channels=1, num_classes=2).to(device).eval()

ckpt = torch.load("best_model.pth", map_location=device)
state = ckpt["model_state_dict"]
state = {k.replace("lgfe.", "lg.").replace("fcom.", "aco."): v for k, v in state.items()}

# Upstream passes strict=False here. Keep it, but check the result: an
# unmatched key means those layers are still randomly initialised.
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing and not unexpected, (missing, unexpected)

volume = np.fromfile("my_volume.dat", dtype=np.float32).reshape(SHAPE)

import inference                            # brings infer() and post_process()
fault = inference.infer(volume, shape=SHAPE, chunk_size=CHUNK)
fault = inference.post_process(fault, min_size=50)
fault.astype(np.float32).tofile("my_volume_fault.dat")
```

`infer()` returns a binary volume, already thresholded at probability 0.5 and
averaged over the overlapping tiles; `post_process()` then drops connected
components smaller than `min_size` voxels. To keep the continuous probability
field instead — usually what you want for a figure or for comparing against
another method — copy `infer()` and return `pred_label` before the `> 0.5`
line.

Importing `inference` executes its module-level code, which builds the model
and loads `./best_model.pth` a second time. It is a 24 MB load, so this is
wasteful rather than harmful, but it means the working directory must contain
the checkpoint symlink before the import.

## Run on Compute2 (Pyxis/enroot)

Import once on a login node:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+geolg-3dfaultnet+v1.sqsh \
    'docker://ghcr.io#bradleylab/geolg-3dfaultnet:v1'
```

Then submit. One GPU is enough; scale by tiles, not by device:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -J geolg-fault

SQSH=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+geolg-3dfaultnet+v1.sqsh
WEIGHTS=/storage3/fs1/alexander.s.bradley/Active/model_weights/geolg-3dfaultnet
WORK=/scratch2/fs1/alexander.s.bradley/geolg

srun --container-image="$SQSH" \
     --container-mounts="$WEIGHTS":/opt/geolg-weights,"$WORK":/workspace \
     --container-workdir=/workspace \
     bash -lc 'export PYTHONNOUSERSITE=1; \
               fetch_weights.sh && python /workspace/scripts/run_fault_seg.py'
```

Two lines there are lab-standard and easy to drop:

- `PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
  into the container, so a stray user-site install on the login node would
  otherwise shadow the container's site-packages. See
  `~/.claude/rules/research-infrastructure.md`.
- `--container-workdir` must point at a **writable** mount. The container
  filesystem is read-only under enroot, and every upstream script resolves its
  inputs and outputs relative to the working directory.

Compute nodes may have no outbound network. Pre-stage the checkpoint on a
login node as shown under *Weights*; `fetch_weights.sh` then verifies and links
it without touching the network.

## Build notes and caveats

- **`torch.load` defaults changed in torch 2.6.** Since that release,
  `torch.load` defaults to `weights_only=True`, which restricts unpickling to
  tensors and plain built-ins. Upstream's call —
  `torch.load(checkpoint_path, map_location=device)` — does not pass the
  argument, so it gets the new default. This checkpoint is expected to load
  cleanly under it: everything `train.py` saves is a tensor, a dict, or a
  Python scalar. If a future checkpoint does not, the supported escape hatch
  is the environment variable, not a patched upstream:

  ```bash
  TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 python "$GEOLG_HOME/inference.py"
  ```

  It is deliberately not set in the image. It disables a security control
  globally for the process, and it should only ever be turned on for a file
  whose digest `fetch_weights.sh` has just verified. In new code, prefer
  passing `weights_only=False` at the one call site that needs it — an
  explicit argument overrides the environment variable, so the variable only
  affects calls that stayed silent.

- **Upstream's `requirements.txt` is a frozen environment, not a dependency
  list.** It is 80 lines and includes `nnunetv2`, `batchgenerators`,
  `pydicom`, `SimpleITK`, `vtk`, `pyvista` and `cigvis` — none of which any of
  the five source files import. The image installs the real import closure
  (`torch`, `numpy`, `scikit-image`, `matplotlib`, `tqdm`) at upstream's
  pinned versions. This keeps the image small and the pin surface honest; the
  cost is that if upstream later adds an import, the build fails on a missing
  module rather than silently succeeding. Add it here with a reason when that
  happens — do not restore the frozen file wholesale.

- **Python 3.12, not upstream's 3.9.21.** Upstream's tested interpreter is
  3.9; the CUDA-on-Ubuntu-24.04 base ships 3.12. Every source file is plain
  torch/numpy with no version-sensitive syntax, and the build-time smoke test
  runs a real forward pass rather than assuming it. Nothing here pins torch to
  a Python version, so a 3.9 base could be built if a discrepancy ever
  appears.

- **Base image chosen so it does not dictate torch.** Upstream tested against
  torch 2.8.0+cu128 exactly. NGC's `pytorch:25.04-py3` bakes torch 2.7.0a0 —
  an NVIDIA alpha build, not upstream's version — and exports
  `PIP_CONSTRAINT=/etc/pip/constraint.txt` pinning `numpy==1.26.4`, which has
  already broken a build in this repo (see `esm/README.md`). The
  `pytorch/pytorch:*` tags fix torch by tag and ship Python 3.11. A plain CUDA
  runtime base leaves the torch version under this Dockerfile's control, which
  is where it belongs when the pin comes from upstream.

- **The build-time smoke test asserts structure, not just imports.** It pins
  the parameter count (5,956,226) and the state-dict key count (136) and runs
  a 32³ forward pass. Both assertions exist because of the `strict=False` load
  described under *Provenance*: they convert an upstream architecture change —
  which would otherwise load silently with random layers — into a build
  failure. It fetches no weights and touches no network.

- **128³ is baked into upstream's scripts.** `inference.py` hardcodes the
  reshape and defaults `infer(shape=(128, 128, 128))`. Larger surveys must be
  tiled by the caller, with an overlap and a merge rule chosen deliberately;
  upstream's internal tiling (stride = `chunk_size // 2`, averaged over
  overlaps) is a reasonable model to copy but only operates within one 128³
  volume.

- **No training path is supported here.** `train.py` ships in the image
  because it came with the checkout, but retraining needs a dataset this lab
  does not have and a `DataLoader` that preloads every volume into RAM. Treat
  the image as inference-only.

## Licensing

Upstream code is MIT (`LICENSE` present in the repository), and the release
weights are published by the same author under that repository. This image
adds only the Dockerfile, the fetch script and this documentation, under the
`ml-containers` repository's own terms. The MIT grant covers use, modification
and redistribution with attribution — the practical constraint on this model
is scientific, not legal, and is the subject of the *Provenance* section above.
