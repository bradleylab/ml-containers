# Evo2 — genomic foundation model

DNA language model (StripedHyena 2) from the Arc Institute for
variant-effect prediction, sequence scoring (log-likelihoods), and genome
generation across all domains of life.

- Upstream: https://github.com/ArcInstitute/evo2 (code Apache-2.0)
- Weights: https://huggingface.co/arcinstitute — `evo2_7b`, `evo2_40b`,
  `evo2_7b_base`, `evo2_1b_base`, `evo2_7b_262k`, … (weight license per the
  HF model cards — verify before redistribution)
- Paper: "Genome modeling and design across all domains of life with Evo 2"
  (Arc Institute, 2025) — see the upstream repo for the current citation.

> **Status: experimental / backfilled.** This recipe reproduces a
> previously ad-hoc `ghcr.io/bradleylab/evo2` image (it had no committed
> recipe), reconstructed from the published image's build history: the NGC
> PyTorch 25.04 base + `pip install evo2 biopython`. The 7B path is the
> intended first target; it has not yet been benchmarked on lab data.

## Image tag

`ghcr.io/bradleylab/evo2:latest` (also `:v1`, `:torch2.7-cu129`)

## Contents

- NGC PyTorch 25.04: torch 2.7.0a0, CUDA 12.9, Transformer Engine 2.2,
  flash-attn, Python 3.12 (`TORCH_CUDA_ARCH_LIST` includes 9.0 → H100).
- `evo2` + `biopython` pip packages.
- `HF_HOME=/root/.cache/huggingface` — override at runtime to a mounted
  scratch dir so weights persist.

## GPU requirements

| Model | Precision | GPU |
|-------|-----------|-----|
| `evo2_7b`, `evo2_1b_base`, `evo2_7b_base` | bf16 | 1× any recent GPU (fits a single C2 H100 80 GB) |
| `evo2_20b`, `evo2_40b` | FP8 (Transformer Engine) | Hopper (H100); **40B needs multiple H100s** |

C2 `general-gpu` allocates a single H100, so the **7B / 1B / base** models
are the practical targets here. The 40B model needs multi-GPU and is out of
scope for a single-GPU job.

**Confirmed 2026-08-06/07:** `evo2_20b` loads and runs (score + generate) on
a single C2 H100 with this image's TE 2.2 — no version bump needed. Weight
fetch + load + inference took ~3.5 min end to end from a cold HF cache.
`evo2_40b` and `evo2_1b_base` remain untested here.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node (large image — point enroot's scratch dirs off
the 50 GB home):

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+evo2+v1.sqsh 'docker://ghcr.io#bradleylab/evo2:v1'
```

Submit a single-H100 job. Mount a scratch dir for the HF weight cache so
the multi-GB checkpoint is fetched once and reused:

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+evo2+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/root/.cache/huggingface,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     --container-workdir=/scratch2/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/evo2_score.py'
```

`--container-workdir` is required, not optional — see the read-only-filesystem
caveat below.

## Minimal inference (Python API — no CLI)

Score a sequence (log-likelihoods / embeddings):

```python
import torch
from evo2 import Evo2

model = Evo2('evo2_7b')                       # downloads weights to HF_HOME on first call
input_ids = torch.tensor(
    model.tokenizer.tokenize('ACGT'), dtype=torch.int
).unsqueeze(0).to('cuda:0')
outputs, _ = model(input_ids)
logits = outputs[0]
```

`dtype=torch.int` is required — without it `torch.tensor(...)` infers a
narrower integer type that the embedding lookup rejects at call time.

Generate:

```python
from evo2 import Evo2
model = Evo2('evo2_7b')
out = model.generate(prompt_seqs=["ACGT"], n_tokens=400, temperature=1.0, top_k=4)
print(out.sequences[0])
```

There is no dedicated CLI — wrap the Python API in a script under scratch
(as referenced in the SLURM example above).

## Build notes / caveats

- **Large image** (~12 GB). The build workflow frees runner disk before
  building; if CI runs out of space or the flash-attn/evo2 build is slow,
  that's the first thing to check.
- **Smoke test is metadata-only.** `import evo2` needs a GPU at import time
  (Transformer Engine), so the build-time check reads versions via
  `importlib.metadata` rather than importing — a real load test must run on
  an H100.
- **Transformer Engine version.** This image uses NGC 25.04's TE 2.2. The
  upstream evo2 README suggests TE 2.3.0 for the FP8 (20B/40B) path; if the
  large models misbehave under FP8, a TE bump is the likely fix. The 7B
  bf16 path does not use TE.
- **`--container-workdir` is mandatory, not a convenience flag.** Importing
  `evo2` pulls in `vortex.logging`, which unconditionally opens
  `activations_debug.log` (a relative path) at *import time* — before your
  script runs any code. Enroot's container root is a read-only squashfs, so
  if the process's working directory is the image's baked-in `/workspace`,
  that open() fails with `OSError: Read-only file system`. There is no env
  var or config flag upstream to disable or redirect this file (checked
  `vortex/logging.py` directly) — the only fix is to start the container
  with its working directory pointed at one of the `--container-mounts`
  paths, so the relative path resolves somewhere writable.
- **Tokenizer output needs an explicit `dtype=torch.int` cast.** Evo2's
  `CharLevelTokenizer` returns a plain list of small integers; letting
  `torch.tensor(...)` infer the dtype from that list can produce a `uint8`
  tensor, which PyTorch's embedding lookup rejects (`Expected ... Long, Int;
  but got torch.cuda.ByteTensor`). Always cast explicitly, matching the
  upstream README's own examples.
