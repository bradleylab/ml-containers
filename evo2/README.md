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
     bash -lc 'export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/evo2_score.py'
```

## Minimal inference (Python API — no CLI)

Score a sequence (log-likelihoods / embeddings):

```python
import torch
from evo2 import Evo2

model = Evo2('evo2_7b')                       # downloads weights to HF_HOME on first call
input_ids = torch.tensor([model.tokenizer.tokenize('ACGT')]).to('cuda:0')
logits, _ = model(input_ids)
```

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
