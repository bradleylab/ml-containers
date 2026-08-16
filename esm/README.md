# esm — ESMC protein language model

Protein language model from Chan Zuckerberg Biohub (formerly
EvolutionaryScale). Produces sequence embeddings and per-residue logits from
amino-acid sequence alone — no structure required. ESMC is the current
generation of Evolutionary Scale Modeling; ESM3 open weights are also
reachable from this image.

- Upstream: https://github.com/Biohub/esm (code MIT). The old
  `github.com/evolutionaryscale/esm` URL 301-redirects here after the move to
  Biohub.
- Weights: https://huggingface.co/biohub — MIT and **ungated**. A separate
  Acceptable Use Policy applies as a behavioral condition, not a weight
  restriction.
- Preprint: "A world model of protein biology: ESMC, ESMFold2 & ESM Atlas"
  (Biohub, 2026) — see the upstream repo for the current citation.

> **Status: experimental.** Not yet benchmarked on lab data. The default
> target is batch embedding with ESMC-600M.

## Image tag

`ghcr.io/bradleylab/esm:latest` (also `:v1`, `:cu128-py312`)

> The `:torch2.7-cu129` tag from the first published build is stale — it was
> named for the NGC base this image no longer uses. Do not pull it.

## Contents

- `nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04` (**Python 3.12** as the
  system interpreter) plus torch from the PyPI cu128 wheels.
- `esm` installed from GitHub at a pinned commit SHA (there is no PyPI
  release), plus its dependency tree — including Biohub's `transformers`
  fork, separately pinned.
- `HF_HOME=/root/.cache/huggingface` — override at runtime to a mounted
  scratch dir so weights persist.
- No wrapper CLI. Upstream ships a Python SDK only; run inference from a
  script (see below).

## Checkpoints and GPU requirements

Nothing is baked into the image. Weights download from the `biohub` HF org on
first `from_pretrained` call.

| Checkpoint | HF ID | Params | Weights (F32) | GPU |
|---|---|---|---|---|
| ESMC 300M | `biohub/ESMC-300M` | 333M | 1.33 GB | trivial — any GPU, or CPU |
| **ESMC 600M** (default) | `biohub/ESMC-600M` | 575M | 2.30 GB | trivial — any GPU |
| ESMC 6B | `biohub/ESMC-6B` | 6.35B | 25.41 GB, 6 shards | 1× H100 80 GB at F32; bf16 halves it |
| ESM3 open | `biohub/esm3-sm-open-v1` | 1.4B | 5.50 GB (all components) | any recent GPU |

ESMC-600M is the workhorse for batch embedding; ESMC-6B is worth the extra
job time for high-value work and fits a single C2 H100 comfortably. The large
ESM3 checkpoints (7B, 98B) are API-only and have never been downloadable, so
they are out of scope for this image.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node (large image — point enroot's scratch dirs off
the 50 GB home):

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+esm+v1.sqsh 'docker://ghcr.io#bradleylab/esm:v1'
```

Submit a single-H100 job. Mount a scratch dir for the HF weight cache so a
multi-GB checkpoint is fetched once and reused across jobs:

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+esm+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/root/.cache/huggingface,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/esm_embed.py'
```

One line in that script is lab-standard and easy to drop:

- `PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
  into the container, so a stray `pip install --user` on the login node would
  otherwise shadow the container's site-packages.

## Minimal inference (Python API — no CLI)

Local inference goes through the Biohub `transformers` fork, not an `esm.*`
model class. Embed a sequence:

```python
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

CHECKPOINT = "biohub/ESMC-600M"       # or ESMC-300M / ESMC-6B

model = AutoModelForMaskedLM.from_pretrained(CHECKPOINT, device_map="auto").eval()
tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)

sequences = ["MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPT"]
inputs = tokenizer(sequences, return_tensors="pt", padding=True)
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    output = model(**inputs)                      # final-layer logits
```

For hidden states from every transformer layer (what you usually want for
embeddings), pass `output_hidden_states=True`:

```python
    output = model(**inputs, output_hidden_states=True)
```

The upstream example calls `huggingface_hub.login()` first. That is not
needed here — the `biohub` checkpoints are ungated, so an anonymous pull
works and the job needs no token on Compute2.

The `esm.sdk` module in this image targets the hosted Biohub Platform API
(`esmc_client(...)`, requiring an API key), not local weights. Use the
`transformers` path above for local GPU inference.

## Build notes / caveats

- **Python 3.12 only, and this is a hard pin.** esm declares
  `requires-python = ">=3.12,<3.13"`, so pip refuses to install on 3.11 or
  3.13 — the build fails outright rather than degrading. Ubuntu 24.04 ships
  3.12 as its system interpreter, which is why the base is CUDA-on-24.04.
  The `pytorch/pytorch:*-runtime` images ship Python 3.11; check
  `python --version` in any tag before switching bases, because the failure
  mode is a hard pip error at the first install layer. The Dockerfile
  asserts the version rather than trusting the tag.
- **Not an NGC base, and deliberately so.** esm requires
  `numpy>=2.0.0,<2.4` at runtime, while the NGC PyTorch images used by
  `evo2` and `ntv3` ship a global pip constraint file
  (`PIP_CONSTRAINT=/etc/pip/constraint.txt`) pinning `numpy==1.26.4`. That
  makes the esm install unresolvable on NGC — it was this image's first CI
  failure, with pip reporting `Cannot install numpy>=2.0 ... The user
  requested (constraint) numpy==1.26.4`. Do not "fix" a future recurrence by
  clearing the constraint on an NGC base; that would leave the NVIDIA-built
  stack resolving against a numpy it was not built for. Use a base without
  the constraint file, as here.
- **Two `@main` git dependencies, both pinned to SHAs.** Upstream ships no
  PyPI release, and esm's pyproject pulls `transformers` from a Biohub fork
  tracked at `@main`. Left alone, two identical builds a week apart would
  produce different images. Both are pinned via Docker ARGs:

  | ARG | Repo | Pinned SHA | Commit date |
  |---|---|---|---|
  | `ESM_REF` | `Biohub/esm` | `26b0bc2b771e3e419ea74f445a5f35cc094a1509` | 2026-07-28 |
  | `ESM_TRANSFORMERS_REF` | `Biohub/transformers` | `ef32577f55da19a4989cd7b22e004dc43a4998cb` | 2026-06-08 |

  The SHAs are recorded in the image labels (`bradleylab.build.esm_ref`,
  `bradleylab.build.transformers_ref`). esm's third git dependency,
  `nrontsis/DockQ`, is already SHA-pinned upstream.

  To move the pin, edit the ARG defaults and commit — that keeps the pin in
  git history where a rebuild can be traced. The CI workflow deliberately
  passes no `build-args`, so a manual re-run reproduces the committed pin.
  For a local experiment: `docker build --build-arg ESM_REF=<sha> esm/`.

- **The transformers fork conflicts with any pinned upstream transformers.**
  Do not add a `transformers==X.Y.Z` line to this image, and do not merge
  this environment with one that has stock transformers installed — the fork
  is what provides the ESMC model class. The Dockerfile installs esm first,
  then force-reinstalls the fork at its pinned SHA with `--no-deps`, so the
  final state is deterministic regardless of where `@main` sits on build day.

- **Smoke test is metadata-only, by design.** `import esm` reaches for
  GPU-only extensions (flash_attn) that the CPU build runner lacks, the same
  situation as evo2, so the build-time check reads versions through
  `importlib.metadata`. It does one thing beyond evo2's: it reads pip's
  `direct_url.json` for both git packages and asserts the installed commit
  IDs match the ARGs, so a silently-drifted pin fails the build rather than
  shipping. A real load test must run on an H100.

- **Upstream version string is ahead of the tags.** The pinned commit
  reports `esm 3.3.0` in package metadata while the newest GitHub tag is
  `v3.2.2.post2`. That is upstream's state, not a packaging error here.
