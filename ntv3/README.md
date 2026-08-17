# ntv3 — Nucleotide Transformer v3

Genomic language model from InstaDeep that reads up to **1 Mb of sequence at
nucleotide resolution**. Beyond sequence embeddings, the post-trained
checkpoints predict roughly 16,000 functional genomic tracks across 24
species — the signal set you would otherwise get from BigWig files — and
base-resolution annotation suitable for writing out as BED.

- Upstream: https://github.com/instadeepai/nucleotide-transformer, in
  particular `docs/nucleotide_transformer_v3.md`. Released December 2025 and
  actively maintained.
- Weights: https://huggingface.co/InstaDeepAI/NTv3_650M_post — **gated**,
  and under a **non-commercial** licence. Read the licence section before
  using this.
- HF collection: `InstaDeepAI/nucleotide-transformer-v3`.

> **Status: experimental.** Not yet benchmarked on lab data. The intended
> first target is embeddings and track prediction at 131 kb windows.

## Licence and gating — read this first

| Artifact | Terms |
|---|---|
| Weights | **InstaDeep NTv3 non-commercial licence.** No commercial use. No training a competing model on this model's outputs. |
| Upstream code | CC BY-NC-SA 4.0 |
| HF repo access | **Gated** — an HF account must accept the terms on the model page before any download works |

University research at WashU is in scope. Commercial work is not, and this
image carries a `bradleylab.model.use_restriction` label saying so, because
the restriction travels with the image and will not be obvious to whoever
pulls it next. It is the second non-commercial image in the catalog, after
`dofa-clip`.

**The HF token is never baked into the image.** It is supplied from the
environment at the moment weights are staged, and jobs then run offline
against the cache. The build-time smoke test fails the build if `HF_TOKEN`
or `HUGGING_FACE_HUB_TOKEN` is set during the build, so a token cannot be
published to GHCR by accident.

## Image tag

`ghcr.io/bradleylab/ntv3:latest` (also `:v1`, `:torch2.7-cu129`)

## Contents

- NGC PyTorch 25.04: torch 2.7.0a0, CUDA 12.9, flash-attn, Python 3.12
  (`TORCH_CUDA_ARCH_LIST` includes 9.0 → H100). Same base as `esm` and
  `evo2`.
- `transformers>=4.55,<5` — upstream's floor for the custom
  `ntv3_posttrained` architecture.
- `HF_HOME=/root/.cache/huggingface` — override at runtime to a mounted
  scratch dir holding the staged cache.
- No wrapper CLI. Run inference from a script (see below).

## Checkpoints and GPU requirements

Nothing is baked into the image.

| Checkpoint | HF ID | Params | Weights | Notes |
|---|---|---|---|---|
| **NTv3 650M post-trained** (recommended) | `InstaDeepAI/NTv3_650M_post` | 650M | 2.72 GB fp32 safetensors | embedding dim 1536; bf16 recommended on H100 |
| NTv3 100M post-trained | `InstaDeepAI/NTv3_100M_post` | 100M | — | cheaper track prediction |
| `*_131kb` variants | `InstaDeepAI/NTv3_{100M,650M}_post_131kb` | as above | — | post-trained at the 131 kb window |
| Pre-trained only | `InstaDeepAI/NTv3_{8M,100M,650M}_pre` | 8M / 100M / 650M | — | embeddings, no functional-track heads |
| Generative | `InstaDeepAI/NTv3_generative` | — | — | separate checkpoint, Jan 2026; out of scope here |

Blank cells are quantities not recorded in the lab's research pass — check
the HF model card rather than assuming.

**On VRAM: there is no official figure, and none is invented here.** What is
known is that the weights are the easy part — 2.72 GB at fp32, roughly half
that in bf16, against 80 GB on a C2 H100. The unknown is activation memory,
which is what actually scales with context length and is the reason a 1 Mb
window may or may not fit. Start at **131 kb windows**, measure with
`torch.cuda.max_memory_allocated()`, and scale up empirically from there.
Record the window size that worked and the memory it used, so the next job
does not rediscover it.

## Input length rules — the easy silent mistake

Two constraints that produce wrong output rather than an error if you get
them wrong:

1. **Input length must be a multiple of 128 bp.** Not "should be" — the
   model's tokenization depends on it.
2. **Pad with `N`, not `[PAD]`.** `N` is the ambiguous-nucleotide symbol and
   is what the model was trained to see in unresolved positions. Reaching
   for the tokenizer's `[PAD]` token, which is the reflex from every other
   HF model, feeds the model something it has no representation for.

So padding a 1,000 bp sequence means appending 24 `N` characters to the
*string*, before tokenization, to reach 1,024 — not passing `padding=True`
to the tokenizer.

Third, for the post-trained track heads: **outputs are cropped to the middle
62.5% of the input window.** The edges are context, not prediction. A
131,072 bp window therefore yields 81,920 bp of usable track (62.5% of
131,072), and tiling a chromosome means stepping by that 81,920 bp with the
remaining 49,152 bp consumed as flanking context — not stepping by the full
window, which would leave gaps.

## Staging the gated weights (login node, token from the environment)

Compute jobs should not be the thing that authenticates. Stage once on a
login node, which has network, then run jobs offline.

Supply the token at the moment of use and never write it to a file in the
repo or into a job script:

```bash
read -rs -p 'HF token: ' HF_TOKEN && export HF_TOKEN
```

or, from a machine that has Doppler configured, `doppler run -- ...` so the
value never enters the shell history. Then:

```bash
HF_HOME=/scratch2/fs1/alexander.s.bradley/hf-cache \
  huggingface-cli download InstaDeepAI/NTv3_650M_post
```

Finally `unset HF_TOKEN`. The account must have accepted the model's terms
on its HF page first, or the download returns 403 rather than prompting.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node (large image — point enroot's scratch dirs off
the 50 GB home):

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+ntv3+v1.sqsh 'docker://ghcr.io#bradleylab/ntv3:v1'
```

Submit a single-H100 job against the staged cache. `HF_HUB_OFFLINE=1` is
deliberate: with a cold cache the job fails immediately and loudly instead
of silently asking for a token it does not have.

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+ntv3+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/root/.cache/huggingface,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; python /scratch2/fs1/alexander.s.bradley/scripts/ntv3_embed.py'
```

One line there is lab-standard and easy to drop:

- `PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
  into the container, so a stray `pip install --user` on the login node
  would otherwise shadow the container's site-packages.

Do **not** pass the HF token into the job with `--container-env`. Stage the
weights first; the job needs no credential.

## Minimal inference (Python API — no CLI)

`trust_remote_code=True` is mandatory — `ntv3_posttrained` is not a
`transformers` architecture, it is Python shipped inside the checkpoint
repo.

```python
import torch
from transformers import AutoModel, AutoTokenizer

CHECKPOINT = "InstaDeepAI/NTv3_650M_post"
WINDOW = 131_072          # multiple of 128; start here, scale empirically

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModel.from_pretrained(
    CHECKPOINT, trust_remote_code=True, torch_dtype=torch.bfloat16
).eval().to("cuda")

def to_window(seq: str, window: int = WINDOW) -> str:
    """Right-pad with N to `window`. N, not [PAD] — see README."""
    if len(seq) > window:
        raise ValueError(f"sequence {len(seq)} bp exceeds window {window} bp")
    return seq + "N" * (window - len(seq))

inputs = tokenizer(to_window(sequence), return_tensors="pt")
inputs = {k: v.to(model.device) for k, v in inputs.items()}

with torch.inference_mode():
    out = model(**inputs, output_hidden_states=True)

embeddings = out.hidden_states[-1]      # (1, tokens, 1536)
```

For the ~16,000 functional tracks and the base-resolution annotation, use
the pipeline modules the checkpoint ships (`ntv3_tracks_pipeline.py`,
`ntv3_interpret_pipeline.py`) rather than calling the model directly —
consult the model card for their entry points, and remember the middle-62.5%
crop when you map predictions back to genomic coordinates.

Writing predictions out as BigWig or BED is left to the caller; this image
ships no track-writer library (no `pyBigWig`). Add one here if it becomes
routine, rather than pip-installing inside a job.

## Build notes / caveats

- **This model cannot share a container with DNABERT-S.** NTv3 needs
  `transformers>=4.55`; `dnabert-s` is held at `transformers==4.27` by its
  own upstream. No version satisfies both, so the two DNA models are two
  images by necessity, not only by the one-model-per-container convention.
  A "consolidate the DNA models" change fails at pip resolution, and forcing
  it past that point breaks one of the two silently.

- **HF PyTorch route, not the GitHub repo.** `instadeepai/nucleotide-transformer`
  is a JAX codebase. This image does not build the JAX path — it uses the
  PyTorch checkpoints on the Hub through `transformers` +
  `trust_remote_code`. Anyone reading the GitHub README will find JAX
  install instructions that do not apply here.

- **The transformers pin is a range, not a SHA.** `>=4.55,<5` reflects
  upstream's stated floor with a ceiling against a surprise 5.x. That means
  two builds months apart can differ. If a result needs to be reproducible
  to the byte, pin the exact version in the Dockerfile and commit it, so the
  pin lives in git history — the same discipline `esm` applies to its two
  git dependencies.

- **Remote code may want packages this image does not ship.** The model's
  Python is fetched at `from_pretrained` time, so its import list is not
  visible at build time and is not pinned here. Stage the checkpoint and run
  one load on a login node before submitting a batch; if an import fails,
  add the package to this Dockerfile and rebuild rather than pip-installing
  inside a job, which would diverge from the committed recipe.

- **Smoke test is a real import, offline.** transformers and torch both
  import on the CPU build runner, so the build resolves `AutoModel` /
  `AutoTokenizer` / `AutoConfig` for real — more than `esm` and `evo2`
  manage. It cannot reach the model class, which lives in gated remote code.
  It also asserts no HF token is present in the build environment.

- **Large image.** The NGC PyTorch 25.04 base is ~9 GB; the build workflow
  frees runner disk first. If CI runs out of space, that is the first thing
  to check.
