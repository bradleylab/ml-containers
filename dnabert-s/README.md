# dnabert-s — species-aware DNA language model

DNA language model from MAGICS-LAB, built for metagenomic binning. It
embeds a nucleotide sequence into a 768-dimensional vector in which contigs
from the same species sit close together, so the embeddings can be clustered
into bins without alignment or reference genomes.

Architecture is DNABERT-2's — a MosaicBERT-style encoder with ALiBi position
biases and a 4096-token BPE vocabulary, ~117M parameters — contrastively
fine-tuned for species separation.

- Upstream: https://github.com/MAGICS-LAB/DNABERT_S. Released 2024, last
  pushed 2026-01-01; effectively frozen but functional.
- Weights: https://huggingface.co/zhihan1996/DNABERT-S — **Apache-2.0 and
  ungated**, 468 MB `pytorch_model.bin` (fp32). No token needed.
- Paper: "DNABERT-S: Pioneering Species Differentiation with Species-Aware
  DNA Embeddings" — see the upstream repo for the current citation.

> **Status: experimental.** Not yet benchmarked on lab data. The intended
> first target is batch embedding of assembled contigs.

## Licence — read before redistributing anything

The two halves carry different terms, and only one of them is permissive:

| Artifact | Licence | Consequence |
|---|---|---|
| Weights + the modelling code inside `zhihan1996/DNABERT-S` | Apache-2.0 (HF model card) | Free to use, ungated, redistributable |
| The `MAGICS-LAB/DNABERT_S` GitHub repo | **No LICENSE file** — formally all rights reserved | Fine to read and run; do **not** redistribute its code |

This image sidesteps the second row entirely: nothing from the GitHub repo
is vendored. The custom modelling files the checkpoint needs
(`bert_layers.py`, `flash_attn_triton.py`, …) live in the Apache-2.0 HF
repo and are fetched at load time by `trust_remote_code=True`. So the image
is Apache-2.0 clean, and the missing GitHub licence only matters if someone
later copies training or evaluation scripts out of that repo into a lab
artifact.

## Image tag

`ghcr.io/bradleylab/dnabert-s:latest` (also `:v1`, `:torch2.5-cu121`)

## Contents

- `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04` + Python 3.11 +
  PyTorch 2.5.1 cu121 (`TORCH_CUDA_ARCH_LIST` includes 9.0 → H100).
- `transformers==4.27.*` — upstream's own pin — with `huggingface_hub<0.26`,
  `einops`, `numpy<2` alongside it.
- **No triton.** Deliberately uninstalled; see the caveats below.
- `HF_HOME=/root/.cache/huggingface` — override at runtime to a mounted
  scratch dir so weights persist.
- No wrapper CLI. Run inference from a script (see below).

## Checkpoint and GPU requirements

Nothing is baked into the image. Weights download on the first
`from_pretrained` call.

| Checkpoint | HF ID | Params | Weights | GPU |
|---|---|---|---|---|
| DNABERT-S (only one) | `zhihan1996/DNABERT-S` | ~117M | 468 MB `pytorch_model.bin`, fp32 | trivial — any GPU, and CPU inference is feasible |

Upstream publishes a single checkpoint; there is no size ladder to choose
from. The model is small enough that a GPU buys throughput, not capability —
the reason to run this on Compute2 is embedding a large contig set in one
pass, not fitting the model.

## Context length and long contigs

`max_position_embeddings` is 512, but the encoder uses ALiBi rather than
learned positional embeddings, so it extrapolates past that. Upstream
trained on sequences up to roughly 10 kb.

There is **no official recipe for contigs longer than that.** Chunk and
pool: split the contig into windows, embed each, then average the window
embeddings (weighted by window length if the last window is short). Record
whatever window and overlap you choose in `METHODS.md` — it is an analysis
decision, not a detail, because it changes the embedding.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+dnabert-s+v1.sqsh 'docker://ghcr.io#bradleylab/dnabert-s:v1'
```

Submit a single-H100 job. Mount a scratch dir for the HF cache so the
checkpoint (and the remote modelling code that comes with it) is fetched
once and reused:

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+dnabert-s+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley/hf-cache:/root/.cache/huggingface,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/dnabert_s_embed.py'
```

One line there is lab-standard and easy to drop:

- `PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME`
  into the container, so a stray `pip install --user` on the login node
  would otherwise shadow the container's site-packages.

Compute nodes reach the network on this cluster, but the first job after an
image refresh still pays the download. Warming the cache from a login node
(`python -c "from transformers import AutoModel; AutoModel.from_pretrained('zhihan1996/DNABERT-S', trust_remote_code=True)"`)
keeps that cost off the GPU allocation.

## Minimal inference (Python API — no CLI)

`trust_remote_code=True` is mandatory: the checkpoint's architecture is not
in `transformers`, it is shipped as Python inside the HF repo.

The upstream single-sequence form:

```python
import torch
from transformers import AutoModel, AutoTokenizer

CHECKPOINT = "zhihan1996/DNABERT-S"

tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT, trust_remote_code=True)
model = AutoModel.from_pretrained(CHECKPOINT, trust_remote_code=True).eval().to("cuda")

dna = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"
input_ids = tokenizer(dna, return_tensors="pt")["input_ids"].to(model.device)

with torch.inference_mode():
    hidden = model(input_ids)[0]          # (1, tokens, 768)

embedding = hidden[0].mean(dim=0)         # 768-d, mean pooling
```

Batched, which is what you actually want for a contig set — note that mean
pooling must respect the padding mask or short sequences get diluted by
`[PAD]` tokens:

```python
batch = tokenizer(sequences, return_tensors="pt", padding=True)
batch = {k: v.to(model.device) for k, v in batch.items()}

with torch.inference_mode():
    hidden = model(**batch)[0]                       # (B, tokens, 768)

mask = batch["attention_mask"].unsqueeze(-1)         # (B, tokens, 1)
embeddings = (hidden * mask).sum(1) / mask.sum(1)    # (B, 768)
```

**Mean pooling is the documented method** for this model — not the `[CLS]`
token, not max pooling. Changing it changes what the embeddings mean and
invalidates comparison against upstream's reported results.

`device_map="auto"` is not available here: `accelerate` is not installed,
because the versions compatible with transformers 4.27 pull the environment
in directions that conflict with the pins above. Use `.to("cuda")`, which
is all a 117M-parameter model needs.

## Build notes / caveats

- **This model cannot share a container with NTv3.** DNABERT-S requires
  `transformers==4.27`; `ntv3` requires `transformers>=4.55`. There is no
  version that satisfies both, so `dnabert-s` and `ntv3` are two images by
  necessity, not only by the one-model-per-container convention. Do not
  "consolidate the DNA models" — the merge fails at pip resolution, and
  forcing it past that point breaks one model silently.

- **Triton is removed, on upstream's advice.** The DNABERT_S repo tells you
  to `pip uninstall triton` on GPUs other than A100: its triton
  flash-attention kernel misbehaves there, and without triton importable the
  model falls back to standard attention. We target H100, so the Dockerfile
  removes it at build time and the build-time smoke test asserts it is
  absent.

  Two consequences worth knowing. First, the uninstall must be the last pip
  step in the Dockerfile — torch declares triton as a Linux dependency, so
  any pip install after it can quietly restore it; if you add a package,
  add it *above* the uninstall line. Second, `torch.compile` and
  TorchInductor do not work in this image. Nothing in the inference path
  uses them, and standard attention on an H100 is not the bottleneck for a
  117M-parameter encoder.

  Not verified on our own H100 yet — this follows upstream's instruction
  rather than a measurement here. If someone does benchmark both paths on
  an H100, record it and update this section.

- **transformers 4.27 dictates the whole stack.** It is a 2023 release and
  it constrains everything around it:

  | Pin | Reason |
  |---|---|
  | Python 3.11, not 3.12 | transformers 4.27 requires `tokenizers<0.14`, which ships no cp312 wheel; 3.11 also still has `distutils` |
  | torch 2.5.1, not 2.6+ | torch 2.6 flipped `torch.load()` to `weights_only=True`; transformers 4.27 calls it without that argument, and this checkpoint is a `.bin`, not safetensors |
  | `huggingface_hub<0.26` | 0.26 removed `cached_download`, which transformers 4.27 imports at package-import time — a floating hub version breaks `import transformers` before any model loads |
  | `numpy<2` | everything above predates the numpy 2.0 API break |

  The `huggingface_hub` bound is the one that bites without warning: it is
  our pin, not upstream's, and upstream's requirements file will not tell
  you about it.

- **Not on the NGC base the other sequence LMs use.** `esm` and `evo2` both
  build on `nvcr.io/nvidia/pytorch:25.04-py3`, which is Python 3.12 — ruled
  out by the tokenizers constraint above. This image uses the CUDA 12.1 +
  Python 3.11 + torch 2.5.1 stack shared with `prithvi-eo`, `satlas`,
  `clay`, and `terramind` instead.

- **A CPU base would also have worked.** At 468 MB of weights this model
  runs on CPU, and for a handful of sequences that is the simpler route —
  run it in a local venv. The image is GPU-based because the job it exists
  for is embedding a whole contig set in one pass.

- **Smoke test is a real import, not just metadata.** Nothing in this stack
  needs a device at import time, so unlike `esm` and `evo2` the build
  resolves `AutoModel` / `AutoTokenizer` / `AutoConfig` for real on the CPU
  runner. It stays offline — no weight fetch — so what it does not cover is
  the remote-code load path, which needs network and is exercised the first
  time a job calls `from_pretrained`.
