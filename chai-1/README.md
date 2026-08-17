# chai-1 — Chai-1 biomolecular structure prediction

Co-folding model from Chai Discovery. Predicts structures of complexes
containing proteins, RNA, DNA, and small molecules, and — unlike Boltz-2 and
AlphaFold3 — reaches most of its accuracy **without MSAs**, using a traced
ESM-2 3B embedder in their place. Local MSAs still help and are supported.

- Upstream: https://github.com/chaidiscovery/chai-lab, release **v0.6.1**
  (2025-03-18). Commits continue on `main` but no newer release has been cut,
  and upstream itself recommends pinning.
- Weights: `chaiassets.com`, with a partial mirror at
  https://huggingface.co/chaidiscovery/chai-1. Ungated.
- **Licence: Apache-2.0 for the code *and* the weights.** Chai Discovery
  relicensed in November 2024; the original September-2024 research-only terms
  no longer apply, and academic and commercial use are both permitted. Any
  note in older lab documents describing Chai-1 as restrictively licensed is
  obsolete.
- Paper: Chai Discovery, "Chai-1: Decoding the molecular interactions of life",
  bioRxiv 2024. doi:10.1101/2024.10.10.615955

> **Status: experimental.** Not yet benchmarked on lab data. The memory
> section below is the part to read before writing an sbatch script — the host
> RAM figure is larger than people expect and is what kills jobs.

## Image tag

`ghcr.io/bradleylab/chai-1:latest` (also `:v1`, `:torch2.6-cu126`)

## Contents

- pytorch/pytorch 2.6.0: torch 2.6.0, CUDA 12.6, cuDNN 9, **Python 3.11**.
- `chai_lab==0.6.1` from PyPI, pinned to the release tag.
- `CHAI_DOWNLOADS_DIR=/opt/chai-downloads` — the weight cache. Override at
  runtime to mounted scratch. It is read at **import** time, so it must be set
  in the environment before Python starts.
- The upstream CLI: `chai-lab fold`, plus `chai-lab a3m-to-pqt` and
  `chai-lab citation`.

## Weights — and the incomplete Hugging Face mirror

Nothing is baked into the image. On first run chai_lab fetches ~7.0 GB across
8 assets into `$CHAI_DOWNLOADS_DIR`:

| Asset | Size | Host |
|---|---|---|
| `models_v2/trunk.pt`, `token_embedder.pt`, `feature_embedding.pt`, `diffusion_module.pt`, `confidence_head.pt`, `bond_loss_input_proj.pt` | ~1.18 GB total | chaiassets.com **and** the HF mirror |
| `esm/traced_sdpa_esm2_t36_3B_UR50D_fp16.pt` | 5.68 GB | **chaiassets.com only** |
| `conformers_v1.apkl` | 125 MB | **chaiassets.com only** |

**The HF mirror hosts only the six `.pt` model components.** Anyone
pre-staging weights by cloning `chaidiscovery/chai-1` will get 1.18 GB, think
they are done, and then have the job reach out to `chaiassets.com` for the
remaining 5.8 GB — which on an air-gapped or firewalled node means the job
fails at the ESM-embedding step, not at startup.

ESM embeddings are on by default, so that 5.68 GB is on the default path.
`--no-use-esm-embeddings` avoids the download, but it **changes what the model
is given as input**. That is a methodological change, not a deployment
convenience: do not reach for it to dodge a download, and record it if you use
it for real.

## GPU and memory requirements

Upstream recommends A100/H100 80 GB, or an L40S 48 GB. A measured PoseBench
benchmark (A100 80 GB, 5 diffusion samples) reports:

| Resource | Peak | Note |
|---|---|---|
| VRAM | **56.2 GB** | 5 diffusion samples, `low_memory=True` |
| **Host RAM** | **58.5 GB** | the figure people miss |
| Runtime | ~115 s per complex | small protein–ligand systems |

**Request ≥64 GB `--mem`.** A GPU-sized job with a default host-memory
allocation is the common way this model dies, and the Slurm OOM message points
at the host, not the GPU, so it reads as unrelated. Keep `low_memory=True` —
it is the default and turning it off raises the VRAM peak further.

## MSAs and templates — no network by default

The stock configuration makes **zero MSA network calls**: `use_msa_server`
defaults to False and `msa_directory` defaults to None. That is the posture we
want, and it is the default, so nothing needs to be disabled.

For better accuracy, supply local MSAs with `--msa-directory` pointing at a
directory of `aligned.pqt` files. The shipped converter turns a3m output from
our own mmseqs2 into that format:

```bash
chai-lab a3m-to-pqt <a3m-dir> --output-directory <pqt-dir>
```

Templates are off by default. Enabling the template server fetches CIFs from
RCSB per hit; the offline alternative is your own `m8` hit table plus
`CHAI_TEMPLATE_CIF_FOLDER`.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node (point enroot's scratch dirs off the 50 GB home):

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+chai-1+v1.sqsh 'docker://ghcr.io#bradleylab/chai-1:v1'
```

Submit a single-H100 job. Note the memory request:

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=96G
#SBATCH --time=04:00:00

SCRATCH=/scratch2/fs1/alexander.s.bradley
IN=$SCRATCH/chai_inputs/complex.fasta
OUT=$SCRATCH/chai_out/run01

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+chai-1+v1.sqsh \
     --container-mounts=$SCRATCH/chai-downloads:/opt/chai-downloads,$SCRATCH:$SCRATCH \
     bash -lc "
       export PYTHONNOUSERSITE=1
       chai-lab fold '$IN' '$OUT'
       ls '$OUT'/pred.model_idx_*.cif >/dev/null 2>&1 \
         || { echo 'chai-lab wrote no structures' >&2; exit 1; }
     "
```

Three things there are lab-standard or model-specific and easy to get wrong:

- `--mem=96G` — see the memory table. 64 GB is the floor, not a comfortable
  setting.
- `PYTHONNOUSERSITE=1` is required on Compute2: enroot bind-mounts `$HOME`
  into the container, so a stray `pip install --user` on the login node would
  otherwise shadow the container's site-packages.
- The output directory **must not exist or must be empty** — `run_inference`
  asserts on a non-empty output dir and dies at startup. Use a fresh
  per-run path.

## Minimal inference

Input is a FASTA with typed chain headers:

```
>protein|target
MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPT
>ligand|aspirin
CC(=O)Oc1ccccc1C(=O)O
```

```bash
export CHAI_DOWNLOADS_DIR=/mnt/scratch/chai-downloads   # before python starts
chai-lab fold complex.fasta ./out
```

Or from Python:

```python
from pathlib import Path
from chai_lab.chai1 import run_inference

candidates = run_inference(
    fasta_file=Path("complex.fasta"),
    output_dir=Path("./out"),          # must be empty or absent
    num_diffn_samples=5,               # drives the 56 GB VRAM peak
    low_memory=True,                   # default; leave it on
    seed=42,
)
```

Outputs are `pred.model_idx_<i>.cif` and `scores.model_idx_<i>.npz`, one pair
per diffusion sample.

## Offline / air-gapped operation

Fully air-gappable once the assets are staged, because the default
configuration contacts nothing: no MSA server, no template server, and no
usage telemetry was found in the package.

Pre-stage by running one small prediction on a networked node with
`CHAI_DOWNLOADS_DIR` pointed at its permanent home, then mount that directory
read-only on the air-gapped side. Do **not** pre-stage from Hugging Face
alone — see the mirror warning above; the 5.68 GB ESM embedder and the 125 MB
conformer pickle come from `chaiassets.com` only.

## Build notes / caveats

- **Why this base and not the NGC image esm/evo2 use.** chai_lab v0.6.1 pins
  `torch>=2.3.1,<2.7`; NGC PyTorch 25.04 ships `2.7.0a0+79aa174`. Whether that
  pre-release satisfies `<2.7` comes down to pip's pre-release handling rather
  than to anything stable, and a build should not rest on that. chai_lab also
  pins `numpy~=1.21` (<2.0), which an NGC base resists for the same reason
  boltz does: NVIDIA's compiled stack sits behind a global pip constraint file
  (`PIP_CONSTRAINT=/etc/pip/constraint.txt`). `pytorch/pytorch:2.6.0` is the
  highest release under chai_lab's ceiling, ships Python 3.11, and its CUDA
  12.6 build covers Hopper and bfloat16.

  The consequence is that this image and the boltz image sit on **different**
  bases (2.6.0-cu126 vs 2.8.0-cu129). That is deliberate and the two ceilings
  are incompatible; do not "harmonise" them.

- **Pinned to the tag, not `main`.** `main` has drifted off the v0.6.1
  dependency pins. Building against it would silently invalidate the base-image
  reasoning above. Move the pin by editing the `CHAI_VERSION` ARG default and
  committing; the CI workflow passes no `build-args`, so a re-run reproduces
  the committed pin.

- **The smoke test imports the real inference entrypoint.** No chai_lab module
  touches CUDA or the network at import scope, so
  `from chai_lab.chai1 import run_inference` works on the CPU build runner —
  no metadata-only fallback needed here. The test asserts the torch version is
  inside `[2.3, 2.7)` and that numpy is still 1.x, so a dependency that swaps
  either one fails the build rather than shipping. Note that `chai_lab` does
  expose a top-level `__version__`; per the ml-containers rule the smoke test
  reads `importlib.metadata.version` instead.

- **Chai-2 is not obtainable and is not what this image runs.** It has a
  bioRxiv technical report, no public weights, and no public inference code —
  it is available only through Chai Discovery's commercial platform. The
  `chai-lab` package ships Chai-1. Do not plan work around Chai-2.
