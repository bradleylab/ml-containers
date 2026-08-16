# boltz — Boltz-2 structure and binding-affinity prediction

Biomolecular co-folding model from the Barzilay/Jaakkola group at MIT.
Predicts the structure of complexes containing proteins, RNA, DNA, and small
molecules from sequence and chemical-component identity, and — from a block in
the same input file — predicts binding affinity for a nominated ligand chain.
Successor to Boltz-1.

- Upstream: https://github.com/jwohlwend/boltz, tag **v2.2.1** (2025-09-08).
  `main` still receives commits but no newer tag has been cut.
- Weights: https://huggingface.co/boltz-community, with a primary host at
  `model-gateway.boltz.bio`. Ungated.
- **Licence: MIT for the code *and* the weights.** Academic and commercial use
  are both permitted, with no acceptance step and no gated download.

> **Status: experimental.** Not yet benchmarked on lab data. Read the silent
> out-of-memory section below before running anything unattended — it is the
> one failure mode that will otherwise cost you a day.

## Image tag

`ghcr.io/bradleylab/boltz:latest` (also `:v1`, `:torch2.8-cu129`)

## Contents

- pytorch/pytorch 2.8.0: torch 2.8.0, CUDA 12.9, cuDNN 9, **Python 3.11**.
- `boltz[cuda]==2.2.1` from PyPI. The `[cuda]` extra pulls the cuEquivariance
  kernel wheels, pinned here to 0.6.1 (see build notes).
- `BOLTZ_CACHE=/opt/boltz-cache` — the weight cache. Override at runtime with
  an **absolute** path pointing at mounted scratch.
- The upstream `boltz` CLI: `boltz predict <input>`.

## Silent out-of-memory — read this first

`boltz predict` catches CUDA OOM per batch, prints

```
| WARNING: ran out of memory, skipping batch
```

and **continues to exit 0** ([upstream issue #167]). A Slurm job that produced
no structures at all therefore reports success, and nothing downstream
notices until someone opens the output directory. This behaviour is in both
`boltz1.py` and `boltz2.py` in v2.2.1.

**Every job must gate on output files existing, not on the exit status.**
Predictions land at:

```
<out_dir>/boltz_results_<input-stem>/predictions/<record-id>/<record-id>_model_0.cif
<out_dir>/boltz_results_<input-stem>/predictions/<record-id>/affinity_<record-id>.json
```

so the gate is a one-liner (it is in the sbatch example below). When a run does
run out of memory, the flags that help are `--no_kernels`, `--subsample_msa`
with `--num_subsampled_msa`, `--max_msa_seqs`, and `--max_parallel_samples`.

[upstream issue #167]: https://github.com/jwohlwend/boltz/issues/167

## Checkpoints and GPU requirements

Nothing is baked into the image. On first run boltz fetches ~6.2 GB into
`$BOLTZ_CACHE`:

| Asset | Size | What it is |
|---|---|---|
| `boltz2_conf.ckpt` | 2.29 GB | structure + confidence model |
| `boltz2_aff.ckpt` | 2.06 GB | affinity head |
| `mols.tar` | 1.86 GB | CCD chemical-component data (HF-only; no gateway mirror) |

| Resource | Figure | Source |
|---|---|---|
| VRAM, structure | ~11 GB | third-party production report |
| VRAM, affinity | ~7–8 GB | same |
| VRAM, vendor floor | **≥48 GB** | NVIDIA NIM support matrix |
| Host RAM | 64 GB is comfortable | — |
| Runtime, ~500 residues | ~10–60 s | stock PyTorch path; NVIDIA's TensorRT numbers are 1.45–6.4× faster and do not apply here |

The gap between the ~11 GB measured and NVIDIA's ≥48 GB requirement is real
and is about headroom on large multimers, not about the typical case. A single
H100 80 GB is comfortable for ordinary work; the risk is large complexes, and
the failure is silent (above).

## MSAs — required, and generated locally

**Boltz-2 needs MSAs and will not fetch them for you.** `--use_msa_server`
defaults to False, and a run with neither an MSA nor an explicit opt-out
fails. Three offline paths:

1. **Precomputed `.a3m` per chain** — point each chain's `msa:` field at a
   file. This is the lab default.
2. **A CSV with `sequence` and `key` columns**, which is how multi-chain MSA
   pairing is expressed for complexes.
3. **`msa: empty`** for single-sequence mode. Upstream does not recommend it;
   it reduces accuracy. Use it only for a deliberate ablation, and say so.

We generate MSAs locally with our own mmseqs2, reusing the AlphaFold3 database
investment, and do not depend on the public ColabFold endpoint. If a
self-hosted ColabFold/MMseqs2 server is ever stood up, `--msa_server_url`
points at it — but the default posture for this image is no MSA network calls
at all.

## Binding affinity

There is no separate affinity CLI. Affinity is requested by a block in the
input YAML:

```yaml
properties:
  - affinity:
      binder: B          # the ligand chain id
```

Constraints, from upstream: exactly one ligand chain; ≤128 heavy atoms, with
the authors advising ≤56; **protein targets only** — RNA, DNA, and cofactor
targets will run but are not reliable. Outputs are `affinity_pred_value`
(log10 IC50) and `affinity_probability_binary`. Affinity adds roughly
1.5–3× to runtime; templates roughly 4×.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node (point enroot's scratch dirs off the 50 GB home):

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+boltz+v1.sqsh 'docker://ghcr.io#bradleylab/boltz:v1'
```

Submit a single-H100 job. Mount scratch over `$BOLTZ_CACHE` so the 6.2 GB is
fetched once and reused, and gate on the output files:

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

SCRATCH=/scratch2/fs1/alexander.s.bradley
IN=$SCRATCH/boltz_inputs/complex.yaml
OUT=$SCRATCH/boltz_out

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+boltz+v1.sqsh \
     --container-mounts=$SCRATCH/boltz-cache:/opt/boltz-cache,$SCRATCH:$SCRATCH \
     bash -lc "
       export PYTHONNOUSERSITE=1
       boltz predict '$IN' --out_dir '$OUT' --accelerator gpu --devices 1
       ls '$OUT'/boltz_results_*/predictions/*/*_model_0.cif >/dev/null 2>&1 \
         || { echo 'boltz exited 0 but wrote no structures — treating as failure' >&2; exit 1; }
     "
```

Two lines there are lab-standard:

- `PYTHONNOUSERSITE=1` is required on Compute2: enroot bind-mounts `$HOME`
  into the container, so a stray `pip install --user` on the login node would
  otherwise shadow the container's site-packages.
- The `ls ... || exit 1` gate is the silent-OOM guard. Do not drop it.

## Minimal inference

Input is a YAML (or FASTA) file. A protein–ligand complex with a precomputed
MSA and an affinity request:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPT
      msa: /scratch2/fs1/alexander.s.bradley/msas/A.a3m
  - ligand:
      id: B
      smiles: "CC(=O)Oc1ccccc1C(=O)O"
properties:
  - affinity:
      binder: B
```

```bash
export BOLTZ_CACHE=/mnt/scratch/boltz-cache      # absolute path, always
boltz predict complex.yaml --out_dir ./out --accelerator gpu --devices 1
```

## Offline / air-gapped operation

The image makes no network calls at inference time once two things are true:
the cache is pre-staged, and MSAs are supplied as files.

Pre-stage the cache by running any small prediction once on a node with
network, with `BOLTZ_CACHE` pointed at its permanent home — the three assets
land there and are reused thereafter. Fetching them by hand also works:
`boltz2_conf.ckpt` and `boltz2_aff.ckpt` come from `model-gateway.boltz.bio`
with a Hugging Face fallback at `boltz-community/boltz-2`; `mols.tar` is
Hugging Face only, with no gateway mirror.

The only other outbound call boltz can make is the ColabFold MSA API, and it
is opt-in (`--use_msa_server`, default False). `wandb` is a dependency but is
not exercised on the inference path, so there is no runtime telemetry.

## Build notes / caveats

- **Why not an NGC base.** boltz pins `numpy>=1.26,<2.0`. NGC PyTorch images
  carry a large NVIDIA-built stack behind a global pip constraint file
  (`PIP_CONSTRAINT=/etc/pip/constraint.txt` — it is in the 25.04 image
  config), and forcing a numpy downgrade under that stack is how those images
  break. A plain `pytorch/pytorch` tag has no such stack, so pip resolves
  numpy freely and the pin costs nothing. boltz's `requires-python
  >=3.10,<3.13` also rules out Python 3.13; this base is 3.11.

- **cuEquivariance is pinned to 0.6.1, and the pin is load-bearing.** boltz's
  `[cuda]` extra asks only for `cuequivariance*>=0.5.0`. Version 0.11.1
  (2026-08-07) promoted `torch>=2.11` from a test extra to a hard runtime
  dependency, so an unpinned build **replaces the base image's torch and CUDA
  stack with PyPI wheels** — quietly, and differently on different build days.
  0.6.1 (2025-09-04) is the release that was current four days before boltz
  v2.2.1 shipped, i.e. what upstream's own install line resolved to at release
  time. Move the pin by editing the `CUEQ_VERSION` ARG default and committing,
  so the pin lives in git history. The CI workflow passes no `build-args`, so a
  re-run reproduces the committed pin.

- **The smoke test imports a real model class.** boltz reaches cuEquivariance
  only from inside `kernel_triangular_mult()`, not at module scope, so
  `from boltz.model.models.boltz2 import Boltz2` succeeds on the CPU build
  runner with no GPU and no network — unlike esm and evo2, which have to stop
  at the metadata layer. The test also asserts the cuEquivariance version, the
  numpy major, and that torch is still the base image's build, so a dependency
  that silently swaps one of them fails the build instead of shipping.

- **`BOLTZ_CACHE` must be absolute.** boltz raises rather than falling back if
  it is relative. A relative path in an sbatch script is a job that dies at
  startup with a message you will not see until you read the log.

- **The image carries a duplicate cuBLAS.** `cuequivariance-ops-cu12` requires
  `nvidia-cublas-cu12>=12.5.0`, so pip installs that wheel alongside the CUDA
  runtime the base image already ships. It costs a few hundred MB and is not
  worth fighting.
