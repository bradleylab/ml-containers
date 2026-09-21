# galaxi — Compute2 smoke test

Two stages, because the interesting failure is not the same as the cheap one.
Stage 1 proves the image runs and simulates a pattern with no staged data at
all. Stage 2 proves the evaluation path works against real pretrained models,
and needs the 618 MB example catalog.

## 0. One-time: import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+galaxi+v1.sqsh \
    'docker://ghcr.io#bradleylab/galaxi:v1'

file -b bradleylab+galaxi+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Stage 1 — simulate a pattern, no staged data

Runs on `general-cpu`. The image declares `NVIDIA_VISIBLE_DEVICES=all`, so a
CPU partition needs the override or enroot's driver hook fails and the
container never starts.

```bash
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=4 --mem=16G --time=00:20:00 \
       -J galaxi-smoke1 -o galaxi-smoke1-%j.out --wrap='
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+galaxi+v1.sqsh \
     --container-workdir=/tmp \
     bash -c "export PYTHONNOUSERSITE=1; python - <<PY
import glob, os
import pymatgen.core as pmg
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from galaxi.workflows.streamlined_workflow import create_default_config

cif = sorted(glob.glob(\"/opt/galaxi/tutorials/data/cif_files/*.cif\"))[0]
structure = pmg.Structure.from_file(cif)
pattern = XRDCalculator().get_pattern(structure)
print(os.path.basename(cif), \"->\", len(pattern.x), \"reflections\")
print(\"strongest at 2theta\", round(float(pattern.x[pattern.y.argmax()]), 3))

cfg = create_default_config()
print(\"config sections:\", sorted(cfg.keys()))
print(\"SMOKE1 OK\")
PY"'
```

## 2. Stage 2 — evaluate real patterns with the pretrained catalog

First stage the example catalog once, from a login node:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/galaxi
git clone https://github.com/Szymanski-Group/galaxi.git repo && cd repo
git checkout eb88287901e555aef20be7977d7ddcfbd77de5d2
python examples/pretrained_catalog/fetch_model_weights.py     # 618 MB, sha256-checked
```

That script verifies a checksum and unpacks one `.pth` per phase beside the
config and report files already in git. Then evaluate the bundled experimental
patterns — these are real measurements, not simulations, so this is the first
step that says anything about whether identification works.

```bash
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=32G --time=01:00:00 \
       -J galaxi-smoke2 -o galaxi-smoke2-%j.out --wrap='
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+galaxi+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley/Active/galaxi:/galaxi \
     --container-workdir=/galaxi/repo \
     bash -c "export PYTHONNOUSERSITE=1; python examples/pretrained_catalog/evaluate.py --help || ls examples/pretrained_catalog"'
```

The exact evaluation entry point is whatever
`examples/pretrained_catalog/README.md` documents at the pinned commit — read
it rather than assuming this command, and record the real one here once it has
run. The pristine patterns have one known target phase per file, so they are
the honest first check: the classifier should name the phase the filename says.

## 3. What is not covered

Neither stage touches the COD structures or the background profiles, so
neither exercises training-data generation. That needs the ~6.2 GB staged (see
`README.md`) and is the path that matters for building a mineral catalog —
worth its own timed run once the phases are chosen.
