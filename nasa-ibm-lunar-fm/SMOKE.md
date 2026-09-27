# nasa-ibm-lunar-fm — Compute2 smoke test

Three steps. Importing the image and staging the weights happen once. The test
itself reruns two of upstream's published benchmark scores on an H100, because
an image that imports but computes something different would pass any lighter
check.

## 0. Import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+nasa-ibm-lunar-fm+v1.sqsh \
    'docker://ghcr.io#bradleylab/nasa-ibm-lunar-fm:v1'

file -b bradleylab+nasa-ibm-lunar-fm+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Stage the weights and benchmarks

12.7 GB: the backbone, the three fine-tuned repositories and the four
SomBench benchmarks (the WAC crater set is 4.1 GB of that), each pinned to a Hugging Face revision in
[`scripts/stage_weights.py`](scripts/stage_weights.py). Copy the script into
the destination first. It runs on `general-cpu`, so the image needs the GPU
override or enroot's driver hook stops the container before it starts. `HF_HOME`
is set inside the container command because pyxis lets the image's own
environment win over `--export`.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/nasa-ibm-lunar-fm
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=4 --mem=16G --time=03:00:00 \
       -J lfm-stage -o $D/stage-%j.out --wrap="
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+nasa-ibm-lunar-fm+v1.sqsh \
     --container-mounts=$D:/stage,/storage3/fs1/alexander.s.bradley/Active/hf-cache:/mnt/hfhome \
     --container-workdir=/stage \
     bash -c 'export HF_HOME=/mnt/hfhome HF_HUB_DISABLE_PROGRESS_BARS=1; python /stage/stage_weights.py'"
```

## 2. Rerun two published benchmarks on an H100

`lfm-verify-benchmarks` links the working directory, then runs
`terratorch test` with the released ice-prospectivity and IMP checkpoints on
their test splits, printing upstream's published scores above each run.

```bash
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=64G --time=01:00:00 \
       -J lfm-verify -o lfm-verify-%j.out --wrap='
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+nasa-ibm-lunar-fm+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley/Active/nasa-ibm-lunar-fm:/weights \
     --container-workdir=/tmp \
     lfm-verify-benchmarks'
```

**Reading the result.** Each published score is a mean ± standard deviation
over five seeds, and each released checkpoint is one of those seeds. A test
score inside that spread reproduces the paper. One well outside it means the
stack computes something different from upstream's, and the dependency versions
are the first place to look.
