# aurora — Compute2 smoke test

Three steps. Importing the image and staging the weights and initial states
happen once. The test itself runs upstream's small-model reference test, whose
answer is fixed, and then one 6-hour forecast from ERA5 and one from IFS HRES
T0, each scored against the verifying state and against persistence. An image
that imports but loads a checkpoint wrongly, or feeds it misordered data, would
pass any lighter check.

## 0. Import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+aurora+v1.sqsh \
    'docker://ghcr.io#bradleylab/aurora:v1'

file -b bradleylab+aurora+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Stage the weights and initial states

10.8 GB of weights and reference data, each file pinned to one Hugging Face revision in
[`scripts/stage_weights.py`](scripts/stage_weights.py) (in the image at
`/opt/aurora/stage_weights.py`), then about 0.8 GB per forecast case read
anonymously from WeatherBench2 on Google Cloud Storage and cached as netCDF, so
the GPU job needs no network. The job runs on `general-cpu`, so the image needs
the GPU override or enroot's driver hook stops the container before it starts.
`HF_HOME` is set inside the container command, as in the lab's other images,
because pyxis lets an image's own environment win over `--export`.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/aurora
mkdir -p $D
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=4 --mem=16G --time=03:00:00 \
       -J aurora-stage -o $D/stage-%j.out --wrap="
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+aurora+v1.sqsh \
     --container-mounts=$D:/stage,/storage3/fs1/alexander.s.bradley/Active/hf-cache:/mnt/hfhome \
     --container-workdir=/stage \
     bash -c 'export HF_HOME=/mnt/hfhome HF_HUB_DISABLE_PROGRESS_BARS=1
              python /opt/aurora/stage_weights.py &&
              aurora-verify-onestep pretrained --cache /stage/initial_states --fetch-only &&
              aurora-verify-onestep finetuned --cache /stage/initial_states --fetch-only'"
```

The two cases are the dates of upstream's worked examples: ERA5 at 2023-01-01
00 and 06 UTC for the pretrained model (`docs/example_era5.ipynb`), and IFS HRES
T0 at 2022-05-11 00 and 06 UTC for the fine-tuned model
(`docs/example_hres_t0.ipynb`). Each cache file also holds the 12 UTC state the
forecast is scored against.

## 2. Reference test and two forecasts on an H100

```bash
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=64G --time=01:00:00 \
       -J aurora-verify -o aurora-verify-%j.out --wrap='
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+aurora+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley/Active/aurora:/weights \
     --container-workdir=/tmp \
     bash -c "export HF_HUB_OFFLINE=1
              nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
              aurora-verify-small --device cpu
              aurora-verify-small --device cuda
              aurora-verify-onestep small --cache /weights/initial_states
              aurora-verify-onestep pretrained --cache /weights/initial_states
              aurora-verify-onestep finetuned --cache /weights/initial_states"'
```

`HF_HUB_OFFLINE=1` makes any stray Hugging Face call fail at once instead of
waiting on a node without egress; every file these scripts read is in
`/weights`. Each command prints PASS or FAIL and the job runs all five, so one
failure does not hide the others.

## Reading the result

**`aurora-verify-small`** repeats upstream's `test_aurora_small`
(`tests/test_model.py` at v2.0.1): the small checkpoint in float64 predicts one
step from a saved batch, and each variable's mean absolute error, relative to
the saved prediction, must fall within upstream's tolerance for that variable
(1e-4 for `2t`, `msl` and `t`; 5e-3 for the winds and `q`). The CPU run is the
pass/fail check, because a Compute2 node is the platform upstream's CI passes
this test on, x86-64 Linux. The CUDA run is scored against the same
tolerances. If it misses only one of the 1e-4 variables by a small margin while
the CPU run passes, the difference most likely lies in the GPU's numerical path
rather than the install; the forecasts that follow then show whether it
matters.

**`aurora-verify-onestep`** prints, for z500, t850, u850, 2t, 10u and msl, the
cos(latitude)-weighted global RMSE of the 12 UTC prediction against the 12 UTC
analysis, beside the same score for persistence (the 06 UTC state carried
forward unchanged), and the ratio of the two. It passes when every score is
finite and every ratio is below 1. Upstream publishes no score for these single
initializations — its examples show maps, not numbers — so there is no expected
value to match. What separates a working model from a broken one is size: a
6-hour Aurora step should sit far below persistence on every variable, while a
misordered grid, a wrong static field or a mismatched checkpoint is likely to
give errors near or above it. The `small` case on the same ERA5 initial state
is the reference within the job; a full pretrained model scoring worse than the
debugging model points at loading or data before anything else. The pretrained case also prints
the peak GPU memory allocated, which upstream puts at approximately 40 GB for
the full model on the 0.25° grid.
