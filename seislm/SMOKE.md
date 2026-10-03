# seislm — Compute2 smoke test

Four steps. Step 1 loads both checkpoints on an H100. Step 3 reruns the
paper's foreshock-aftershock classification for SeisLM-base and SeisLM-large,
because checkpoints that load and give finite features would pass any lighter
check even if the stack used them wrongly.

The weights are WashU-internal and held privately (see README.md). Every job
mounts the lab's copy at `/weights`; nothing here downloads them.

## 0. Import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+seislm+v1.sqsh \
    'docker://ghcr.io#bradleylab/seislm:v1'

file -b bradleylab+seislm+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Both checkpoints on the H100

`seislm-selftest` checks each checkpoint's size and SHA-256, loads it the way
upstream's demo notebook does, and runs it on a synthetic 30 s window. Expect
`cuda build 13.0`, capability `(9, 0)`, 11,359,568 parameters for base and
90,683,648 for large, and `SELFTEST OK` after each of the two runs.

`HF_HOME` and `SEISBENCH_CACHE_ROOT` are set inside the container command
because pyxis lets an image's own environment win over `--export`. Nothing in
these jobs downloads through either; setting them keeps any cache the imports
create on Storage3 rather than in `$HOME`.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/seislm
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=4 --mem=16G --time=00:30:00 \
       -J seislm-selftest -o $D/selftest-%j.out --wrap="
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+seislm+v1.sqsh \
     --container-mounts=$D/weights:/weights,$D:/seislm \
     --container-workdir=/tmp \
     bash -c 'export HF_HOME=/seislm/hf-home SEISBENCH_CACHE_ROOT=/seislm/seisbench-cache
              seislm-selftest --device cpu &&
              seislm-selftest --device cuda'"
```

## 2. Stage the NRCA data

The data are the ones upstream's demo notebook
(`examples/demo_seislm_foreshock_aftershock_classification.ipynb`) downloads:
`wetransfer_classify_generic_norcia-py_2024-06-24_1530.zip`, 524 MB, from
Google Drive file `1saaRH175pSFgl0zfQWFpedgj44pJKK3_`, holding three pickled
DataFrames of NRCA traces (pre-mainshock, Visso and post-mainshock; named
`.csv` but read with `pd.read_pickle`). `seislm-verify-shock stage` downloads
it with gdown, prints its SHA-256, unzips it and reads each file, printing its
trace and event counts.

Reading a pickle runs code, so `scripts/seislm-verify-shock` records the
archive's SHA-256 (549,375,104 bytes, `08c60ea9…4461`) and refuses any other
archive. The archive carries no license. The same Laurenti et
al. (2024) dataset is published as D-SET on Zenodo
([doi:10.5281/zenodo.12795621](https://doi.org/10.5281/zenodo.12795621),
CC-BY-4.0, 5.08 GB, nine stations including NRCA as HDF5 waveforms with CSV
attributes), but upstream's pipeline reads only the Drive pickles and provides
no conversion from D-SET, so the test uses the Drive archive.

The job runs on `general-cpu`, so the image needs the GPU override or enroot's
driver hook stops the container before it starts.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/seislm
mkdir -p $D/data/foreshock_aftershock_NRCA
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=2 --mem=32G --time=01:00:00 \
       -J seislm-stage -o $D/stage-%j.out --wrap="
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+seislm+v1.sqsh \
     --container-mounts=$D:/seislm \
     --container-workdir=/tmp \
     bash -c 'export HF_HOME=/seislm/hf-home SEISBENCH_CACHE_ROOT=/seislm/seisbench-cache
              seislm-verify-shock stage /seislm/data/foreshock_aftershock_NRCA'"
```

## 3. Rerun the published accuracies on an H100

The target is Liu et al. (2024), arXiv:2410.15765v1, Figure 6 (page 9),
"Confusion matrices of models evaluated on the test fold of the
foreshock–aftershock classification dataset", whose panel titles give
**SeisLM-base 65.11 %** and **SeisLM-large 74.22 %**, against 58.33 % for the
ConvNet baseline of Laurenti et al. (2024). The paper has no table of
downstream results; the phase-picking results exist only as plotted points, so
these titles are its only exact downstream numbers. They come from nine classes
in time order: four foreshock windows FEQ1–FEQ4, the Visso event and four
aftershock windows AEQ1–AEQ4.

"Acc" there is the mean of the nine diagonal entries of the row-normalized
confusion matrix, each rounded to a whole percent, which is what upstream's demo
notebook computes. All three published values are a whole number divided by
nine (586/9, 668/9, 525/9), which fits that definition. The script reports this
figure, the unrounded mean recall, and Lightning's test accuracy.

[`scripts/seislm-verify-shock`](scripts/seislm-verify-shock) follows upstream's
run script (`src/foreshock_aftershock/foreshock_aftershock_run.py`) and the
config for each model (`seisLM/configs/foreshock_aftershock/seisLM_{base,large}_shock_classifier.json`):
upstream's data pipeline with a temporal 70/10/20 split by event, seed 42, each
class truncated to equal size, per-trace demean and standard-deviation scaling;
the pretrained encoder with its convolutional feature encoder frozen and
SpecAugment-style time masking during training (probability and span
inherited from the pretraining config); two strided convolution blocks, mean pooling
and a linear layer as the head; AdamW at 4e-4 with weight decay 0.1 (and
eps 1e-7 for base), cosine decay with no warmup, 15 epochs; then
`trainer.test` on `last.ckpt`. Points upstream leaves open:

- **GPUs and batch.** Both configs train on 2 GPUs × 16 under DDP; upstream's
  demo notebook trains on one GPU at 32, the same global batch. The script
  follows the notebook, so the head's batch normalization sees 32 traces per
  step rather than 16 per GPU.
- **Number of runs.** The paper reports one run per model with no spread; the
  configs fix seed 42. `--seed N` reseeds training (head initialization, time
  masking, batch order) after the data pipeline, which keeps upstream's seed-42
  trace selection, so repeated runs measure training variation alone.
- **Number of classes.** The run script defaults to four classes; the figure
  shows nine, and the notebook sets nine. The script uses nine and the full
  training fold.
- **Validation fold.** Upstream's code passes the test fold as the validation
  set during fitting. It feeds only the logs and the best-checkpoint pick, and
  the result is read from `last.ckpt`, so the script keeps it as written.
- **Numerics.** The script sets what upstream's `__main__` sets
  (`cudnn.deterministic`, `cudnn.benchmark`, float32 matmul precision `high`).
  Upstream's runs used A100s (the paper's pretraining, the demo notebook's
  fine-tune), so bitwise agreement is not expected.

Upstream's demo notebook log shows the SeisLM-base fine-tune taking 665 s for
15 epochs on one A100-40GB; upstream logs no time for SeisLM-large, which has
eight times the parameters. The time limit below leaves room for both.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/seislm
mkdir -p $D/runs
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=64G --time=04:00:00 --array=0-1 \
       -J seislm-shock -o $D/runs/shock-%A_%a.out --wrap='
set -- base large; shift $SLURM_ARRAY_TASK_ID; V=$1
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+seislm+v1.sqsh \
     --container-mounts='$D'/weights:/weights,'$D':/seislm \
     --container-workdir=/tmp \
     bash -c "export HF_HOME=/seislm/hf-home SEISBENCH_CACHE_ROOT=/seislm/seisbench-cache
              nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
              seislm-verify-shock run /seislm/data/foreshock_aftershock_NRCA /seislm/runs/shock_$V $V"'
```

Each task makes `runs/shock_<variant>/` a git repository with the data linked
in at `data/foreshock_aftershock_NRCA`, which is where upstream's
`project_path` looks, and writes the Lightning checkpoints and CSV logs under
`results/models/` and the scores to `result.json`. The log ends with the
accuracy beside the published one.

**Reading the result.** The published figures are single runs. On an H100,
six runs per model (the config's seed 42 and `--seed 0` to `4`) gave:

| Model | Runs (seed 42; seeds 0–4) | Mean | Range | Published |
|---|---|---|---|---|
| SeisLM-base | 67.78; 72.89, 73.44, 61.56, 64.22, 64.33 | 67.37 % | 61.56–73.44 % | 65.11 % |
| SeisLM-large | 69.44; 75.89, 76.44, 71.56, 72.00, 72.22 | 72.92 % | 69.44–76.44 % | 74.22 % |

Both published values fall inside the spread, and every run beats the ConvNet's
58.33 %. Training alone moves a single run by up to 12 points for base and 7
for large, so compare a new run against these ranges, not against the
published figure alone. One near the
ConvNet's 58.33 % or below means the pretrained encoder is not what the head is
seeing, and the checkpoint loading is the first place to look; the build's
selfcheck runs the same pipeline from random weights, so the pipeline itself
runs either way.
