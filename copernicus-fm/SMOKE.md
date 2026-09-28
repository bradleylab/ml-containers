# copernicus-fm — Compute2 smoke test

Four steps. Importing the image, staging the weights and staging the dataset
happen once. The test
itself reruns one of upstream's published Copernicus-Bench scores on an H100,
because an image that imports but computes something different would pass any
lighter check.

**The published score.** Wang et al. (2025), "Towards a Unified Copernicus
Foundation Model for Earth Vision", [arXiv:2503.11849v3](https://arxiv.org/abs/2503.11849),
Table 4: Copernicus-FM ViT-B/16, frozen encoder with a linear probe, Copernicus-Bench
EuroSAT-S1, test overall accuracy **87.2 ± 0.1 %** (mean and standard deviation of
three runs). The same table gives DOFA 81.7 ± 0.1 and CROMA 83.9 ± 0.1 on that
task. EuroSAT-S1 is Sentinel-1 VV and VH at 64 × 64 pixels, 10 land-cover classes,
16,200 / 5,400 / 5,400 train, validation and test images (Table 3), and downloads
without credentials.

## 0. Import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+copernicus-fm+v1.sqsh \
    'docker://ghcr.io#bradleylab/copernicus-fm:v1'

file -b bradleylab+copernicus-fm+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Stage the weights

The checkpoint is kept on lab storage rather than in the image (README,
"Weights"). The job runs on `general-cpu`, so it needs the GPU override.

```bash
W=/storage3/fs1/alexander.s.bradley/Active/copernicus-fm/weights
mkdir -p $W
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=2 --mem=8G --time=00:30:00 \
       -J cfm-weights -o $W/stage-%j.out --wrap="
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+copernicus-fm+v1.sqsh \
     --container-mounts=$W:/stage \
     python /opt/copernicus-fm/stage_weights.py"
```

The log should end with `SHA-256 ok`.

## 2. Stage EuroSAT-S1

`eurosat_s1.zip` is 920,683,574 bytes. TorchGeo's `CopernicusBenchEuroSATS1`
fetches it from the `wangyi111/Copernicus-Bench` dataset at a pinned revision,
checks its SHA-256 and extracts it. Staging it once keeps the three benchmark jobs
from racing to download the same file. The job runs on `general-cpu`, so the
image needs the GPU override or enroot's driver hook stops the container before it
starts.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/copernicus-fm/copernicus-bench
mkdir -p $D
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=2 --mem=8G --time=01:00:00 \
       -J cfm-stage -o $D/stage-%j.out --wrap="
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+copernicus-fm+v1.sqsh \
     --container-mounts=$D:/data \
     python -c \"from torchgeo.datasets import CopernicusBenchEuroSATS1 as D; [print(s, len(D('/data', split=s, download=True))) for s in ('train', 'val', 'test')]\""
```

The log should end with `train 16200`, `val 5400` and `test 5400`.

## 3. Rerun the published score on an H100, three seeds

`cfm-verify-eurosat-s1` trains the linear probe for 50 epochs on the frozen
encoder, keeps the epoch with the best validation accuracy, and reports test
accuracy with the published value beside it. Upstream's committed runner uses seed
42 and does not record the other two; 43 and 44 stand in for them.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/copernicus-fm
mkdir -p $D/verify
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=64G --time=04:00:00 --array=0-2 \
       -J cfm-verify -o $D/verify/cfm-verify-%A_%a.out --wrap='
SEED=$((42 + SLURM_ARRAY_TASK_ID))
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+copernicus-fm+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley/Active/copernicus-fm/copernicus-bench:/data,/storage3/fs1/alexander.s.bradley/Active/copernicus-fm/verify:/work,/storage3/fs1/alexander.s.bradley/Active/copernicus-fm/weights:/weights \
     --container-workdir=/work \
     cfm-verify-eurosat-s1 --root /data --seed $SEED --out /work/eurosat-s1-seed$SEED.json'
```

Each log opens with the device line, which should read `device cuda` and
`cuda build 12.9`, prints one line per epoch, and finishes with a line of the form
`EuroSAT-S1 test OA ... (best val OA ... at epoch ...); published 87.2 ± 0.1`
followed by the same result as JSON. Collect the three results with:

```bash
grep -h 'EuroSAT-S1 test OA' /storage3/fs1/alexander.s.bradley/Active/copernicus-fm/verify/cfm-verify-*.out
```

**Reading the result.** The published standard deviation of 0.1 over three seeds
says the protocol barely moves with the seed, so the three-seed mean should sit
close to 87.2. A mean several tenths away reflects a difference in the stack or
the protocol, not seed noise. A mean down near DOFA's 81.7 or CROMA's 83.9 says the
encoder is not using what Copernicus-FM learned; check the wavelength units
(nanometers) and that the log has no `Downloading:` line from torch.hub, which
would mean the staged checkpoint was not found at `/weights`. The first thing to rerun on a
shortfall is the same job with `--layernorm-eps 1e-6`, which builds the encoder
with upstream's LayerNorm constant in place of TorchGeo's default (README,
"Differences from upstream's code").
