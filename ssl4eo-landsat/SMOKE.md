# ssl4eo-landsat — Compute2 smoke test

Four steps. Step 1 checks the GPU path with every baked checkpoint. Step 3
reruns one published SSL4EO-L benchmark score on an H100, because weights that
load and produce finite numbers would pass any lighter check even if the stack
used them wrongly.

## 0. Import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+ssl4eo-landsat+v1.sqsh \
    'docker://ghcr.io#bradleylab/ssl4eo-landsat:v1'

file -b bradleylab+ssl4eo-landsat+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Every checkpoint on the H100

`ssl4eo-check cuda` prints the torch build, the device name and compute
capability, then loads all 30 checkpoints from the bake and runs each on
synthetic input. Expect `cuda build 12.9`, capability `(9, 0)`, and
`SSL4EO-L CHECK OK: 30 checkpoints on cuda`.

```bash
IMG=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+ssl4eo-landsat+v1.sqsh
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=4 --mem=16G --time=00:30:00 \
       -J ssl4eo-check -o ssl4eo-check-%j.out --wrap="
srun --container-image=$IMG --container-workdir=/tmp ssl4eo-check cuda"
```

## 2. Stage the benchmark

The test uses the Landsat 7 ETM+ SR imagery with CDL 2019 masks, the smallest
SSL4EO-L benchmark pair: 6.56 GB of imagery and 0.42 GB of masks as `.tar.gz`
on Hugging Face (`torchgeo/ssl4eo-l-benchmark` at the revision TorchGeo 0.10.0
pins). `ssl4eo-benchmark stage` downloads both through TorchGeo, checks their
SHA-256 and extracts them next to the archives, so the destination needs room
for the extracted files as well. It runs on `general-cpu`, so the image needs
the GPU override or enroot's driver hook stops the container before it starts.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/ssl4eo-landsat/benchmark
mkdir -p $D
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=4 --mem=16G --time=03:00:00 \
       -J ssl4eo-stage -o $D/stage-%j.out --wrap="
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+ssl4eo-landsat+v1.sqsh \
     --container-mounts=$D:/data \
     --container-workdir=/tmp \
     ssl4eo-benchmark stage /data"
```

The log ends with the number of training patches: the split is 70/15/15 of
the 25,000-patch dataset, fixed by the dataset class's own seed.

## 3. Rerun the published score on an H100

The target is Stewart et al. (2023), SSL4EO-L, NeurIPS Datasets and Benchmarks,
Table 2, row Landsat 7 (ETM+) · Level-2 (SR) · ResNet-18 · MoCo, CDL column:
**67.30 % overall accuracy, 50.71 mIoU**. The same U-Net on ImageNet weights
scores 60.70 % and 43.58. That 6.60-point accuracy gap is the largest MoCo gain
in Table 2, so this cell separates the SSL4EO-L weights from a backbone that
merely runs.

[`scripts/ssl4eo-benchmark`](scripts/ssl4eo-benchmark) follows the paper's
recipe and TorchGeo's released config for this cell
(`experiments/ssl4eo/landsat/conf/ssl4eo_benchmark_etm_sr_cdl.yaml`,
`releases/v0.5`): the `ResNet18_Weights.LANDSAT_ETM_SR_MOCO` encoder frozen, a
U-Net decoder trained with cross-entropy and class 0 ignored, 18 CDL classes,
batch 64, 20–100 epochs, learning-rate plateau patience 6, seed 0. Two points
are not fully specified upstream:

- **Learning rate.** The paper tuned only the learning rate and does not list
  the value chosen per cell; the released config has 1e-3 and the paper names
  3e-3 as the most common optimum. The job runs both as a two-task array.
- **Early stopping.** The paper stopped early but gives no stopping patience.
  The script trains to 100 epochs and tests the checkpoint with the lowest
  validation loss, the model early stopping would have kept.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/ssl4eo-landsat
mkdir -p $D/runs
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=16 --mem=64G --time=12:00:00 --array=0-1 \
       -J ssl4eo-bench -o $D/bench-%A_%a.out --wrap='
set -- 1e-3 3e-3; shift $SLURM_ARRAY_TASK_ID; LR=$1
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+ssl4eo-landsat+v1.sqsh \
     --container-mounts='$D'/benchmark:/data,'$D'/runs:/runs \
     --container-workdir=/tmp \
     ssl4eo-benchmark run /data /runs/etm_sr_cdl_lr$LR $LR'
```

Each task writes its best checkpoint and a CSV metrics log under `runs/etm_sr_cdl_lr<LR>/`, and its log ends with the test
accuracy and mIoU beside the published and ImageNet values.

**Reading the result.** Take the learning rate whose best validation loss is
lower; the paper does not state its selection criterion, and validation loss
is what the script monitors. Table 2 reports one run per cell with
no spread; the three-seed cloud-detection results in Table 1 give a sense of
seed-to-seed variation, with standard deviations of 1.94 to 5.17 accuracy points
across the ResNet MoCo rows. An accuracy near 67.30 % reproduces the paper. One
nearer the ImageNet 60.70 % than the MoCo 67.30 % means the SSL4EO-L encoder is
not what the decoder is seeing, and the weight loading is the first place to
look.

**Which IoU to compare.** TorchGeo 0.5, the first release containing the
paper's code, logged micro-averaged Jaccard as its IoU. In TorchGeo 0.10 that is
`test_OverallJaccardIndex`, which the script reports; `test_AverageJaccardIndex`
is the macro average, a different quantity. With every pixel assigned a class,
micro IoU is fixed by accuracy as acc / (2 − acc), and Table 2 follows that
identity to within 0.02 in 39 of its 45 rows, this one included
(0.6730 / 1.3270 = 0.5072). The accuracy is therefore the one independent
number to compare.
