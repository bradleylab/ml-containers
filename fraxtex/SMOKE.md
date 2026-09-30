# fraxtex — Compute2 smoke test

Three steps. Importing the image and staging FraXet happen once. Step 1 runs
the self-test with CUDA required and then rescores the published models on the
FraXet test split, because a model that loads but is fed the wrong input still
produces plausible masks; only the published numbers catch that. Step 2 is the
run on the lab's own outcrop imagery.

## 0. Import the image

On a login node — a download, not compute:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+fraxtex+v1.sqsh \
    'docker://ghcr.io#bradleylab/fraxtex:v1'

file -b bradleylab+fraxtex+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. Reproduce the published FraXet scores

### Stage FraXet

2.64 GB from Zenodo ([doi:10.5281/zenodo.17069947](https://doi.org/10.5281/zenodo.17069947),
CC-BY-4.0). `fraxet-stage` checks the archive against the md5 the record lists
before unpacking it. It runs on `general-cpu`, so the image needs the GPU
override or enroot's driver hook stops the container before it starts.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/fraxtex
mkdir -p $D
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=2 --mem=8G --time=02:00:00 \
       -J fraxet-stage -o $D/stage-%j.out --wrap="
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+fraxtex+v1.sqsh \
     --container-mounts=$D:/data \
     fraxet-stage /data"
```

### Score the four models on an H100

`fraxtex-selftest --device cuda` fails unless CUDA runs every model. The two
RGB+DEM models are then scored twice, once with the upstream loader's DEM
treatment and once with per-patch min–max (see "The DEM channel" in
[`README.md`](README.md)); the RGB models once each.

```bash
D=/storage3/fs1/alexander.s.bradley/Active/fraxtex
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=32G --time=01:00:00 \
       -J fraxtex-verify -o $D/verify-%j.out --wrap="
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+fraxtex+v1.sqsh \
     --container-mounts=$D:/data \
     bash -c 'set -e
fraxtex-selftest --device cuda
for m in unet-rgbdem segformer-rgbdem; do
  for d in loader patch-minmax; do
    fraxtex-eval-fraxet --fraxet /data/FraXet_v0.1 --device cuda \
      --model \$m --dem-mode \$d --json /data/eval-\$m-\$d.json
  done
done
for m in unet-rgb sam2-rgb; do
  fraxtex-eval-fraxet --fraxet /data/FraXet_v0.1 --device cuda \
    --model \$m --json /data/eval-\$m.json
done'"
```

**Reading the result.** The reference is Table D1 of Fatihi et al.
([doi:10.5194/egusphere-2026-1097](https://doi.org/10.5194/egusphere-2026-1097)),
repeated as Table E1 in the revised manuscript, for all 1,812 test patches, and
the M_all rows of Table 6 for each dataset. The script prints them beside each
measured value. Over the whole test split:

| Model | AE | MSE | SSIM | PSNR | Acc | Prec | F1 | Rec | Spec | ROC | IoU | CK | FracSim | Loss |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| U-Net | 0.07 | 0.03 | 0.49 | 14.7 | 0.84 | 0.50 | 0.48 | 0.45 | 0.92 | 0.69 | 0.31 | 0.39 | 3.40 | 0.02 |
| SegFormer | 0.10 | 0.05 | 0.40 | 13.2 | 0.79 | 0.38 | 0.44 | 0.52 | 0.84 | 0.68 | 0.28 | 0.32 | 3.43 | 0.02 |

The published values are rounded to two decimals and come from a test loader
that flipped patches at random, so agreement can be judged to about the second
decimal and no further. A DEM mode that lands on the published F1, IoU, precision and
recall for both models is the treatment the released weights expect, and that
is the `--dem-scaling` to use in step 2 (`zero` or `patch-minmax-255` if it is
the loader mode, `patch-minmax` if it is min–max). Scores several hundredths
away in both modes mean the weights or the preprocessing differ from what was
published, and the RGB+DEM models should not be used until that is explained.
The preprint publishes nothing for `unet-rgb` or `sam2-rgb`; their scores are
the lab's own reference on this protocol, and the SAM 2 figures include the 2×
enlargement described in the README.

## 2. The lab's outcrop orthomosaics

Inputs, per site:

- an RGB orthomosaic as a GeoTIFF, 8-bit (otherwise choose `--rgb-range` for the
  sensor), with nodata or an alpha band marking the area outside the outcrop;
- optionally a DEM resampled onto exactly the orthomosaic's grid (same CRS,
  geotransform, width and height), for the RGB+DEM models;
- optionally traced fractures, for scoring: rasterized onto the orthomosaic grid
  as 1-pixel lines (for example `rio rasterize --like ORTHO.tif` from GeoJSON
  traces).

The ground sampling distance of each orthomosaic should be recorded next to its
results; FraXet spans about 0.5 mm to 3 cm per pixel.

```bash
DATA=<directory holding the orthomosaics and DEMs>
OUT=<directory for the results>
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=64G --time=02:00:00 \
       -J fraxtex-outcrop -o $OUT/fraxtex-%j.out --wrap="
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+fraxtex+v1.sqsh \
     --container-mounts=$DATA:/in:ro,$OUT:/out \
     bash -c 'set -e
fraxtex-predict --model unet-rgb --device cuda \
  /in/<ORTHO>.tif /out/<ORTHO>_unet-rgb.tif --mask /out/<ORTHO>_unet-rgb_mask.tif --threshold 0.1
fraxtex-predict --model sam2-rgb --device cuda \
  /in/<ORTHO>.tif /out/<ORTHO>_sam2-rgb.tif
fraxtex-predict --model unet-rgbdem --device cuda \
  --dem /in/<DEM>.tif --dem-scaling <mode chosen in step 1> \
  /in/<ORTHO>.tif /out/<ORTHO>_unet-rgbdem.tif'"
```

Each run prints the model's contract, the execution provider, the tile grid and
the counts of tiles run, tiles skipped and nodata pixels; the output GeoTIFFs
carry the same information as tags. Probability maps are for review against the
imagery, and a threshold chosen on one outcrop is not a calibration for another.
