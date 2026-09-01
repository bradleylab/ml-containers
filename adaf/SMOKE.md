# adaf — Compute2 smoke test

> [!warning] CPU-partition runs need `NVIDIA_VISIBLE_DEVICES=void`
> This image sets `NVIDIA_VISIBLE_DEVICES`, so enroot's hook tries to inject a
> driver wherever the job lands. On `general-cpu` there is none and the
> container fails to start. Add `--export=ALL,NVIDIA_VISIBLE_DEVICES=void`.

## What this proves, and what it does not

The build already loads a real checkpoint into a real model, so this file is
about the **pipeline**: DTM → SLRM → tiles → detections → vectors.

It uses a **synthetic DTM with a planted mound**. That is a positive control,
not a validation: a barrow genuinely is a circular rise of roughly the right
relief, so the barrow model firing on one is meaningful signal. But a plausible
response to a clean synthetic mound says nothing about recall in real forest
terrain with real noise.

**Detection quality is unproven for this lab.** The experiment that would settle
it is in the README: run the pretrained models over a real survey DTM at both
0.25 m and 0.5 m, and score recall against targets already identified by eye.
Until that is done, treat output as untested.

## 0. One-time: import the image

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+adaf+v1.sqsh \
    'docker://ghcr.io#bradleylab/adaf:v1'
```

## 1. The test script

```python
# smoke_adaf.py — synthetic DTM with a planted mound, through the full pipeline.
import numpy as np, rasterio, torch
from pathlib import Path
from rasterio.transform import from_origin
from adaf.adaf_inference import run_aitlas_segmentation

SEED, RES, SIZE = 42, 0.5, 1024
OUT = Path("/work/smoke"); OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(SEED)

# Gently sloping ground + correlated roughness + one 0.8 m Gaussian mound 12 m
# across, which is a plausible barrow. Relief is what SLRM keys on, so the
# slope must be there for the local-relief model to have anything to remove.
yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
dem = 0.004 * xx + 0.002 * yy + rng.normal(0, 0.05, (SIZE, SIZE)).astype(np.float32)
cy, cx, sigma = SIZE // 2, SIZE // 2, 12.0 / RES / 2.355
dem += 0.8 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))

dem_path = OUT / "synthetic_dtm.tif"
with rasterio.open(
    dem_path, "w", driver="GTiff", height=SIZE, width=SIZE, count=1,
    dtype="float32", crs="EPSG:3857",
    transform=from_origin(0, SIZE * RES, RES, RES),
) as dst:
    dst.write(dem, 1)
print("wrote", dem_path, "| cuda:", torch.cuda.is_available())

# SLRM at 512 px is what the weights were trained on — see README.
from adaf.adaf_vis import tiled_processing
vis = tiled_processing(
    input_vrt_path=str(dem_path), ll_dir=str(OUT), save_dir=str(OUT),
    tile_size=512, nr_processes=1,
)
print("visualisation:", vis)

W = "/opt/adaf-weights"
seg = run_aitlas_segmentation(
    ["custom"], str(OUT),
    custom_model=f"{W}/barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
)
print("segmentation output dirs:", seg)
assert seg, "segmentation returned nothing"
print("SMOKE OK — pipeline ran end to end")
```

## 2. Submit

```bash
srun -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
     --mem=32G --cpus-per-task=8 --time=00:30:00 \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+adaf+v1.sqsh \
     --container-mounts=$PWD:/work \
     --container-workdir=/work \
     python /work/smoke_adaf.py
```

## Known unknowns

- `adaf_vis.tiled_processing`'s exact signature is taken from upstream's
  notebooks and **has not been executed here**. If it has drifted, fix the call
  rather than assuming the image is broken.
- The synthetic DTM carries no CRS meaning — `EPSG:3857` is a placeholder so
  rasterio and GDAL are satisfied. Real runs need a real projected CRS.
- Nothing here checks the object-detection path. Add
  `run_aitlas_object_detection(["custom"], ..., custom_model=f"{W}/OD_barrow.tar")`
  once the segmentation path is confirmed.
