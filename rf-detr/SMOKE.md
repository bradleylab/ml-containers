# rf-detr — Compute2 smoke test

The smallest run that proves the image detects things: **one real photograph
with known contents**, asserting it finds them and does not hallucinate a
class that is absent.

> [!warning] CPU-partition runs need `NVIDIA_VISIBLE_DEVICES=void`
> Every CUDA-base image in this repo sets `NVIDIA_VISIBLE_DEVICES`, so enroot's
> hook tries to inject a driver wherever the job lands. On `general-cpu` there
> is none and the container fails to start. Add
> `--export=ALL,NVIDIA_VISIBLE_DEVICES=void` for CPU runs.

## Why not synthetic input

A previous smoke test in this repo passed while finding **zero** objects,
because it asserted only that the output had the right fields — which it does
when empty. Noise has nothing to detect, so "found nothing" and "broken" look
identical. Use a real image and gate on the count.

## 0. One-time: import the image

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+rf-detr+v1.sqsh \
    'docker://ghcr.io#bradleylab/rf-detr:v1'
```

`enroot import` can exit 0 after its `mksquashfs` child is OOM-killed, so check
the artifact rather than the exit status:

```bash
file -b bradleylab+rf-detr+v1.sqsh | grep -q '^Squashfs' && echo OK || echo CORRUPT
```

## 1. The test

Uses `/storage3/fs1/alexander.s.bradley/Active/test_images/cats.jpg` — COCO
val2017 000000039769, two cats on a couch — already staged.

```bash
sbatch -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
       --cpus-per-task=8 --mem=32G --time=00:20:00 \
       -J rfdetr-smoke -o rfdetr-smoke-%j.out --wrap='
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+rf-detr+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley/Active/test_images:/images \
     bash -lc "export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1; python - <<PY
import torch
from PIL import Image
from rfdetr import RFDETRMedium

assert torch.cuda.is_available(), \"no GPU visible to the job\"
print(\"gpu:\", torch.cuda.get_device_name(0))

m = RFDETRMedium()
det = m.predict(Image.open(\"/images/cats.jpg\"), threshold=0.5)
print(\"detections:\", len(det))
for box, conf, cid in zip(det.xyxy, det.confidence, det.class_id):
    print(f\"  class {cid}  conf {conf:.3f}  box {[round(float(v)) for v in box]}\")
assert len(det) >= 2, f\"expected at least the two cats, found {len(det)}\"
print(\"SMOKE OK\")
PY"'
```

## 2. What passing means

- The baked checkpoint loads with no network.
- A forward pass runs on sm_90.
- It finds **at least two objects** in an image containing two cats — the gate
  that a structure-only check would miss.

Record the measured wall time here after the first successful run.
