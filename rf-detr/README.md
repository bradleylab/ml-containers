# rf-detr

Trainable object detector. Roboflow RF-DETR, ICLR 2026. **Apache-2.0.**

## What it's for

Detection you can **fine-tune on your own annotations**. That was the gap:

| Need | Use |
|---|---|
| find tree crowns | `deepforest` — trained for that one class |
| find anything I can name, no training | `grounding-dino` — zero-shot, no labels needed |
| **I annotated 500 frames, give me a detector** | **`rf-detr`** |
| segment what I point at | `sam2` |
| segment everything I name | `sam3` |

Real-time, DINOv2 backbone, state of the art on COCO. Also does instance
segmentation and, in preview, keypoints.

## Why this and not YOLO

Ultralytics YOLO is the obvious pick and is **AGPL-3.0**. This repo publishes
to a **public** registry, so an AGPL model makes the image AGPL-encumbered
along with anything built from it — and AGPL's network clause reaches any
service that ever serves its predictions. The lab already runs public web
services, so that isn't hypothetical.

RF-DETR does the same job under Apache-2.0. Deliberate substitution, not an
oversight.

## The licence split inside RF-DETR

Upstream ships two packages and **only one is Apache-2.0**:

| Package | Licence | Here? |
|---|---|---|
| `rfdetr` — includes RF-DETR-Medium | **Apache-2.0** | installed |
| `rfdetr_plus` — RF-DETR-XL, -2XL | **PML 1.0** | **not installed** |

Reaching for the bigger model would put PML-licensed weights inside an image
labelled Apache-2.0. The build **asserts `rfdetr_plus` is absent**, so that
cannot happen quietly later. Same trap as `depth-anything-3`, where checkpoints
in one family don't share a licence.

## Usage

```bash
docker run --rm --gpus all -v "$PWD":/work ghcr.io/bradleylab/rf-detr:v1 python -c "
from rfdetr import RFDETRMedium
from PIL import Image
m = RFDETRMedium()
det = m.predict(Image.open('/work/photo.jpg'), threshold=0.5)
print(det.xyxy, det.confidence)
"
```

Fine-tuning needs the training extras, which are not in this image — it is
built for inference. Add `rfdetr[train]` in a derived image if you want to
train inside the container.

## Verification

The build smoke test loads the baked checkpoint and runs a forward pass
offline, and asserts the Apache/PML boundary holds. It does **not** prove the
detections are meaningful — noise has nothing to find. That needs a real
photograph on a GPU; see `SMOKE.md`.

**Not yet executed on Compute2.**
