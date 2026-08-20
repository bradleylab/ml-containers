# dinov3

DINOv3 ViT-L/16 pretrained on LVD-1689M — 1.689 billion curated web images —
as a frozen dense-feature extractor. Encoder only, no task head.

Siméoni, Vo, Oquab et al. (2025), [DINOv3](https://arxiv.org/abs/2508.10104).

## What this is

A general-purpose visual backbone. It turns an image into a grid of 1024-D
patch tokens that a downstream head consumes — a linear probe, a segmentation
decoder, a nearest-neighbour index. It is not trained for any task, which is
the point: the features transfer, and a head on top needs far fewer labels
than training from scratch.

This is the **general** counterpart to `dinov3-sat` in this repo. Both are
ViT-L/16 DINOv3 with 1024-D tokens; they differ only in what they were
pretrained on, and that difference is the whole reason both exist.

| Use | Image |
|---|---|
| Nadir RGB aerial / satellite orthoimagery | `dinov3-sat` |
| Everything else — field photographs, thin sections, microscopy, close-range and oblique drone, lab and specimen imagery | `dinov3` (this one) |

## Three things that bite

**The normalization constants differ between the two images, and using the
wrong ones fails silently.** These weights want ImageNet statistics — mean
`(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`. The SAT weights want
`(0.430, 0.411, 0.296)` / `(0.213, 0.156, 0.143)`. Feed either model the
other's constants and nothing raises; the embeddings just get worse. Resolve
them from `timm.data.resolve_model_data_config(model)` rather than hardcoding,
and never copy them between the two images. The build smoke test asserts the
expected values, so a checkpoint swap that changes them fails the build.

**Input size is variable, and that is a decision, not a default.** The
position embedding interpolates, so any multiple of 16 works. Larger input
means more tokens and finer spatial detail at proportionally more compute.
The pretrained configuration is 256×256; the minimum is 128×128.

**Not every DINOv3 checkpoint on the timm mirror carries the same licence.**
The `*_qkvb.eupe_lvd1689m` variants are released under
`fair-noncommercial-research-license`, which is stricter than the DINOv3
Licence this image ships under. If you swap the weights, re-check the licence
of what you swapped to.

## Usage

```bash
docker run --rm --gpus all \
  -v "$PWD":/work \
  ghcr.io/bradleylab/dinov3:v1 \
  python /opt/dinov3/extract_features.py \
    --images /work/photos \
    --out /work/features.npz \
    --size 256
```

### Compute2 (enroot)

```bash
srun -A compute2-alexander.s.bradley -p general-gpu \
     --gpus=1 --cpus-per-task=4 --mem=32G --time=00:30:00 \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+dinov3+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley:/storage3/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; export HF_HUB_OFFLINE=1; \
       python /opt/dinov3/extract_features.py --images /path/to/images --out /path/to/features.npz'
```

The weights are baked into the image, so `HF_HUB_OFFLINE=1` is safe and the
job needs no outbound network.

## Licence — read before redistributing

The DINOv3 Licence is **not open source**. The full Agreement is shipped
inside the image at `/opt/licenses/LICENSE.dinov3.md` and mirrored in this
directory.

Two obligations that matter here:

1. **Redistribution requires shipping the Agreement.** Anything built `FROM`
   this image inherits that obligation — keep `/opt/licenses/` intact.
2. **Publications using these features must acknowledge DINO Materials.**
   This is a condition of use, not a courtesy.

## Verification

The build runs an offline smoke test that loads the baked weights and asserts
the feature width (1024), the patch size (16×16), and both normalization
constants, then runs a forward pass. A build that reaches GHCR has passed all
of it.

Not yet executed on Compute2 — this image has no `SMOKE.md` run recorded.
