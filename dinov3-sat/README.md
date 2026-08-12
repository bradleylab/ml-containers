# dinov3-sat

DINOv3 **SAT-493M** ViT-L/16 as a frozen dense-feature extractor for RGB
aerial and satellite orthoimagery.

Pull: `ghcr.io/bradleylab/dinov3-sat:v1`

## What this is

DINOv3 (Siméoni, Vo, Oquab et al., 2025) is a self-supervised vision
transformer. This image ships the **satellite-pretrained** variant — trained on
493 million 512×512 Maxar RGB ortho-rectified tiles at 0.6 m ground sampling —
not the web-image weights. For nadir RGB orthoimagery that is a far closer prior
than general-purpose DINOv2/DINOv3 checkpoints or the multispectral satellite
foundation models (`clay`, `croma`, `prithvi-eo`), which expect bands this
imagery does not have.

It is an **encoder only**. There is no task head: it emits dense patch tokens
for a downstream head to consume, in the same role as `remoteclip` (global
embeddings) or `croma` (radar-optical tokens). If you want a trained detector,
that is a separate image built `FROM` this one.

| | |
|---|---|
| Architecture | ViT-L/16, 1024-D |
| Weights | `timm/vit_large_patch16_dinov3.sat493m`, baked at build |
| Pretraining | SAT-493M — 493M Maxar RGB ortho tiles, 0.6 m GSD |
| Input | RGB, square, any multiple of 16 (position embedding interpolates) |
| Output | `(grid, grid, 1024)` patch features, `grid = size / 16` |
| Runtime | CUDA 12.1 / PyTorch 2.5.1 (H100-tested); runs on CPU, slowly |

## Three things that bite

**1. Normalization is not ImageNet.** The SAT weights use
`mean = [0.430, 0.411, 0.296]`, `std = [0.213, 0.156, 0.143]`. Feeding ImageNet
statistics raises no error — it just returns worse features. The values were
corrected on the model card after release, so resolve them from timm's
`pretrained_cfg` rather than copying them anywhere. The build's smoke test
asserts them, so a checkpoint swap that changes them fails the build.

**2. There are 5 prefix tokens.** DINOv3 prepends a CLS token and 4 register
tokens, so a 512 px input returns **1029** tokens, not 32×32 = 1024. Reshaping
the raw sequence into a spatial grid silently scrambles it. `extract_features.py`
strips `model.num_prefix_tokens` and asserts the remaining count matches the
expected grid.

**3. Pretraining GSD may not match yours.** Patches covered ~9.6 m of ground
during pretraining (16 px × 0.6 m). On finer imagery the same patch covers much
less — on 0.152 m aerial ortho, 2.44 m — so objects are presented at a very
different scale from what the encoder saw. There is no input size that satisfies
both a fine patch grid and the pretraining scale; pick by measurement on your own
task rather than by assuming. Consider a feature upsampler (FeatUp, AnyUp) if
your targets are only a few patches across.

## Usage

```bash
docker run --rm --gpus all \
  -v "$PWD/chips:/work/chips:ro" \
  -v "$PWD/features:/work/features" \
  ghcr.io/bradleylab/dinov3-sat:v1 \
  python /opt/dinov3-sat/extract_features.py \
    --input /work/chips \
    --out /work/features \
    --input-size 1024 \
    --batch-size 8 \
    --fp16
```

One `.npz` per image holding a `(grid, grid, 1024)` array, plus
`_features_meta.json` recording the checkpoint, input size, normalization and
library versions actually used — keep it with the features, since a head trained
on one configuration cannot be applied to another.

Feature volume is worth arithmetic before you cache: at 1024 px input that is
64×64×1024 in fp16 ≈ **8.4 MB per image**. Caching a few thousand is fine;
caching tens of thousands is hundreds of gigabytes — stream those instead.

### Compute2 (enroot)

```bash
enroot import 'docker://ghcr.io#bradleylab/dinov3-sat:v1'
```

Weights are baked in, so compute nodes need no outbound network.

## Licence — read before redistributing

The weights are governed by the **DINOv3 License**, shipped inside the image at
`/opt/licenses/LICENSE.dinov3.md` and alongside this README. It is **not** an
open-source licence. Three obligations:

1. **Redistribution requires shipping the Agreement.** Meta permits
   redistribution and derivative works, but any onward distribution must carry a
   copy of the licence. This image satisfies that by including it; anything built
   `FROM` it inherits the file — do not delete it.
2. **Publications must acknowledge DINO Materials.** Any paper, abstract or
   poster reporting results obtained with this model has to say so.
3. **Trade-control and no-military/ITAR terms apply.**

You own any head you train on top; Meta retains the backbone.

If you would rather not redistribute the weights at all, an alternative is to
build without the bake step and fetch at runtime into `HF_HOME` (the pattern
`remoteclip` uses) — the timm mirror is ungated, unlike `facebook/dinov3-*`,
which requires a manual access request.

## Verification

The build's smoke test runs offline against the baked cache and asserts
1024 features, patch size 16, and the exact SAT normalization values.
Loading, token geometry and the feature grid were verified against real
0.152 m aerial chips before the recipe was written (timm 1.0.28).
