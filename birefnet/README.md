# birefnet

BiRefNet — high-resolution foreground segmentation.
Zheng et al. (2024), [BiRefNet](https://arxiv.org/abs/2401.03407). MIT.

## What this is

One high-quality foreground mask per image, at full resolution, with clean
edges on thin structures — hair, twigs, antennae, grain boundaries — that
coarser segmenters smear.

Unprompted: it returns the salient foreground without being told what to look
for. For "cut this specimen out of its background" that needs no interaction.
For "segment this particular thing" use `sam2` instead.

## Read this before bumping the revision

This checkpoint is `custom_code`. Loading it **executes `birefnet.py` and
`BiRefNet_config.py` fetched from the model repository** — pinning the
transformers version does not pin that code.

So the revision is pinned to an exact commit:

```
e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4
```

Bumping it means reading the diff of those two files first. This is executable
content arriving from a third-party repo, not data.

## kornia is pinned, deliberately

`kornia==0.8.2`, not 0.8.3. Version 0.8.3 runs `torch.jit.script` at module
scope and segfaults against torch 2.5.x — the failure that broke `prithvi-eo`
and `terramind` in the 2026-08 build-rot sweep, and which took three attempts
to diagnose because a segfault names no import. Do not float it.

## Usage

```bash
docker run --rm --gpus all -v "$PWD":/work ghcr.io/bradleylab/birefnet:v1 python -c "
import torch, numpy as np
from PIL import Image
from transformers import AutoModelForImageSegmentation
m = AutoModelForImageSegmentation.from_pretrained(
    'ZhengPeng7/BiRefNet', trust_remote_code=True,
    revision='e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4').eval()
img = Image.open('/work/photo.jpg').convert('RGB').resize((1024, 1024))
x = torch.from_numpy(np.asarray(img)).permute(2,0,1)[None].float()/255
with torch.inference_mode(): out = m(x)
mask = (out[-1] if isinstance(out, (list, tuple)) else out)
mask = (mask[-1] if isinstance(mask, (list, tuple)) else mask).sigmoid()
print(mask.shape)
"
```

## Verification

Build smoke test runs offline on CPU: a real forward pass asserting the mask
comes back single-channel at input resolution, which is what any compositing
step assumes.

**Not yet executed on Compute2.**
