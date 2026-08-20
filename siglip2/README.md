# siglip2

SigLIP 2 `so400m-patch14-384` — general image-text scoring.
Tschannen et al. (2025), [SigLIP 2](https://arxiv.org/abs/2502.14786). Apache-2.0.

## What this is

Encodes images and text into a shared space, so an image can be scored against
arbitrary text with no training. Use it to triage a photo archive, flag frames
containing some feature, or make a collection text-searchable.

The catalog already had `remoteclip` and `dofa-clip`; both are remote-sensing
specific and score poorly on anything that is not overhead imagery. This is the
general one.

## What it does not do

It returns a **similarity score**, not a location or a mask.

| Question | Image |
|---|---|
| does this image contain X? | `siglip2` |
| where in the image is X? | `grounding-dino` |
| give me a mask of X | `sam2` |

## Usage

```bash
docker run --rm --gpus all -v "$PWD":/work ghcr.io/bradleylab/siglip2:v1 python -c "
from transformers import AutoModel, AutoProcessor
from PIL import Image
mid = 'google/siglip2-so400m-patch14-384'
m, p = AutoModel.from_pretrained(mid).eval(), AutoProcessor.from_pretrained(mid)
img = Image.open('/work/photo.jpg')
x = p(text=['an outcrop', 'a thin section'], images=img, padding='max_length', return_tensors='pt')
print(m(**x).logits_per_image.softmax(-1))
"
```

Weights are baked; set `HF_HUB_OFFLINE=1` on compute nodes.

## Verification

Build smoke test runs offline on CPU: a real forward pass scoring noise against
two prompts, asserting one logit per prompt and finite values. A transformers
bump that breaks the model fails the build.

**Not yet executed on Compute2.**
