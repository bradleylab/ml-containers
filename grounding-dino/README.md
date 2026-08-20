# grounding-dino

Grounding DINO base — open-vocabulary object detection from free text.
Liu et al. (2023), [Grounding DINO](https://arxiv.org/abs/2303.05499). Apache-2.0.

## What this is

Detects objects you name in a prompt, with no training and no fixed class list.
Pass `"a boulder. a fracture. a tree."` and get boxes back.

That is the difference from `deepforest`, which detects the one class it was
trained for, and from `sam2`, which segments what you point at but cannot find
it from a description. The natural pairing is **grounding-dino to locate, sam2
to mask**.

## The prompt format matters more than it looks

Upstream expects **lower-case phrases separated by periods, with a trailing
period**:

```
"a boulder. a fracture. a tree."     correct
"Boulder, fracture, tree"            silently worse
```

Capitalisation and comma separators do not raise an error. They just reduce
recall, which is the kind of failure you find months later in the results
rather than at the time.

## Usage

```bash
docker run --rm --gpus all -v "$PWD":/work ghcr.io/bradleylab/grounding-dino:v1 python -c "
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
mid = 'IDEA-Research/grounding-dino-base'
p, m = AutoProcessor.from_pretrained(mid), AutoModelForZeroShotObjectDetection.from_pretrained(mid).eval()
img = Image.open('/work/photo.jpg')
x = p(images=img, text='a boulder. a tree.', return_tensors='pt')
with torch.inference_mode(): out = m(**x)
print(p.post_process_grounded_object_detection(out, x.input_ids, threshold=0.3, target_sizes=[img.size[::-1]])[0])
"
```

Weights are baked; set `HF_HUB_OFFLINE=1` on compute nodes.

## Verification

Build smoke test runs offline on CPU with a well-formed prompt and asserts the
post-processor returns the keys downstream code indexes into, so a transformers
bump that renames them fails the build.

**Not yet executed on Compute2.**
