# crater-detection

Lunar crater detection and crater-based absolute position fixing.
Doppenberg (2021), TU Delft. MIT.

## What this is

A Mask R-CNN variant that detects crater rims in lunar orbital imagery, plus a
matching stage that identifies detected craters against a catalog — which is
what turns a detection into a position fix.

## Why this and not LunarFM

LunarFM is the stronger lunar model and we have access to it. But it is
**PolyForm Strict 1.0.0**, which forbids redistribution *and* derivative works,
so it cannot be baked into a GHCR image at all. This is the only lunar model in
the catalog that can be containerized normally.

They also do different jobs: LunarFM produces surface embeddings, this produces
crater instances.

## Two upstream traps, both handled here

**The checkpoints are Git LFS pointers.** `blobs/CraterRCNN.pth` is 134 bytes in
a plain clone — a text pointer, against a real object of 221 MB. A `git clone`
without LFS bakes the pointer, and the image then fails at load with an
unpickling error that names nothing useful. This build fetches the LFS object
over HTTPS and **hard-fails anything under 1 MB**, so a pointer cannot ship.

**`requirements.txt` cannot install as written.** It contains

```
git+git://github.com/SurRenderSoftware/surrender_client_API@master
```

and GitHub disabled the unauthenticated `git://` protocol in 2022, so that line
fails regardless of whether the repo exists. `surrender` is a rendering client
for synthesising training data, not for inference. Rather than assume, the
dependencies are installed explicitly and the build smoke test imports the
detection path — if `surrender` were genuinely needed there, the build fails
instead of shipping a broken image.

## The synthetic-data path does not work on modern Python

`src/common/surrender.py` uses `from collections import Iterable`, removed in
Python 3.10, and depends on the package whose `git://` install line is already
broken. Importing the `src.detection` **package** pulls it in via
`__init__.py` -> `.training`.

So use the module, not the package:

```python
from src.detection.model import CraterDetector   # works
import src.detection                             # ImportError on py3.10+
```

Inference does not need any of it. Training-data synthesis with SurRender would
need both that dependency and an older Python, and is out of scope here.

## Maintenance status

Upstream has had no commits since 2025-01, so the commit is pinned
(`1365817fb62247378bb04085d41a2216a666ac25`). Treat it as archival: it works,
but nobody is fixing it.

## Verification

The build smoke test loads the real checkpoint and asserts it is a populated
state dict of more than 50 entries — which is also what proves the pointer
guard worked and the `surrender` omission is safe.

**Not yet executed on Compute2.**
