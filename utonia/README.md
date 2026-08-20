# utonia

Self-supervised point-cloud encoder, the generation after PTv3, Sonata and
Concerto. Pointcept, ICML 2026. Apache-2.0 code, **CC-BY-NC-4.0 weights**.

## Why it is here alongside three other encoders

`point-transformer-v3`, `sonata` and `concerto` are all in the catalog and all
one generation behind this. They stay: they are already imported on Compute2,
already probed, and referenced by existing work.

Utonia's distinguishing property is that it is the first in this line
**pretrained across remote-sensing point clouds**, not indoor scans alone —
which is the domain most of this lab's lidar actually comes from.

## It has no flash-attention. You must pass `enable_flash=False`

Upstream installs flash-attn from git. That package ships source-only, takes
hours to compile, and exhausts memory on a CI runner — which is why upstream
themselves point at third-party Docker images rather than shipping one.

Utonia does not require it. `import flash_attn` is wrapped in try/except
(`utonia/model.py:37-39`) and the attention module takes `enable_flash` with
non-fused branches. Flash attention is a memory-layout optimisation, not a
different operation, so the non-fused path computes the same thing — slower,
and with a larger activation footprint at the same batch size.

```python
model = Utonia(..., enable_flash=False)   # required in this image
```

Leaving the default `True` raises `AssertionError: Make sure flash_attn is
installed.` — a clear failure, not a silent one.

If throughput matters more than build simplicity later, the fix is a **prebuilt
flash-attn wheel** matching torch, CUDA and Python exactly. Not a source build
in CI.

## Licence: the weights are the binding half

Code is Apache-2.0. **Weights are CC-BY-NC-4.0.** As shipped this image is
non-commercial: fine for internal research, blocks a public service or an
industry-funded derivative. The image label records the combination rather than
the code licence alone, so a licence audit sees the real constraint.

## Pins

`torch-scatter` and `spconv-cu124` publish wheels per **exact** torch+CUDA
pair, so this image is on CUDA 12.4 rather than the 12.1 used elsewhere in the
repo, matching upstream's `environment.yml` (cuda 12.4 / pytorch 2.5). Floating
either is how the image stops building.

## Verification

The build smoke test asserts **both halves** of the no-flash contract: that
`flash_attn` really is absent, and that the package imports and the model class
resolves without it. An import-only check would pass on an image whose only
usable path had been removed.

**Not yet executed on Compute2.**
