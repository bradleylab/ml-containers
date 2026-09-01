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
import utonia
model = utonia.load(
    "/opt/weights/utonia.pth",
    custom_config=dict(enable_flash=False, enc_patch_size=[1024] * 5),
)
```

Leaving `enable_flash` at its default raises `AssertionError: Make sure
flash_attn is installed.` — a clear failure, not a silent one. `enc_patch_size`
belongs with it: upstream's own no-flash path sets both
(`demo/3_batch_forward.py`), because non-fused attention holds a larger
activation footprint at the same batch size.

## Do NOT call `from_pretrained()`

`PointTransformerV3` subclasses `PyTorchModelHubMixin`, so `from_pretrained()`
exists and is the obvious thing to reach for. **It is wrong here**, and it fails
in a way that looks like a bug in the weights:

```
AssertionError: Head dimension must be divisible by 3 for 3D RoPE, 16
```

`Pointcept/Utonia` publishes no `config.json` — only `.pth` files. So
`from_pretrained()` has nothing to configure from and silently falls back to the
constructor defaults, `enc_channels (48, 96, 192, 384, 512)` over
`enc_num_head (3, 6, 12, 24, 32)`. That is head dimension **16** at every stage,
and 3D RoPE requires a multiple of 3.

The real architecture lives inside the checkpoint, at `ckpt["config"]`: channels
`(54, 108, 216, 432, 576)` over the same head counts, i.e. head dimension **18**
throughout, with `in_channels=9` for `[coord, color, normal]`. `utonia.load()`
reads it; `custom_config` overlays that config rather than replacing it, which
is how `enable_flash` gets turned off without discarding the architecture.

Note also that `utonia.load()`'s Hugging Face branch cannot reach a baked cache:
its `repo_id` defaults to `Pointcept/utonia` (lowercase u) against a real repo
named `Pointcept/Utonia`, and it passes `local_dir=~/.cache/utonia/ckpt`, which
bypasses `HF_HOME`. Hence the fixed path — `load()` also accepts a plain file,
which resolves nothing and needs no network.

The class is `PointTransformerV3`, not `Utonia` — the repository is named for
the release, the class for the architecture it extends.

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

**Verified on Compute2, 2026-08-31, job 2950544, c2-gpu-005 (H100 80GB).**

The checkpoint's own config read back as `in_channels 9`, `enc_channels
(54, 108, 216, 432, 576)`, `enc_depths (3, 3, 3, 12, 3)`, `enc_num_head
(3, 6, 12, 24, 32)` — head dimension 18 at every stage. Model built to
137,253,744 parameters and a forward pass over a synthetic 20,000-point cloud
returned features of shape `(2194, 576)`, all finite.

The build smoke test asserts that architecture, plus the parameter count and the
absence of `flash_attn`. The previous version resolved the model *class* and
stopped — which passed on an image whose only documented calling path could not
construct a model at all. That defect survived to an H100 run weeks later, which
is why the build now constructs the real model from the staged checkpoint.
