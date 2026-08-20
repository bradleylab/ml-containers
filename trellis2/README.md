# trellis2

TRELLIS.2-4B — single image to textured 3D mesh. Microsoft (2025). MIT.

## Scope: this is generation, not measurement

The output is a **plausible** 3D asset, not a survey of a real object. Built for
the lab's XR work: assets to drop into a scene.

For measured geometry use `odm` or `splat-pipeline`, which reconstruct from many
views. **Do not substitute a TRELLIS mesh for a photogrammetric one in an
analysis** — the geometry it invents where the single view had no information is
indistinguishable, by eye, from geometry it observed.

Outputs meshes with PBR materials, GLB export, textures to 4096.

## It runs on H100 and nothing older

The five CUDA extensions — nvdiffrast, nvdiffrec, CuMesh, FlexGEMM, o-voxel —
are compiled with `TORCH_CUDA_ARCH_LIST=9.0`, i.e. sm_90 only.

That is a deliberate trade. Compiling the default architecture list builds every
kernel back to Pascal and is the difference between a CI build that finishes and
one that does not. Compute2's GPU fleet is entirely H100, so sm_90 covers it.
If this ever needs to run on an older card, the arch list is the one line to
change — and the build will take substantially longer.

## flash-attn comes from a prebuilt wheel

Upstream's `setup.sh` runs `pip install flash-attn==2.7.3`, which is source-only
on PyPI: hours of compilation and enough memory to kill a CI runner. This image
installs the matching prebuilt wheel instead — cu12 / torch 2.6 / cp311 /
cxx11abiFALSE, matching what the pytorch base image's own torch was built
against.

If that wheel ever stops matching, TRELLIS.2 accepts `ATTN_BACKEND` in
`{xformers, flash_attn, flash_attn_3, sdpa, naive}`. **`sdpa` uses torch's
built-in attention and needs no extra package** — that is the fallback, at some
cost in speed and memory.

## Why the smoke test checks extensions, not imports

A CUDA extension that installs with **no kernels still exits 0**. This repo has
already shipped that failure once — `natten` and `torch-harmonics` installed
kernel-less on GPU-less runners and produced green builds of degraded images.

So the build smoke test imports each compiled extension and asserts its native
module resolved, rather than trusting pip. It also prints the arch list, so the
sm_90 constraint is visible in the build log rather than buried here.

A real forward pass needs a GPU and lives in `SMOKE.md`.

## Usage

```bash
docker run --rm --gpus all -v "$PWD":/work ghcr.io/bradleylab/trellis2:v1 python -c "
from trellis2.pipelines import Trellis2ImageTo3DPipeline
from PIL import Image
p = Trellis2ImageTo3DPipeline.from_pretrained('microsoft/TRELLIS.2-4B')
p.cuda()
mesh = p.run(Image.open('/work/object.png'))[0]
"
```

Weights are baked; set `HF_HUB_OFFLINE=1` on compute nodes.

## Verification

**Not yet executed on Compute2.** The build proves the extensions carry kernels
and the pipeline class resolves; it does not prove a mesh comes out.
