# dinov3-sat

DINOv3 **SAT-493M** ViT-L/16 as a frozen dense-feature extractor for RGB
aerial and satellite orthoimagery.

Pull: `ghcr.io/bradleylab/dinov3-sat:v2`

From `v2` the image's ENTRYPOINT is the geospatial executor's contract program
(see [Running under the geospatial executor](#running-under-the-geospatial-executor)).
To run the extractor yourself, override the entrypoint as the Usage example
does. `v1` has no ENTRYPOINT; its default command prints the extractor's help.

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
  --entrypoint python \
  ghcr.io/bradleylab/dinov3-sat:v2 \
  /opt/dinov3-sat/extract_features.py \
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
enroot import 'docker://ghcr.io#bradleylab/dinov3-sat:v2'
```

Weights are baked in, so compute nodes need no outbound network.

Under `srun --container-image` (pyxis), the image's ENTRYPOINT is not run by
default (pyxis's `--container-entrypoint` turns it on), so `python
/opt/dinov3-sat/extract_features.py ...` as the job command runs the extractor
directly. `enroot start` behaves as Docker does and passes any command to the
ENTRYPOINT as its arguments, so for direct use there prefer pyxis or `v1`.

## Running under the geospatial executor

The ENTRYPOINT, `/usr/local/bin/dinov3-sat-contract`, implements version 1 of
the geospatial executor's entrypoint contract
(`docs/contracts/entrypoint-v1.md` in `fossettlab/geospatial-executor`). It
takes exactly three options, `--input-dir`, `--output-dir` and `--params-json`,
runs `extract_features.py` on every image staged under `<input-dir>/primary/`,
and writes the results and a `run.json` manifest under `<output-dir>`. It needs
no environment variables: it sets the weight cache (`/opt/hf-cache`) and
offline mode for the extractor itself.

**Input.** Every `.png`, `.jpg`, `.jpeg`, `.tif` or `.tiff` file under
`primary/`, searched recursively, as `extract_features.py` searches. The
encoder expects 8-bit RGB chips. Each image is resized (bicubic) to a square of
`input_size` pixels, so a non-square image is stretched and a whole
orthomosaic is shrunk to one chip: tile large imagery first. Outputs are named
after each input's file name without its extension, so two inputs that share
that name are refused rather than allowed to overwrite each other.

**Parameter.** One, sent as a string as contract v1 requires. Unknown
parameters are refused.

| Name | Choices | Default | Meaning |
|---|---|---|---|
| `input_size` | `256`, `512`, `1024` | `512` | Square input size in pixels. The feature grid is `input_size / 16` patches on a side, so larger sizes give a finer grid at about quadratic cost. 512 is the extractor's default and the pretraining tile size; 1024 is the Usage example; 256 is the size the build smoke test runs the model at. |

Everything else is the extractor's default: batch size 8, a float32 forward
pass, features stored as float16, and the GPU when one is visible.

**Outputs.**

| Path | Manifest role | Contents |
|---|---|---|
| `features/<name>.npz` | `features` | The `(grid, grid, 1024)` patch features for one image, prefix tokens removed, under the key `features` |
| `features/_features_meta.json` | `feature_metadata` | Checkpoint, input size, grid, prefix-token count, normalization, dtype and library versions |
| `previews/<name>.png` | `preview` | A false-color picture of one image's features, one pixel per patch |

The preview projects an image's patch features onto their first three
principal components, computed for that image alone, and scales each
component from its own minimum and maximum to 0-255 as red, green and blue.
Patches with similar features get similar colors, which shows at a glance
whether the encoder separates the surfaces in the scene. Because each image
has its own components and its own scaling, colors cannot be compared between
images.

`run.json` also carries warnings when images were resized or stretched, when
any were not stored as 8-bit RGB, and when no GPU was visible, so the encoder
ran on the CPU. The program exits 0 when every output and the manifest were
written, 2 when the input or parameters were refused (with the reason on
standard error, before the model is loaded), and 1 when the extractor failed
or did not write a file it should have.

A run by hand, outside the executor:

```bash
mkdir -p job/input/primary job/output
cp chips/*.png job/input/primary/
echo '{"input_size": "512"}' > job/params.json
docker run --rm --gpus all \
  -v "$PWD/job:/job" \
  ghcr.io/bradleylab/dinov3-sat:v2 \
  --input-dir /job/input \
  --output-dir /job/output \
  --params-json /job/params.json
```

Every run of this entrypoint uses DINO Materials, so a publication that
reports its features or previews must acknowledge the use of DINO Materials
(DINOv3 License, section 1.b.ii; see the next section).

The build runs the entrypoint end to end on two synthetic images under an
empty environment (`tests/contract_check.py`). The entrypoint's own logic is
also tested without the model:

```bash
uv run --python 3.11 --with numpy --with pillow --with pytest \
  pytest dinov3-sat/tests/test_contract.py
```

## License — read before redistributing

The weights are governed by the **DINOv3 License**, shipped inside the image at
`/opt/licenses/LICENSE.dinov3.md` and alongside this README. It is **not** an
open-source license. Three obligations:

1. **Redistribution requires shipping the Agreement.** Meta permits
   redistribution and derivative works, but any onward distribution must carry a
   copy of the license. This image satisfies that by including it; anything built
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
