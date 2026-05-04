# dofa

[DOFA](https://arxiv.org/abs/2403.15356) (Dynamic One-For-All;
Xiong et al. 2024) — multispectral / SAR / optical / hyperspectral
foundation model with a wavelength-conditioning hypernetwork. A
single ViT backbone adaptable to arbitrary spectral configurations.
Trained with masked image modelling on SatlasPretrain +
Five-Billion-Pixels + HySpecNet-11k.

This is the **base model**, not the CLIP variant. For text-prompt
zero-shot retrieval, see `bradleylab/dofa-clip` (separate container,
CC-BY-NC-4.0 license).

## Image tag

`ghcr.io/bradleylab/dofa:v1` (also `:latest`, `:torch2.5-cpu`).

Multi-arch: `linux/amd64` + `linux/arm64`.

## Stack

- Base: `python:3.11-slim`
- PyTorch 2.5.1 + torchvision 0.20.1 (CPU wheels)
- `torchgeo>=0.6` ([torchgeo](https://github.com/microsoft/torchgeo))
- `timm>=1.0`, `huggingface_hub>=0.25`, Pillow

## Variants

| variant | params | embedding dim | weights | bundled |
|---|---|---|---|---|
| `--variant base` (default) | ~111M | 768 | `dofa_base_patch16_224-a0275954.pth` (445 MB) | yes (baked at build) |
| `--variant large` | ~336M | 1024 | `dofa_large_patch16_224-0ff904d3.pth` (1.35 GB) | no — fetched lazily on first use |

Large weights download to `$TORCH_HOME=/opt/torch-cache` on first
call. Bind-mount that directory if you want to persist the cache
across runs.

## License

Weights are **CC-BY-4.0** per the `torchgeo/dofa` HF model card.
Free to redistribute and use commercially with attribution to
Xiong et al. 2024.

## Wavelength conditioning

DOFA's hypernetwork generates input projection weights from the
per-band wavelength. **You must pass the wavelengths explicitly**;
torchgeo does not infer them from sensor metadata. Wavelengths are
in **micrometers**.

The inference script provides convenience flags for the common
sensor configurations:

| flag | bands |
|---|---|
| `--sentinel2-12band` | S2 L1C without B10 (cirrus): B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 |
| `--sentinel2-10band` | S2 land-only: B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12 |
| `--sentinel1` | S1 VV+VH (both at C-band 5.405 µm) |
| `--naip-rgb` | NAIP RGB (R, G, B) |
| `--wavelengths 0.49 0.56 0.665 ...` | arbitrary user-supplied list |

For other sensors (Landsat 8/9, Gaofen, hyperspectral), pass
`--wavelengths` explicitly with the per-band central wavelengths
in micrometers.

## Inference

```bash
docker run --rm \
  -v "$PWD:/work" \
  ghcr.io/bradleylab/dofa:v1 \
  python /opt/scripts/dofa_embed.py \
    --image /work/s2_tile.npy \
    --sentinel2-12band \
    --variant base \
    --out /work/embed.npz
```

The output `.npz` contains:
- `embedding` — float32 array of shape `(1, 768)` for Base or `(1, 1024)` for Large
- `wavelengths` — the wavelength list used (for reproducibility)
- `variant` — string, "base" or "large"

## Inputs

- `.npy` or `.pt` file containing a `(C, H, W)` float32 tensor.
- `C` must equal the number of wavelengths supplied.
- The script will resize to 224×224 with bilinear interpolation if
  the input is a different size; pass pre-cropped tiles when
  possible to avoid resampling artefacts.

## What the embedding is good for

- **Downstream classification / segmentation / regression heads.**
  Train a small head on top of the 768-D / 1024-D vectors for
  task-specific outputs (deforestation, crop-type, water extent,
  surface-mineral mapping). Standard finetuning protocol — fix the
  backbone, train the head with LR ~1e-3 first, then unfreeze for
  end-to-end finetuning at ~1e-4.
- **Image-similarity / retrieval.** Cosine over embeddings finds
  visually-related tiles across sensors and across spectral
  configurations.
- **Change detection by embedding distance.** L2 or cosine between
  pre/post embeddings flags scenes that have changed substantially.

DOFA embeddings are NOT directly comparable across model variants
(Base vs Large live in different latent spaces) and shift slightly
with different wavelength configurations even on the same backbone
— pin a `--variant` and a wavelength list per project.

## Run on Compute2

CPU is fine for one-shot embedding extraction (~1-3 s/image at 224x224
for Base). For batched embedding across many tiles, the GPU variant
(future work) on Compute2 H100 will give 50-100× throughput. CPU
job-array template:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=4 --mem=8G --time=04:00:00 \
       --array=0-99 \
       --wrap='srun --container-image=$IMG \
         --container-mounts=/scratch2/fs1/$USER:/scratch2/fs1/$USER \
         --container-workdir=/work \
         bash -lc "export PYTHONNOUSERSITE=1; \
                   python /opt/scripts/dofa_embed.py \
                     --image /scratch2/fs1/$USER/tiles/${SLURM_ARRAY_TASK_ID}.npy \
                     --sentinel2-12band \
                     --out /scratch2/fs1/$USER/embed/${SLURM_ARRAY_TASK_ID}.npz"'
```

## Caveats

- **Embedding-only.** No classification / segmentation / detection
  head is shipped. Train your own.
- **Wavelength normalisation.** The hypernetwork was trained with
  wavelengths in a specific scaling; passing exotic values (e.g.
  thermal IR > 10 µm) is out-of-distribution.
- **Spatial resolution.** Default is 224×224. Larger contexts
  require positional-encoding interpolation (not yet exposed in
  the inference script).
- **No CLIP / text-prompt support.** This is the base DOFA. The
  text-aligned variant lives in `bradleylab/dofa-clip` (CC-BY-NC).
