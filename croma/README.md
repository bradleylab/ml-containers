# croma

[CROMA](https://arxiv.org/abs/2311.00566) (Contrastive Radar-Optical
Masked Autoencoders; Fuller, Millard & Green, NeurIPS 2023) — a
Sentinel-1/Sentinel-2-native foundation model pretrained with combined
contrastive + masked-autoencoding objectives. It has a SAR encoder, an
optical encoder, and a cross-modal **joint** encoder; each emits
per-patch token encodings plus a global-average-pooled (GAP) embedding.

CROMA is **Sentinel-1/Sentinel-2-specialised**: SAR input is exactly
2 channels (VV, VH) and optical is exactly 12 bands (Sentinel-2 L1C with
B10/cirrus removed). For arbitrary-sensor / wavelength-conditioned
embeddings see `bradleylab/dofa`; for the RGB CLIP-style triage used in
`nc-helene-imagery-seg` see `bradleylab/remoteclip`.

## Image tag

`ghcr.io/bradleylab/croma:v1` (also `:latest`, `:torch2.5-cpu`).

Multi-arch: `linux/amd64` + `linux/arm64`.

## Stack

- Base: `python:3.11-slim`
- PyTorch 2.5.1 (CPU wheels) — CROMA imports only `torch` + `einops`
  (no torchvision, no compiled extensions)
- `huggingface_hub>=0.25`, numpy
- Upstream model code (`use_croma.py`) pinned to commit
  `59505a6` of [`antofuller/CROMA`](https://github.com/antofuller/CROMA)

## Variants

| variant | backbone | embedding dim | weights | bundled |
|---|---|---|---|---|
| `--variant base` (default) | ViT-B | 768 | `CROMA_base.pt` | yes (baked at build) |
| `--variant large` | ViT-L | 1024 | `CROMA_large.pt` | no — fetched lazily on first use |

Large weights download to `$HF_HOME=/opt/hf-cache` on first call.
Bind-mount that directory to persist the cache across runs.

## License

Both the code ([`antofuller/CROMA`](https://github.com/antofuller/CROMA))
and the weights ([HF `antofuller/CROMA`](https://huggingface.co/antofuller/CROMA))
are **MIT** — free to redistribute and use commercially with attribution
to Fuller, Millard & Green (2023).

## Inputs

CROMA's channel counts are fixed by the pretrained architecture:

| modality | tensor | channels |
|---|---|---|
| SAR (Sentinel-1) | `--sar` | 2 — VV, VH |
| optical (Sentinel-2) | `--optical` | 12 — S2 L1C **without B10 (cirrus)**: B1, B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12 |

- `.npy` or `.pt` file containing a `(C, H, W)` float tensor.
- Spatial size must be a multiple of 8 (the patch stride); the
  pretrained models use 120×120. Inputs of other sizes are bilinearly
  resized to `--image-resolution` (default 120) — pass pre-cropped
  120×120 chips when possible to avoid resampling artefacts.
- `--modality both` needs both `--sar` and `--optical`; `SAR` / `optical`
  need only the corresponding input.

## Inference

```bash
# Joint S1+S2 embedding
docker run --rm -v "$PWD:/work" \
  ghcr.io/bradleylab/croma:v1 \
  python /opt/scripts/croma_embed.py \
    --sar /work/s1.npy \
    --optical /work/s2_12band.npy \
    --variant base \
    --out /work/embed.npz

# SAR-only embedding
docker run --rm -v "$PWD:/work" \
  ghcr.io/bradleylab/croma:v1 \
  python /opt/scripts/croma_embed.py \
    --sar /work/s1.npy --modality SAR \
    --out /work/s1_embed.npz
```

The output `.npz` contains, depending on `--modality`:
- `SAR_GAP`, `optical_GAP`, `joint_GAP` — float32 `(1, D)` embedding
  vectors (`D` = 768 Base / 1024 Large) for whichever modalities were run
- `*_encodings` — `(1, N, D)` per-patch tokens (`N = (res/8)²`, e.g. 225
  at 120 px) only if `--save-encodings` is passed
- `variant`, `modality`, `image_resolution` — for reproducibility

## What the embedding is good for

- **Downstream disturbance / change heads.** Train a small head on the
  GAP vectors (or the patch tokens for dense prediction). The intended
  use in this lab is `nc-helene-sar` Stage 3 — CROMA frozen embeddings as
  a fusion feature for predicting LiDAR-DoD-measured geomorphic change
  (standard protocol: freeze the backbone, train the head, optionally
  fine-tune later).
- **Radar-optical fusion.** The `joint_GAP` embedding is the cross-modal
  representation — the reason to prefer CROMA over a single-sensor
  backbone when both S1 and S2 are available.
- **Change detection by embedding distance.** L2 / cosine between pre/post
  embeddings flags scenes that changed.

CROMA embeddings are NOT comparable across variants (Base vs Large are
different latent spaces) — pin a `--variant` per project.

## Run on Compute2

CPU is fine for one-shot embedding extraction. For batched embedding
across many chips, a GPU variant (future work) on Compute2 H100 will give
large throughput gains. CPU job-array template:

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=4 --mem=8G --time=04:00:00 \
       --array=0-99 \
       --wrap='srun --container-image=$IMG \
         --container-mounts=/scratch2/fs1/$USER:/scratch2/fs1/$USER \
         --container-workdir=/work \
         bash -lc "export PYTHONNOUSERSITE=1; \
                   python /opt/scripts/croma_embed.py \
                     --sar /scratch2/fs1/$USER/s1/${SLURM_ARRAY_TASK_ID}.npy \
                     --optical /scratch2/fs1/$USER/s2/${SLURM_ARRAY_TASK_ID}.npy \
                     --out /scratch2/fs1/$USER/embed/${SLURM_ARRAY_TASK_ID}.npz"'
```

`enroot import` to a `.sqsh` on scratch2 first, per
`~/.claude/rules/research-infrastructure.md`.

## Caveats

- **Embedding-only.** No classification / segmentation / regression head
  is shipped. Train your own (the `nc-helene-sar` Track B / Stage 3 job).
- **Fixed channel layout.** SAR must be 2-ch (VV, VH) and optical 12-band
  (cirrus removed) — the encoders have no wavelength-conditioning, unlike
  DOFA. Mismatched band counts error out.
- **Normalisation.** Feed the inputs scaled as CROMA expects (the upstream
  `use_croma.py` README documents the 8-bit / [0,1] convention); embedding
  quality degrades on out-of-distribution scaling.
- **Spatial resolution.** Default 120×120 (the pretrained resolution).
  Larger contexts change the patch-token count and are out-of-distribution
  for the baked positional encodings.
