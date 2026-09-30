# panopticon

Panopticon (Waldmann, Shah et al., 2025; best paper at the CVPR 2025
EarthVision workshop) is an any-sensor Earth-observation foundation model: a
DINOv2 ViT-B/14 whose patch embedding takes any number of channels, each
identified by its center wavelength in nm or, for SAR, by a polarization and
orbit category. It was pretrained with DINOv2's self-distillation on fMoW,
fMoW-Sentinel, SatlasPretrain, MMEarth and SpectralEarth, the last being 202-band
EnMAP hyperspectral cubes. The output is a normalized 768-D class token and 768-D
patch tokens. It is packaged here as a hyperspectral baseline, beside `dofa`,
for the lab's TanagerFM project on Planet Tanager-1 imagery (426 bands).

Paper: [arXiv:2503.10845](https://arxiv.org/abs/2503.10845) ·
Code: [Panopticon-FM/panopticon](https://github.com/Panopticon-FM/panopticon) @ `24a229a` ·
Weights: [lewaldm/panopticon](https://huggingface.co/lewaldm/panopticon) @ `c8c2bb9`

Pull: `ghcr.io/bradleylab/panopticon:v1`

## Input contract: one channel id per band

The model takes a dictionary `{"imgs": (B, C, H, W), "chn_ids": (B, C)}`. Every
band needs an id, in band order:

- **Optical bands: the center wavelength in nm** (664 for red, not 0.664). The
  channel embedding floors the value to a whole nanometer and encodes it with a
  2304-D sinusoid, so two bands whose centers floor to the same integer get the
  same embedding. Tanager-1 bands are about 5 nm apart and EMIT bands about
  7.4 nm, so neither collides. Band width is not used: the released model encodes
  the center only (the paper's Appendix E variant with the response width is not
  the released one).
- **SAR bands: a category id from −1 to −12**, from upstream's
  `dinov2/configs/data/satellites/sentinel1.yaml` (the table the upstream README
  links in geobreeze is identical):

  | Orbit | VV | VH | HH | HV |
  |---|---|---|---|---|
  | both or unknown | −1 | −2 | −3 | −4 |
  | ascending | −5 | −6 | −7 | −8 |
  | descending | −9 | −10 | −11 | −12 |

The model adds the channel embeddings to the band tokens by broadcasting, so a
single id passed for a multi-band image runs without error and labels every band
the same. `panopticon_model.validate_channel_ids` refuses any list whose length
differs from the band count, and any negative id outside −1 to −12.

**Channel counts beyond pretraining.** In pretraining, each global view carried
4 to 13 channels and each local view 1 to 4, subsampled at random from the
sensor's bands (paper §3.2; `global_crops_spectral_size` and
`local_crops_spectral_size` in upstream `dinov2/configs/stage2.yaml`). No
pretraining view held more than 13 bands. The largest input the paper evaluates
is EnMAP-Corine with all 202 bands (linear probing, Fig. 3). A full Tanager-1 or
EMIT cube is therefore outside both the training and the evaluation
distributions in channel count.

## Image size and normalization

Pretraining used 224 × 224 global crops, which the patch size of 14 turns into
a 16 × 16 grid of 256 patch tokens. Other sizes work if both sides are multiples
of 14: the positional embedding is stored for a 37 × 37 grid and interpolated.
Upstream asks for 224 × 224 inputs for best results.

Upstream standardizes each band to zero mean and unit variance, using the mean
and standard deviation of each pretraining dataset's training split (paper
§3.5). For a lab corpus, compute those statistics once over its training split
and pass them to the embedding script with `--normalize stats`. The script's
`--normalize patch` mode, which standardizes each band with the patch's own
statistics, removes absolute reflectance differences between patches and suits
engineering tests rather than analysis.

## Memory and time as the band count grows

The patch embedding (`dinov2/models/panopticon.py`) works band by band, and its
cost is linear in the number of bands C. For an image of H × W pixels it forms
L = (H/14)(W/14) patches, 256 at 224 × 224. A 1 × 14 × 14 3-D convolution maps
every band of every patch to its own 2304-D token (`attn_dim` in `hubconf.py`),
giving a B × C × L × 2304 tensor. The wavelength embedding is added in place,
and 2304 × 2304 key and value projections are applied to every one of those
tokens. A single learned query then attends over the C tokens of each patch,
and the result is projected to 768-D. The ViT-B trunk after that sees 257 tokens
whatever C is.

Per band and per image at 224 × 224:

- one fp32 2304-D token block is 256 × 2304 × 4 bytes = 2.25 MiB, and at least
  four such blocks are alive together at the peak: the convolution output, its
  permuted copy, and the key and value projections. The attention matmuls can
  add layout copies on top;
- the key and value projections cost 2 × 256 × 2304² ≈ 2.7 × 10⁹
  multiply-accumulates, and the convolution adds 256 × 2304 × 196 ≈ 1.2 × 10⁸.

The 12-block trunk costs about 2.3 × 10¹⁰ multiply-accumulates per image,
independent of C (12 × (12·N·D² + 2·N²·D) with N = 257 tokens, D = 768). The
patch embedding therefore overtakes the trunk at about 8 bands, is about 1.5
times the trunk at 12 bands, and about 50 times at 426 Tanager bands. At 426
bands four blocks come to about 3.7 GiB per image in fp32, so a batch of 8 needs
at least 30 GiB for the patch embedding alone; bf16 autocast halves the block
size. All of this also scales with H × W. The attention scores themselves
are small (B × L × 16 heads × C values). These are arithmetic from the code;
[`SMOKE.md`](SMOKE.md) measures peak memory and time on real Tanager and EMIT
patches on an H100.

## Running it

```bash
python /opt/scripts/panopticon_embed.py \
  --image /work/patch.tif \
  --channel-ids-file /work/wavelengths_nm.txt \
  --normalize stats --band-stats /work/band_stats.json \
  --out /work/embedding.npz
```

`--image` takes a `(C, H, W)` `.npy` array or a GeoTIFF. Channel ids come from
`--channel-ids 664 559 493 ...` or from a whitespace-separated file, one per
band; a count mismatch exits with status 2 before the model loads. `--resize N`
resamples bilinearly to N × N. The output `.npz` holds `cls` (1, 768), `patch`
(1, L, 768), the channel ids and the normalization mode. On a GPU node the
script uses the GPU.

From Python, with `/opt/scripts` already on `PYTHONPATH`:

```python
import torch
from panopticon_model import load_teacher, embed

model = load_teacher("cuda")
out = embed(model, images.cuda(), wavelengths_nm)   # images: (B, C, H, W), standardized
out["cls"]     # (B, 768), normalized class token from the last block
out["patch"]   # (B, L, 768), normalized patch tokens from the last block
```

Features from intermediate blocks, which upstream uses for segmentation
(blocks 3, 5, 7 and 11), come from the model's own
`get_intermediate_layers(x_dict, n=[...], return_class_token=True)`.

## What is baked

- The teacher checkpoint `panopticon_vitb14_teacher.pth` (395,965,930 bytes,
  sha256 `55024f41…d9b833b6`, full hash in the Dockerfile) at
  `/opt/panopticon-weights/`. The model card describes it as the teacher weights
  from the full training checkpoint, sufficient for inference; the full
  student–teacher checkpoint (978 MB), the RGB DINO heads and the pretraining
  metadata files are not baked.
- The upstream repository at `24a229a` in `/opt/panopticon`, loaded by
  `torch.hub.load(..., source="local")`. It is not pip-installed: its `setup.py`
  installs a package named `dinov2` with upstream's full training requirements.

Upstream's public entrypoint `panopticon_vitb14` downloads the checkpoint from
Hugging Face on every call, even when the file exists. `panopticon_model.load_teacher`
builds the identical network with upstream's `_panopticon_vitb14` and applies
the same strict `load_state_dict` to the baked file, so nothing reaches the
network. `PANOPTICON_REPO` and `PANOPTICON_WEIGHTS` override the two paths.

## Stack

- Base: `python:3.12-slim-bookworm`.
- `torch==2.12.1` from the CUDA 12.9 index, the build `nasa-ibm-lunar-fm` runs
  on Compute2's H100s under driver 580.105.08, in place of upstream's torch 2.0.0
  / CUDA 11.7 training pin. The inference path imports only torch; xformers is
  optional upstream and serves the nested-tensor batches of training, so it is
  not installed.
- numpy, h5py and rasterio for the lab scripts.
- [`constraints.txt`](constraints.txt) freezes the full resolution, 37 pins
  resolved 2026-09-28.

The build test imports torch and the upstream model code in separate steps under
faulthandler, then loads the baked teacher with every socket connect disabled and
embeds a synthetic 224 × 224 batch twice: RGB at 664, 559 and 493 nm, and the 12
Sentinel-2 L2A bands at upstream's band centers. It asserts shapes (2, 768) and
(2, 256, 768) and finite values, and prints package versions.

## Relation to `dofa` and Copernicus-FM

All three take a per-band wavelength and produce one embedding for any band set.
They differ in how the bands are combined:

- **DOFA** (`dofa`) generates the patch-embedding weights for each band from its
  wavelength with a hypernetwork. Wavelengths are given in micrometers. The lab
  image is CPU-only, with the base weights baked.
- **Copernicus-FM** (Wang et al., 2025, arXiv:2503.11849) extends DOFA's
  dynamic hypernetwork to non-spectral modalities and adds metadata encoding;
  its weights are CC-BY-4.0.
- **Panopticon** keeps a token per band and patch, in a 2304-D space, and fuses
  the bands by cross-attention. That is the source of the linear, comparatively
  steep growth in memory and time described earlier.

The Panopticon paper compares against DOFA with frozen backbones. On
hyperspectral-derived tasks, Panopticon gives 85.8 and 86.2 mAP on Corine
synthesized as SuperDove and MODIS bands (DOFA 81.0 and 80.2) and an MSE of 0.313
and 0.321 on Hyperview synthesized the same way (DOFA 0.334 and 0.338) (Table 2).
On GEO-Bench m-eurosat, linear probing gives 96.4 % against DOFA's 92.9 % (Table
3). Embeddings from the three models live in different spaces and are not
interchangeable.

## License

- **Code: Apache-2.0.** The repository LICENSE is Apache-2.0, and the
  DINOv2-derived files carry Meta's Apache-2.0 header. Two dataset loaders
  adapted from TorchGeo (`dinov2/data/datasets/benv2.py` and `resisc45.py`) carry
  Microsoft's MIT header. The upstream README describes this as Apache-2.0 with
  portions of third-party code under MIT.
- **Weights: CC-BY-4.0, as the upstream README states.** The Hugging Face
  repository that hosts the checkpoint declares MIT instead. Both permit
  redistribution with attribution, so the image carries the stricter of the two;
  cite Waldmann, Shah et al. (2025) when using the embeddings.

The image label `org.opencontainers.image.licenses` is `Apache-2.0 AND CC-BY-4.0`.
