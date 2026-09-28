# copernicus-fm

Copernicus-FM (Wang et al., ICCV 2025) is an Earth-observation foundation model
from the zhu-xlab group that takes any Copernicus input, spectral or not, into one
ViT-B/16 encoder. It extends DOFA's wavelength-conditioned patch embedding in two
ways. Spectral bands are described by central wavelength and bandwidth, and
non-spectral variables (Sentinel-5P trace gases, the Copernicus DEM) by a language
embedding of the variable's name, from which a second hypernetwork generates the
patch-embedding weights. Location, acquisition time and patch area enter as
optional Fourier-encoded metadata, and the patch-embedding kernel is resized per
modality. The model was pretrained for 100 epochs on Copernicus-Pretrain (18.7M
aligned images from Sentinel-1, -2, -3, -5P and the Copernicus DEM) with masked
image modeling and continual distillation. This image runs TorchGeo 0.10.0's
implementation, `torchgeo.models.copernicusfm_base`; the pretrained weights are
staged on lab storage and mounted at run time.

Paper: [arXiv:2503.11849](https://arxiv.org/abs/2503.11849) ·
Code: [zhu-xlab/Copernicus-FM](https://github.com/zhu-xlab/Copernicus-FM) @ `da95635` ·
Weights: [torchgeo/copernicus-fm](https://huggingface.co/torchgeo/copernicus-fm) @ `f395812`

Pull: `ghcr.io/bradleylab/copernicus-fm:v1` (also `:latest`, `:torch2.13-cu129`)

## Inputs

Each forward pass takes an image tensor `(B, C, H, W)`, a metadata tensor
`(B, 4)`, an `input_mode`, and a `kernel_size` (the patch size in input pixels).
The values below are the ones the model was pretrained with: wavelengths and
bandwidths from the paper's Table 9, kernel sizes from its implementation details
(section 4.3). **Wavelengths and bandwidths are in nanometers**, unlike the `dofa`
image, which takes micrometers.

| Input | `input_mode` | Band description | Kernel size |
|---|---|---|---|
| Sentinel-1 GRD, VV and VH | `spectral` | wavelengths `[5e7, 5e7]`, bandwidths `[1e9, 1e9]` | 16 |
| Sentinel-2 TOA, 13 bands (B1–B12, B8A) | `spectral` | wavelengths `[440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1370, 1610, 2190]`, bandwidths `[20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180]` | 16 |
| Sentinel-3 OLCI, 21 bands | `spectral` | per Table 9 of the paper | 8 |
| Sentinel-5P CO, NO₂, SO₂, O₃ | `variable` | 2048-long embedding of the variable name | 4 |
| Copernicus DEM | `variable` | 2048-long embedding of the variable name | 64 |

Any other spectral sensor goes in the same way, as a list of per-band central
wavelengths and bandwidths in nanometers. The encoder accepts wavelengths from
100 nm to 10⁹ nm and bandwidths from 1 nm to 10⁹ nm, and raises an error
outside those ranges.

**Metadata** is one row per image: longitude (degrees), latitude (degrees), time
(days since 1970-01-01; TorchGeo's Copernicus-Bench datasets report seconds, so
divide by 86,400) and the ground area of one patch token (km²). `NaN` marks
an unknown field, and the model substitutes a learned token for it. The check
runs over the whole batch, so one `NaN` longitude or latitude drops location for
every image in that batch, and the same holds for time and area. Batches that mix
known and unknown metadata should be split.

**Output** is a 768-long embedding per image: the mean of the patch tokens after
the last transformer block, followed by a LayerNorm. That final LayerNorm is not in
the pretrained checkpoint; it starts at unit scale and zero shift, as it does in
upstream's benchmark code.

## Weights

| File | Size | Source | License | Where |
|---|---|---|---|---|
| `CopernicusFM_ViT_base_varlang-085350e4.pth` (ViT-B/16) | 557,931,327 bytes | `torchgeo/copernicus-fm` @ `f395812` | CC-BY-4.0 | staged on Storage3, mounted at `/weights` |
| `varname_embed/varname_embed_llama3.2_1B.pt` | 116,962 bytes | `wangyi111/Copernicus-FM` @ `e1db406` | Llama 3.2 Community License | fetched by the user when needed |

The checkpoint has the same SHA-256
(`085350e4fa0ebc6047a501aed1307394c4fcc23c013dd026db65af06411d6bee`) as
`CopernicusFM_ViT_base_varlang_e100.pth` in the authors' `wangyi111/Copernicus-FM`
repository; TorchGeo's copy differs only in file name.

**The weights are staged, not baked.** They were pretrained on the Llama 3.2
variable-name encodings described next, and Meta's Llama 3.2 Community License
places conditions on distributing a model trained with Llama outputs that
upstream's CC-BY-4.0 release does not address. The lab therefore keeps the file on
its own storage rather than redistributing it in a public image.
`/opt/copernicus-fm/stage_weights.py` fetches it through TorchGeo's pinned weights
enum and checks the full SHA-256; the lab's copy is on Storage3 at
`Active/copernicus-fm/weights`. Mount that directory at `/weights`: the image sets
`TORCH_HOME=/weights/torch-cache`, so `CopernicusFM_Base_Weights.CopernicusFM_ViT`
loads it without network access.

The variable-name embeddings are Llama-3.2-1B encodings of fourteen names, seven
of them Copernicus variables: `Sentinel 5P Nitrogen Dioxide`,
`Sentinel 5P Sulfur Dioxide`, `Sentinel 5P Carbon Monoxide`, `Sentinel 5P Ozone`,
`Copernicus Digital Elevation Model`, `Sentinel 1 Synthetic Apeature Radar` and
`Sentinel 2 Top of Atmosphere` (the misspelling is the key as published). The
other seven are video-game titles and mountain names, out-of-domain probes of the
kind the paper plots in its Fig. 16. Upstream distributes the file under the
Llama 3.2 Community License, which requires a copy of that license and a "Built
with Llama" notice wherever the file or a derivative is redistributed, so it is
not baked. The variable hypernetwork was pretrained on these encodings (upstream's
pretraining instructions; paper, appendix B.3), and upstream publishes the model
weights under CC-BY-4.0.

The authors' Hugging Face repository also holds a ViT-L checkpoint (1.5 GB). TorchGeo 0.10.0
has no weights entry for it, and it is not in the image.

## Running it

Spectral input, Sentinel-2, with location and time unknown:

```python
import torch
from torchgeo.models import copernicusfm_base, CopernicusFM_Base_Weights

model = copernicusfm_base(weights=CopernicusFM_Base_Weights.CopernicusFM_ViT).eval().cuda()
x = torch.randn(8, 13, 224, 224, device='cuda')  # normalized S2 TOA
meta = torch.full((8, 4), float('nan'), device='cuda')
with torch.no_grad():
    emb = model(
        x, meta,
        wavelengths=[440, 490, 560, 665, 705, 740, 783, 842, 860, 940, 1370, 1610, 2190],
        bandwidths=[20, 65, 35, 30, 15, 15, 20, 115, 20, 20, 30, 90, 180],
        input_mode='spectral', kernel_size=16,
    )  # (8, 768)
```

Variable input, Sentinel-5P NO₂, after fetching the name embeddings (pinned to the
revision above) into a directory of your own:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    'wangyi111/Copernicus-FM', 'varname_embed/varname_embed_llama3.2_1B.pt',
    revision='e1db406d517a122c8373802e1c130c5fc4789f84', local_dir='/work/varname',
)
names = torch.load(path, weights_only=True)
# no2: (B, 1, H, W), normalized tropospheric NO₂ column density
emb = model(
    no2, meta, language_embed=names['Sentinel 5P Nitrogen Dioxide'].cuda(),
    input_mode='variable', kernel_size=4,
)
```

On Compute2, after `enroot import` (see [`SMOKE.md`](SMOKE.md)):

```bash
srun -A compute2-alexander.s.bradley -p general-gpu --gpus=1 \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+copernicus-fm+v1.sqsh \
     --container-mounts=$PWD:/work,/storage3/fs1/alexander.s.bradley/Active/copernicus-fm/weights:/weights \
     --container-workdir=/work \
     python my_script.py
```

Normalize inputs before the encoder. Upstream's Copernicus-Bench configs
standardize each band with a per-band mean and standard deviation, for example
Sentinel-1 VV and VH with means `[-12.548, -20.192]` and standard deviations
`[5.257, 5.912]` (`cobench_eurosat_s1.yaml`).

## Benchmark check

`cfm-verify-eurosat-s1` reruns one published Copernicus-Bench score on the GPU: a
linear probe on the frozen encoder for EuroSAT-S1 (Sentinel-1 VV and VH, 10
land-cover classes), whose Table 4 test accuracy is 87.2 ± 0.1 % over three runs.
It follows upstream's benchmark code for normalization, resizing, flips, metadata,
the probe head, optimizer, schedule and checkpoint selection, and fetches the
dataset through TorchGeo, which pins its revision and checksum. Upstream's
committed runner uses seed 42 and does not record the other two seeds.
[`SMOKE.md`](SMOKE.md) runs it for three seeds.

## Differences from upstream's code

TorchGeo's `copernicusfm_base` builds its LayerNorms with PyTorch's default
`eps=1e-5`; upstream's pretraining and benchmark ViTs use `eps=1e-6`.
`cfm-verify-eurosat-s1 --layernorm-eps 1e-6` builds the encoder with upstream's
value.
Upstream's model returns `(logits, embedding)`; TorchGeo's returns the head output,
which is the embedding itself when `num_classes=0`, the default.

## Stack

- Base: `python:3.12-slim-bookworm` (TorchGeo 0.10.0 requires Python ≥ 3.12)
- `torch==2.13.0` and `torchvision==0.28.0` from the CUDA 12.9 index: the pair
  TorchGeo 0.10.0's `uv.lock` pins. PyTorch builds its CUDA 12.9 x86-64 wheels
  with sm_90 kernels, Compute2's H100 driver (580.105.08) supports CUDA up to
  13.0, and `nasa-ibm-lunar-fm`, which also runs TorchGeo, uses CUDA 12.9 too.
- `torchgeo==0.10.0`, core dependencies only (kornia, timm, Lightning,
  segmentation-models-pytorch, rasterio, einops).
- [`constraints.txt`](constraints.txt): the full resolved set, 105 pins, resolved
  2026-09-28, with every package TorchGeo's lockfile also pins held to the
  lockfile's version.

The build test is offline: after baking the checkpoint, it loads the model with
network access blocked and runs Sentinel-2 (13 bands), Sentinel-1 (2 bands, with
metadata) and variable-mode (56 × 56, kernel 4) inputs through it, asserting a
finite `(2, 768)` embedding for each. The variable-mode pass uses a random vector
in place of a name embedding, so it checks that the path runs, not what it
computes.

## Related images

- `dofa` is the predecessor: DOFA-Base on a CPU build of TorchGeo, conditioned on
  wavelength alone (in micrometers), with no bandwidth, variable or metadata
  input. On EuroSAT-S1 in Table 4 it scores 81.7 ± 0.1 against Copernicus-FM's
  87.2 ± 0.1, and on EuroSAT-S2 97.2 ± 0.1 against 97.9 ± 0.1. Its embeddings and
  Copernicus-FM's live in different spaces and are not interchangeable.
- `dofa-clip` is DOFA aligned with text for zero-shot retrieval (CC-BY-NC-4.0);
  Copernicus-FM has no text encoder.
- `croma` covers Sentinel-1 and -2 only; Table 4 puts it at 83.9 ± 0.1 on
  EuroSAT-S1.

## License

The model code in the image is TorchGeo's (MIT). The weights are CC-BY-4.0; use
of them requires attribution to Wang, Y., Xiong, Z., Liu, C., Stewart, A. J.,
Dujardin, T., Bountos, N. I., Zavras, A., Gerken, F., Papoutsis, I., Leal-Taixé,
L., and Zhu, X. X. (2025), "Towards a Unified Copernicus Foundation Model for Earth
Vision", arXiv:2503.11849. Upstream's research code is Apache-2.0 with MIT and
CC-BY-NC-4.0 (MAE) third-party portions; none of it is in the image. The
variable-name embeddings are under the Llama 3.2 Community License and are not in
the image. The weights are CC-BY-4.0 as released, and are kept on lab storage for
the reason given under "Weights". The Copernicus-Bench EuroSAT-S1 data that `cfm-verify-eurosat-s1`
downloads are CC-BY-4.0 (paper, Table 3).
