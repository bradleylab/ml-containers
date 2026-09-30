# satclip

SatCLIP (Klemmer, Rolf, Robinson, Mackey and Rußwurm; AAAI 2025) is a global,
general-purpose location encoder. It was trained contrastively, in the manner
of CLIP, to match Sentinel-2 image patches with the coordinates they came from,
using S2-100K: 100,000 Sentinel-2 L2A patches (256 × 256 px, 12 bands at 10 m)
sampled roughly uniformly over land. What the
location encoder learned from that pairing is carried in a 256-dimensional
embedding of any longitude and latitude, which serves as a covariate for
downstream geospatial regression and classification.

Paper: [arXiv:2311.17179](https://arxiv.org/abs/2311.17179) ·
[AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/32457) ·
Code: [microsoft/satclip](https://github.com/microsoft/satclip) @ `1ead106` ·
Weights: [Hugging Face](https://huggingface.co/models?other=arxiv:2311.17179)

Pull: `ghcr.io/bradleylab/satclip:v1`

## Location encoder and image encoder

A SatCLIP model has two halves. The image encoder is a Sentinel-2 backbone
(ResNet18, ResNet50 or ViT-S/16, pretrained with MoCo and frozen during SatCLIP
training except for its last projection layer). The location encoder evaluates
a spherical-harmonic basis of L Legendre polynomials (L² terms) at the point and
passes it through a SIREN network; the paper puts it at about one million
parameters, against about 22 million in the ViT16 image encoder. The location
encoder is the useful product: an embedding for any coordinate, with no imagery
needed at inference time.

This image carries the location encoder only. Upstream's documented loader,
`load.get_satclip`, rebuilds the whole training module, and for every released
checkpoint that means fetching torchgeo's Sentinel-2 MoCo backbone weights from
the network to construct an image encoder that embedding locations never uses.
The image instead loads checkpoints through upstream's own
`load_lightweight.get_satclip_loc_encoder`, which rebuilds only the location
encoder, runs offline, and needs nothing beyond torch, numpy and einops. The
image encoder and the training stack (Lightning, torchgeo, timm, albumentations,
rasterio) are not installed.

The location encoder also runs on CPU, which is enough for most tables. Its
cost grows with L²: an L = 40 checkpoint evaluates 1,600 spherical-harmonic
terms per point against 100 for L = 10, and on an 8-thread CPU embeds 100,000
points in tens of seconds where L = 10 takes about one. The image carries CUDA
13.0 torch for the H100 nodes, which suits large tables at L = 40 and lets the
embeddings feed GPU training in the same job.

## Baked checkpoints

All six released checkpoints are in the image under `/opt/satclip-weights`,
each fetched at a pinned Hugging Face revision and checked against its published
SHA-256. They are MIT-licensed and ungated, and together they are 598 MB.

| `--checkpoint` | Image encoder | L | File size | Hugging Face repository @ revision |
|---|---|---|---|---|
| `vit16-l40` (default) | ViT-S/16 | 40 | 121 MB | `microsoft/SatCLIP-ViT16-L40` @ `0ef2acc` |
| `vit16-l10` | ViT-S/16 | 10 | 103 MB | `microsoft/SatCLIP-ViT16-L10` @ `829a790` |
| `resnet50-l40` | ResNet50 | 40 | 130 MB | `microsoft/SatCLIP-ResNet50-L40` @ `b66714c` |
| `resnet50-l10` | ResNet50 | 10 | 111 MB | `microsoft/SatCLIP-ResNet50-L10` @ `f41ed77` |
| `resnet18-l40` | ResNet18 | 40 | 76 MB | `microsoft/SatCLIP-ResNet18-L40` @ `983a9d9` |
| `resnet18-l10` | ResNet18 | 10 | 57 MB | `microsoft/SatCLIP-ResNet18-L10` @ `5263ff2` |

Every checkpoint yields a 256-dimensional embedding. The paper names ViT16 as
its default backbone, while the caption of its main results table (Table 2)
names ResNet50, so both are kept for comparison with published numbers.

**Choosing L matters more than choosing the backbone.** The paper reports that
the image encoder changes downstream scores by less than 1%, whereas L changes
them more. L = 40 resolves finer structure and does better at spatial
interpolation, where training and test points are interleaved, and on regional
tasks such as California housing prices. L = 10 is smoother and does better at
geographic generalization, where the test region (a held-out continent in the
paper) has no training points. The L = 40 models also overfit more readily
during pretraining (paper Appendix C.2).

## Embedding a table of coordinates

`satclip-embed` reads a CSV or Parquet table with longitude and latitude
columns in decimal degrees (WGS 84) and writes one embedding per row, in input
order. Parquet output keeps the input columns and adds `satclip_000` to
`satclip_255`; `.npy` output is an `(n_rows, 256)` float64 array.

```bash
satclip-embed sites.csv sites_satclip.parquet
satclip-embed sites.parquet sites_l10.npy \
  --checkpoint vit16-l10 \
  --lon-col longitude \
  --lat-col latitude
```

The defaults are `--lon-col lon`, `--lat-col lat`, `--checkpoint vit16-l40`, and
the GPU when one is visible. A row with a missing, non-numeric or out-of-range
coordinate (longitude outside −180 to 180, latitude outside −90 to 90) stops the
run: the offending rows are listed and nothing is written, so no row is dropped
without notice.

On Compute2, from a CPU partition, the GPU hook is switched off with
`NVIDIA_VISIBLE_DEVICES=void` or enroot refuses to start the container:

```bash
sbatch -A compute2-alexander.s.bradley -p general-cpu \
       --cpus-per-task=4 --mem=16G --time=00:30:00 \
       -J satclip-embed -o satclip-embed-%j.out --wrap='
srun --export=ALL,NVIDIA_VISIBLE_DEVICES=void \
     --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+satclip+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/data \
     --container-workdir=/data \
     satclip-embed sites.csv sites_satclip.parquet'
```

From Python inside the image, the same loader is one import away
(`/opt/satclip/satclip` is on `PYTHONPATH`):

```python
import torch
from load_lightweight import get_satclip_loc_encoder

encoder = get_satclip_loc_encoder("/opt/satclip-weights/satclip-vit16-l40.ckpt", "cpu")
lonlat = torch.tensor([[-90.1978, 38.6272]], dtype=torch.float64)  # (lon, lat)
with torch.no_grad():
    emb = encoder(lonlat)  # shape (1, 256)
```

The encoder takes `(lon, lat)` in that order, as float64.

## Limits

These come from upstream's model card and paper.

- **Global scale, not fine-grained.** The encoders were pretrained for
  global-scale use. The model card places fine-grained problems, those confined
  to a small area or with many closely spaced locations, out of scope. Nearby
  points get nearly identical embeddings: St. Louis and Kirkwood, MO, about
  20 km apart, have a cosine similarity of 0.998 or higher under every
  checkpoint.
- **Landscape-scale content.** Training imagery is Sentinel-2 at 10 m per
  pixel, so the embeddings can reflect structures such as cities and mountain
  ranges but not small objects, and only what is visible in that imagery at all.
- **No time axis.** The S2-100K scenes span January 2021 to May 2023 and the
  model marginalizes over them; an embedding describes a place, not a date or a
  season.
- **Research use.** The model card describes SatCLIP as trained and tested for
  research and not intended for production, and places defense and surveillance
  uses out of scope.

## Stack

- Base: `python:3.12-slim-bookworm`.
- `torch==2.14.0+cu130` from the PyTorch CUDA 13.0 index. Upstream pins no
  torch; 13.0 is the newest CUDA that Compute2's driver (580.105.08) supports,
  and PyTorch's release build for it includes sm_90 kernels for the H100.
- The upstream source tree at `1ead106` in `/opt/satclip`, with its inner
  `satclip/` directory on `PYTHONPATH`. Upstream is not a pip package, and its
  modules import one another by bare name, as its notebooks show with
  `sys.path.append('./satclip')`.
- [`constraints.txt`](constraints.txt) freezes the full resolved dependency set
  (35 pins, resolved 2026-09-28), so a rebuild installs the same versions.
- Scripts on `PATH`: `satclip-embed`, `satclip-selftest` and
  `satclip-verify-elevation`.

The build test is offline. It checks that torch is the CUDA 13.0 build, runs
`satclip-selftest` (every checkpoint loads, returns finite 256-dimensional
embeddings, and places St. Louis nearer to Kirkwood, MO than to the Sahara or
the Amazon basin), and runs `satclip-embed` end to end, including its refusal
of an out-of-range row. [`SMOKE.md`](SMOKE.md) is the Compute2 test, which
reruns the paper's global elevation regression on an H100.

## Related images

- `geoclip` runs the inverse task. GeoCLIP maps a photograph to a location,
  retrieving likely coordinates from a GPS gallery; SatCLIP maps a location to
  an embedding. GeoCLIP's own location encoder, trained on the MP-16 geotagged
  photo collection, is one of the baselines in the SatCLIP paper.

## License

MIT for the code ([microsoft/satclip](https://github.com/microsoft/satclip),
Copyright Microsoft Corporation) and for all six checkpoints, as stated on each
Hugging Face model card. The model card scopes use to research.
