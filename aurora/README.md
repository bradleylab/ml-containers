# aurora

Aurora, Microsoft's foundation model for the Earth system (Bodnar et al.,
*Nature*, 2025): a 1.3-billion-parameter 3D Swin Transformer between Perceiver
encoder and decoder layers, pretrained on a broad mix of weather and climate
data and then fine-tuned per task. Across its checkpoints the training data
include reanalysis, analysis, forecast and climate-simulation products (ERA5,
CMIP6, IFS HRES, GFS, CAMS). The 0.25° weather models take the global
atmosphere at two times six hours apart, 4 surface variables and 5 atmospheric
variables on 13 pressure levels, and predict the state six hours later; longer
forecasts are autoregressive rollouts of that step. This image and its Compute2 check
([`SMOKE.md`](SMOKE.md)) target those 0.25° weather checkpoints. The same
package holds the classes for the 0.1°, air-pollution, ocean-wave and Aurora 1.5
checkpoints, whose input data are what limit them here.

Paper: [doi:10.1038/s41586-025-09005-y](https://doi.org/10.1038/s41586-025-09005-y)
([arXiv:2405.13063](https://arxiv.org/abs/2405.13063)) ·
Code: [microsoft/aurora](https://github.com/microsoft/aurora) @ `v2.0.1`
(`0ad2d9a`), PyPI `microsoft-aurora==2.0.1` ·
Weights: [microsoft/aurora](https://huggingface.co/microsoft/aurora) on Hugging Face

Pull: `ghcr.io/bradleylab/aurora:v1` (also `:latest`, `:torch2.13-cu129`)

> **Status: experimental.**

## Released checkpoints

Every checkpoint is in the one Hugging Face repository, MIT-licensed and
ungated. Each pairs with a model class in the package and targets one source of
initial conditions; upstream's documentation is emphatic that
performance depends on feeding a checkpoint the data it was trained on,
regridded the same way. Sizes are from the Hugging Face API.

| Checkpoint | Class | Grid | Step | Initial conditions it expects | Size | Staged |
|---|---|---|---|---|---|---|
| `aurora-0.25-pretrained.ckpt` | `AuroraPretrained` | 0.25°, 721 × 1440 | 6 h | ERA5, or any source without its own fine-tune | 5.03 GB | yes |
| `aurora-0.25-finetuned.ckpt` | `Aurora` | 0.25° | 6 h | IFS HRES T0 | 5.04 GB | yes |
| `aurora-0.25-small-pretrained.ckpt` | `AuroraSmallPretrained` | 0.25° | 6 h | debugging only, in upstream's words | 0.45 GB | yes |
| `aurora-0.25-12h-pretrained.ckpt` | `Aurora12hPretrained` | 0.25° | 12 h | as the pretrained model | 5.03 GB | no |
| `aurora-0.1-finetuned.ckpt` | `AuroraHighRes` | 0.1°, 1801 × 3600 | 6 h | IFS HRES analysis | 4.81 GB | no |
| `aurora-0.25-v1.5.ckpt` | `AuroraV1p5` | 0.25° | 6 h, sub-steps to 1 h | IFS HRES T0 with 19 surface inputs and 36 static fields | 5.03 GB | no |
| `aurora-0.25-v1.5-ensemble.ckpt` | `AuroraV1p5Ensemble` | 0.25° | as Aurora 1.5, stochastic | as Aurora 1.5 | 5.04 GB | no |
| `aurora-0.25-wave.ckpt` | `AuroraWave` | 0.25° | 6 h | IFS HRES T0 plus HRES-WAM wave analysis | 5.04 GB | no |
| `aurora-0.4-air-pollution.ckpt` | `AuroraAirPollution` | 0.4°, 451 × 900 | 12 h | CAMS analysis | 5.10 GB | no |

The full repository is 41.2 GB, the nine checkpoints plus static-field files and
a test batch. Each class's `load_checkpoint()` downloads its default checkpoint
from a Hugging Face revision hard-coded in the package; `load_checkpoint_local()`
takes a path instead, which is how this image uses the staged copy.

**IFS HRES T0 is not IFS HRES analysis.** The analysis adds a surface
assimilation step. The 0.25° fine-tune expects T0 and the 0.1° fine-tune
expects the analysis, and upstream warns against swapping them.

**Aurora 1.5 is upstream's recommendation over the 0.25° fine-tune**, with
better accuracy, 26 surface variables and hourly lead times. It is not staged
because of its inputs: 19 surface variables, among them 2 m dewpoint; total,
low, medium and high cloud cover; skin temperature; soil temperature and
moisture; sea ice; and snow depth, plus 36 static fields. The WeatherBench2
HRES T0 store that feeds this image's fine-tuned model carries 5 of those
inputs and no static fields.

## Weights, staged beside the image

Nothing is baked. The weights are MIT and ungated, so size is the only reason.
The lab's copy is on Storage3 at `Active/aurora`, produced by
[`scripts/stage_weights.py`](scripts/stage_weights.py), which the image carries
at `/opt/aurora/stage_weights.py`. It pins revision
`0be7e57c685dac86b78c4a19a3ab149d13c6a3dd`, the one microsoft-aurora 2.0.1
itself pins for these three checkpoints, so each staged file is the file
`load_checkpoint()` would fetch.

```
Active/aurora/                                         10.8 GB
├── aurora-0.25-pretrained.ckpt                        5.03 GB
├── aurora-0.25-finetuned.ckpt                         5.04 GB
├── aurora-0.25-small-pretrained.ckpt                  0.45 GB
├── aurora-0.25-static.pickle                          12 MB    lsm, slt, z on the 0.25° grid
├── aurora-0.25-small-pretrained-test-input.pickle     164 MB   upstream's reference batch
├── aurora-0.25-small-pretrained-test-output.pickle     82 MB   and its expected prediction
└── initial_states/                                    ~0.8 GB per case, from the SMOKE.md staging job
```

The pretrained model is the one for ERA5, the reanalysis for past events. The
fine-tuned model is the one for IFS HRES T0, which WeatherBench2 serves for
2016 to early 2023. The small model and upstream's reference batch let a
compute node check the stack against a known answer without network access.

## Initial conditions

A forecast needs the global state at two times six hours apart: `2t`, `10u`,
`10v` and `msl` at the surface; `z`, `u`, `v`, `t` and `q` on 50, 100, 150,
200, 250, 300, 400, 500, 600, 700, 850, 925 and 1000 hPa; and the static
fields `lsm`, `slt` and `z`.

Upstream's ERA5 example fetches ERA5 and the static fields from the Copernicus
Climate Data Store, which needs an account and an API key, and its HRES T0
example fetches the static fields there too. This image reads the same fields
anonymously from [WeatherBench2](https://weatherbench2.readthedocs.io/) on
public Google Cloud Storage instead, and takes the static fields from
`aurora-0.25-static.pickle`, which upstream publishes as a convenience copy of
the ERA5 fields and pairs with WeatherBench2 in its own Foundry demo:

| Store | Contents | Coverage |
|---|---|---|
| `gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721.zarr` | ERA5, 0.25°, 13 levels, 6-hourly | 1959-01-01 to 2023-01-10 |
| `gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr` | IFS HRES T0, same grid and levels | 2016-01-01 to 2023-01-10 |

Aurora requires latitudes to decrease from north to south. The ERA5 store is
already in that order; the HRES T0 store runs south to north and must be
flipped, as upstream's HRES T0 example does. Longitudes run 0 to 359.75° in
both. One initial state plus its verifying time is about 0.8 GB;
`aurora-verify-onestep --cache` writes it to netCDF so a GPU job can run
without network.

## Running it

Mount the staged directory at `/weights`. The model classes and checkpoint
names are the ones in the table; the batch is built as in
[`scripts/aurora-verify-onestep`](scripts/aurora-verify-onestep).

```python
import torch
from aurora import AuroraPretrained, rollout

model = AuroraPretrained()
model.load_checkpoint_local("/weights/aurora-0.25-pretrained.ckpt")
model = model.eval().to("cuda")

with torch.inference_mode():
    preds = [p.to("cpu") for p in rollout(model, batch, steps=4)]   # 4 x 6 h
```

Moving each prediction to the CPU as it arrives is upstream's pattern and keeps
GPU memory flat over a long rollout. Every prediction has 720 latitude rows, not
721: the model drops the South Pole row to reach a multiple of its patch size.

**GPU memory.** Upstream gives approximately 40 GB for the full model on the
global 0.25° grid, so one 80 GB H100 is enough; `aurora-verify-onestep` prints
the peak allocation. The small model also runs on CPU.

**Reproducibility.** Upstream's recipe for deterministic output is
`torch.use_deterministic_algorithms(True)` plus `model.eval()`. Decide before a
forecast goes into a manuscript whether it needs that, and record the choice.

## Air-pollution model: CAMS data access

`aurora-0.4-air-pollution.ckpt` forecasts CAMS atmospheric composition at 0.4°
in 12-hour steps: particulate matter (PM1, PM2.5, PM10), total columns of CO,
NO, NO2, SO2 and ozone, and CO, NO, NO2, SO2 and ozone on the 13 pressure
levels, alongside the meteorological variables. Its initial states must come
from CAMS analysis, which upstream's example retrieves from the Copernicus
Atmosphere Data Store. That service requires a registered account, an API key
and acceptance of the dataset's terms of use, so the checkpoint is not staged
and this image makes no claim about its output. The package in the image
includes its model class, `AuroraAirPollution`, and its static fields are
`aurora-0.4-air-pollution-static.pickle` in the same Hugging Face repository;
what it needs beyond this image is the account and the data.

## Stack

- Base: `python:3.11-slim-bookworm`. Upstream's CI tests Python 3.10 and 3.11.
- `torch==2.13.0` from the PyTorch CUDA 12.9 index, with the torchvision that
  timm needs. 2.13.0 is the torch upstream's CI tested 2.0.1 with; CI took
  PyPI's CUDA 13.0 build, and this image uses the CUDA 12.9 build that
  `nasa-ibm-lunar-fm` runs on the same H100 nodes (driver 580.105.08).
- `microsoft-aurora==2.0.1`, plus `xarray`, `zarr` and `gcsfs` for anonymous
  WeatherBench2 reads.
- [`constraints.txt`](constraints.txt) freezes the resolution, 99 pins, with
  PyPI limited to releases before 2026-08-05, the day after 2.0.1 was
  published. The package pins nothing, and that cutoff reproduces the versions
  upstream's CI installed at the release commit for every one of its
  dependencies: numpy 2.4.6, timm 1.0.28, xarray 2026.7.0, huggingface-hub
  1.26.0, scipy 1.17.1, einops 0.8.2, pydantic 2.13.4, netcdf4 1.7.4 and
  azure-storage-blob 12.30.0.
- `aurora-verify-onestep` and `aurora-verify-small` on `PATH`; see
  [`SMOKE.md`](SMOKE.md).

The build test is offline and weight-free. It imports torch alone and asserts a
CUDA 12.9 build, then runs upstream's documented random-batch example through
the small architecture with random initialization and checks that every output
variable is finite, on the expected cropped grid and at the expected valid
time. [`SMOKE.md`](SMOKE.md) is the Compute2 test: upstream's small-model
reference test, then one 6-hour step from ERA5 and from HRES T0, each scored
against the verifying analysis and against persistence.

## Related images

- `fourcastnet3` and `aifs` are the other global medium-range models in this
  repository, both 6-hourly. FourCastNet 3 runs on the same 0.25° grid with 72
  variables and a stochastic core built for ensembles; AIFS runs on the N320
  grid (~31 km) and initializes from ECMWF Open Data.
- `stormcast` is regional: 3 km over CONUS in 1-hour steps, conditioned on a
  coarse global forecast. `corrdiff` downscales 0.25° ERA5 to kilometer-scale
  fields over Europe. Neither is a global forecaster.
- Earth-2 Studio, the framework in `fourcastnet3`, `stormcast` and `corrdiff`,
  has its own Aurora wrapper around the 0.25° pretrained checkpoint. Those
  images do not request it; this one uses Microsoft's package directly, which
  also reaches the fine-tuned and other checkpoints.

## License

MIT for the code (`microsoft/aurora`, `LICENSE.txt`, copyright Microsoft
Corporation) and for all released checkpoints (`microsoft/aurora` on Hugging
Face). Upstream's README adds two statements that are not license terms: it
asks anyone interested in commercial use to contact Microsoft at
AIWeatherClimate@microsoft.com, and it describes the release as research code,
not an operational forecast service, whose output should not drive decisions
without domain-expert review.

Cite Bodnar et al. (2025), *A Foundation Model for the Earth System*, *Nature*,
doi:10.1038/s41586-025-09005-y.
