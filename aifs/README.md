# aifs — ECMWF AIFS Single 2.0

ECMWF's Artificial Intelligence Forecasting System, single (deterministic)
model, version 2.0. Given the state of the atmosphere at two times six hours
apart, it forecasts the global state six hours ahead, and is rolled forward
autoregressively to produce a 15-day forecast on the N320 grid (~31 km, 14
pressure levels). ECMWF runs it operationally four times a day.

Architecturally it is a graph neural network encoder and decoder around a
sliding-window transformer processor. Version 2.0 added a wave component — the
first operational data-driven wave forecasts from ECMWF — plus a snow-cover
variable and a 10 hPa stratospheric level.

- Upstream: https://github.com/ecmwf/anemoi-inference (Apache-2.0), part of the
  [Anemoi](https://anemoi.readthedocs.io/) framework co-developed by ECMWF and
  European national met services.
- Weights: https://huggingface.co/ecmwf/aifs-single-2.0 — **CC-BY-4.0 and
  ungated**. No access request, no token, no non-commercial clause.
- Papers: [AIFS (2024)](https://arxiv.org/abs/2406.01465),
  [update (2025)](https://arxiv.org/abs/2509.18994),
  [surface ocean (2026)](https://arxiv.org/abs/2604.25559).

> **Status: experimental.** This is the lab's first atmospheric model and
> nothing here has been run against lab science yet. Treat the first forecast
> as a plumbing test, not a result.

## Image tag

`ghcr.io/bradleylab/aifs:latest` (also `:v1`, `:cu126-py312`)

## Contents

- `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04` (**Python 3.12** as the system
  interpreter) plus torch 2.7.0 and torch-geometric 2.6.1 from PyPI.
- flash-attn 2.8.3, installed from ECMWF's prebuilt wheel rather than compiled.
- The anemoi inference stack pinned to the versions in the checkpoint repo's
  own `pyproject.toml` — `anemoi-inference[huggingface]==0.8.3`,
  `anemoi-models==0.9.3`, `anemoi-graphs==0.6.4`,
  `anemoi-transform==0.1.16.post2`, `anemoi-utils==0.4.35.post3`,
  `anemoi-plugins-ecmwf-inference[opendata]==0.2.1`.
- The earthkit / ecCodes data layer: `earthkit-data==0.19.3`,
  `earthkit-regrid==0.5.1`, `eccodes==2.47.0`, `ecmwf-opendata==0.3.29`. The
  ecCodes binary library arrives as a wheel (`eccodeslib`, via `mir-python`),
  not from apt.
- `HF_HOME=/root/.cache/huggingface` and `XDG_CACHE_HOME=/root/.cache` —
  override at runtime so weights and interpolation matrices persist.
- A CLI (`anemoi-inference run <config>.yaml`) *and* a Python API. Both are
  first-class; see below.

Deliberately **not** included: `anemoi-datasets` (the zarr training-dataset
reader — not on the Open Data or GRIB input paths) and `earthkit-plots` (which
drags in cartopy and fiona for the demo notebook's maps). Plot forecasts
outside the container, or add them to a downstream image.

## Checkpoint, inputs, and GPU requirements

Nothing is baked into the image. The checkpoint downloads from Hugging Face on
first run.

| Artifact | Source | Size |
|---|---|---|
| `aifs-single-mse-2.0.ckpt` | `ecmwf/aifs-single-2.0` | 0.99 GB |
| Full repo snapshot (14 files, includes `lsm.grib` static field and `inference.yaml`) | same | 1.00 GB |
| Initial conditions, one forecast | ECMWF Open Data | a few hundred MB per run |
| earthkit-regrid 0.25° → N320 matrices | `sites.ecmwf.int` | fetched once, cached |

| Resource | Requirement |
|---|---|
| GPU architecture | **Ampere or newer** (compute capability ≥ 8.0) — flash-attn does not support older cards. C2's H100s are 9.0. |
| GPU memory | Upstream's demo notebook runs on a Colab L4 (24 GB), so an 80 GB H100 is ample. Peak usage on H100 is **unmeasured here** — do not quote a number until someone has one. |
| Host memory | `--mem=64G` is the lab's default for these jobs and has not been shown insufficient. |
| Wall time | Unmeasured. A 12-hour lead time (2 model steps) is the cheapest useful test; 15 days is 60 steps. |

If a forecast runs out of GPU memory, the supported knobs are
`ANEMOI_INFERENCE_NUM_CHUNKS` (chunks the encoder/decoder mapper; upstream
suggests `16`) and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. Both
trade speed for memory and neither changes the forecast.

## Where initial conditions come from

This is the practical reason to prefer AIFS over the other weather foundation
models: **it runs off a free, public feed.** ECMWF publishes its operational
0.25° analyses and forecasts as [Open Data](https://www.ecmwf.int/en/forecasts/datasets/open-data)
under CC-BY-4.0, with no registration and no paid subscription. ERA5 reanalysis
(Copernicus, free with a CDS account) is the other route, and is what you want
for hindcasts of past events.

Two ways to get Open Data initial conditions in:

**1. Let anemoi fetch them.** The checkpoint repo ships an `inference.yaml`
whose `input: opendata` is served by `anemoi-plugins-ecmwf-inference[opendata]`.
That plugin drives the `ecmwf-opendata` client, retrieves both required times
(t−6h and t0), and regrids 0.25° → N320 via `earthkit-regrid`. Nothing to
write.

**2. Build the state yourself.** Upstream's
[`run_AIFS_v2.0.ipynb`](https://huggingface.co/ecmwf/aifs-single-2.0/blob/main/run_AIFS_v2.0.ipynb)
does it explicitly with `earthkit.data.from_source("ecmwf-open-data", ...)`,
which is the route to take when initial conditions come from somewhere other
than Open Data. It is also a readable specification of exactly which fields the
model expects, including the transformations that are easy to get wrong:
geopotential height must be multiplied by 9.80665 to give geopotential, mean
wave direction is split into `cos_mwd`/`sin_mwd`, soil fields are renamed
(`sot_1`→`stl1`, `vsw_1`→`swvl1`, …), and snow depth and soil moisture are set
to NaN over sea via `lsm.grib`.

Note the fields come from Open Data on a 0.25° regular grid and are regridded
to N320, whereas ECMWF's operational runs go straight from the native O1280
analysis to N320. Forecasts from this container will therefore differ slightly
from ECMWF's published AIFS output. That is expected, and it is a reason not to
present a container forecast as "the operational AIFS forecast".

**Both routes need network access at run time.** Plan for that on Compute2 —
see pre-staging below.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node (point enroot's scratch dirs off the 50 GB home):

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+aifs+v1.sqsh 'docker://ghcr.io#bradleylab/aifs:v1'
```

Submit a single-H100 job, mounting a cache directory so weights and
interpolation matrices are fetched once:

```bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+aifs+v1.sqsh \
     --container-mounts=/storage3/fs1/alexander.s.bradley/Active/model_cache/aifs:/root/.cache,/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     --container-workdir=/scratch2/fs1/alexander.s.bradley \
     bash -lc 'export PYTHONNOUSERSITE=1; anemoi-inference run /scratch2/fs1/alexander.s.bradley/aifs/inference.yaml'
```

Two lines there are lab-standard and easy to drop:

- `PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts `$HOME` into
  the container, so a stray `pip install --user` on the login node would
  otherwise shadow the container's site-packages. Given how tightly pinned this
  image is, a shadowed `anemoi-models` is exactly the failure that would be
  hardest to diagnose.
- The mount is `/root/.cache` in full, not just `/root/.cache/huggingface`.
  Three separate caches matter here — Hugging Face for the checkpoint,
  earthkit-regrid for the interpolation matrices, earthkit-data for retrieved
  GRIB — and they all sit under it.

### Pre-staging to Storage3

The cache goes on **Storage3, not scratch**. It is only ~1 GB, it is the same
bytes for every job, and scratch is purgeable — re-downloading a checkpoint
because scratch was swept is a self-inflicted outage. (Larger models in this
repo cache to `/scratch2/.../hf-cache` because they are tens of GB; AIFS is
small enough that the persistent tier is the better home.)

Stage it from a login node, which has network:

```bash
mkdir -p /storage3/fs1/alexander.s.bradley/Active/model_cache/aifs
HF_HOME=/storage3/fs1/alexander.s.bradley/Active/model_cache/aifs/huggingface \
  python -c "from huggingface_hub import snapshot_download; \
             snapshot_download('ecmwf/aifs-single-2.0')"
```

The interpolation matrices cannot be staged by name — they are selected by the
source and target grids at call time. Get them by running one short forecast on
a node that has network, with the cache mounted at its permanent home; they
land in `earthkit-regrid`'s cache and are reused thereafter.

Initial conditions cannot be pre-staged in any general sense, because a
forecast is defined by its start date. For a *reproducible* run, retrieve the
Open Data fields once to a GRIB file on Storage3 and feed that file as input,
rather than letting each run pull "latest" — a job that fetches `latest()` is
not reproducible by construction.

## Minimal inference

### Route 1 — the CLI, using the shipped config

The checkpoint repo's `inference.yaml` is a complete, working configuration:
Open Data input, the wave-direction and land-sea-mask filters, GRIB output. Get
it and the static field, then run:

```bash
huggingface-cli download ecmwf/aifs-single-2.0 --local-dir aifs-single-2.0
cd aifs-single-2.0
anemoi-inference run inference.yaml
```

`lead_time` and `date` are set on the command line or in the config; output
lands in `output.grib`.

### Route 2 — the Python API

```python
from anemoi.inference.runners.simple import SimpleRunner
from anemoi.inference.outputs.printer import print_state

runner = SimpleRunner({"huggingface": "ecmwf/aifs-single-2.0"})

# input_state = {"date": <datetime>, "fields": {<name>: <ndarray (2, npoints)>}}
# The two rows are t-6h and t0. See the upstream notebook for how each field is
# retrieved and transformed — that step is the fiddly part, not the model call.

states = []
for state in runner.run(input_state=input_state, lead_time=12):   # hours
    states.append(state)
    print_state(state)
```

Each yielded `state` carries `date`, `latitudes`, `longitudes`, and a `fields`
dict of 1-D arrays on the N320 grid.

## Licensing

The weights are **CC-BY-4.0** — unusually permissive for a forecasting
foundation model. There is no non-commercial clause, no gate, no acceptable-use
agreement, and no access request. The obligation is attribution: cite Lang et
al. 2024 (arXiv:2406.01465) and say the forecast came from AIFS Single v2.

The code is Apache-2.0 throughout: `anemoi-inference` and the rest of the
anemoi stack, and the demonstration notebook and scripts in the checkpoint
repo. ECMWF Open Data, the input feed, is CC-BY-4.0 as well.

So the whole path — model, code, and input data — is redistributable and
publishable with attribution alone. Worth knowing, because the comparable
models are not: several are non-commercial-only or gated, and most of the
paid-feed models require a data subscription that AIFS-on-Open-Data does not.

## Build notes / caveats

- **`constraints.txt` is what makes the build terminate, and it is not
  optional.** The top-level pins alone gave pip nothing to steer by, so it
  searched: mirlib, then contourpy, then cftime, each walked downwards one
  release at a time, and finally cfunits, where it reached the 3.3.2 sdist
  seventeen minutes in and died — that sdist's `setup.py` no longer runs under
  current setuptools. Nothing in the pin set actually conflicts; uv resolves it
  in seconds, and its answer matches the checkpoint repo's own `uv.lock` on
  cfunits (3.3.7), eccodeslib (2.46.2.19) and mir-python (1.28.1.19). The file
  is that resolution, passed as `pip install -c`, so pip has exactly one
  candidate per package and no search to do. Its header carries the `uv pip
  compile` line that regenerates it; regenerate rather than hand-edit, and do
  it whenever a top-level pin moves.

- **The pins are old on purpose.** anemoi-inference is at 0.11.2 and
  anemoi-models at 0.18.0 on PyPI (2026-08-18); this image installs 0.8.3 and
  0.9.3. An anemoi checkpoint is a pickled torch object whose classes come from
  the installed `anemoi.models`, so the loadable versions are the ones ECMWF
  shipped with this checkpoint — the set enumerated in the checkpoint repo's
  `pyproject.toml`, reproduced exactly here. Upgrading them without a
  checkpoint change is not "keeping current"; it risks an unpickling error at
  best and a silently different model at worst. Move the pins in step with the
  checkpoint, and only after a GPU load test.

- **One pin is load-bearing beyond reproducibility: `earthkit-data<1`.**
  anemoi-inference 0.8.3 requires only `earthkit-data>=0.12.4`, with no
  ceiling, and earthkit-data has since gone to 1.x. An unpinned build today
  would therefore install 1.1.1 against a runner released before it existed.
  Later anemoi-inference versions added the cap; 0.8.3 predates it, which is
  why upstream's own `pyproject.toml` states `earthkit-data<1` by hand and why
  this image pins `==0.19.3`. Do not relax it while the runner stays at 0.8.3.

- **Python 3.12 is a hard pin, set by flash-attn.** The prebuilt flash-attn
  wheel ECMWF publishes is `cp312`-only. Compiling flash-attn from source
  instead is a multi-hour build that would not survive CI. Ubuntu 24.04 ships
  3.12 as its system interpreter; the Dockerfile asserts the version rather
  than trusting the base tag.

- **Not an NGC base — but not for the reason `esm` isn't.** `esm` had to leave
  NGC because NGC's global pip constraint (`PIP_CONSTRAINT=/etc/pip/constraint.txt`,
  numpy==1.26.4) conflicted with its numpy 2.x requirement. Here the opposite
  holds: `anemoi-graphs==0.6.4` requires `numpy<2,>=1.26`, so NGC's constraint
  would have been *satisfied*. What rules NGC out is torch. The flash-attn
  wheel is tagged `cu12torch2.7cxx11abiFALSE`, i.e. compiled against the PyPI
  torch 2.7.0 build; NGC 25.04 ships NVIDIA's own `torch 2.7.0a0`, and a
  flash-attn binary built for a different torch is an ABI mismatch that
  surfaces as an undefined-symbol ImportError when the first attention layer is
  constructed. torch 2.7.0's PyPI wheel bundles CUDA 12.6, hence the 12.6.3
  base. **Do not "fix" a future flash-attn ImportError by clearing a pip
  constraint** — that is the wrong lever for this image.

- **`torch.load` is fine here, and that was checked, not assumed.** torch 2.6
  flipped `torch.load`'s default to `weights_only=True`, which breaks pickled
  model objects. `anemoi-inference` 0.8.3 passes `weights_only=False`
  explicitly when loading the checkpoint, so the 2.7.0 pin is safe. If the
  anemoi pin ever moves, re-check that line before assuming it still holds.

- **UDUNITS-2 needs two apt packages, not one.** `cfunits`, a hard dependency
  of `anemoi-transform`, loads `libudunits2.so.0` *and* reads its unit
  definitions from `/usr/share/xml/udunits/udunits2.xml` at import time. On
  Ubuntu those ship in `libudunits2-0` and `libudunits2-data` respectively.
  Install only the first and the failure is an import error deep in a
  pre-processor filter, not at build. The smoke test imports `cfunits` so a
  regression fails the build.

- **ecCodes comes from a wheel, transitively.** The `eccodes` Python package
  finds its binary library via `findlibs`; the library itself arrives as
  `eccodeslib`, pulled in by `mir-python`, pulled in by
  `anemoi-plugins-ecmwf-inference[opendata]`. So the GRIB stack works with no
  apt `libeccodes` — but it also means dropping the `[opendata]` extra would
  silently break GRIB reading. The smoke test imports `eccodes` for that
  reason.

- **Smoke test is offline and CPU-side, by design.** It imports the anemoi
  stack, reaches the model class the checkpoint unpickles into
  (`AnemoiModelEncProcDec`), and asserts torch is a CUDA build, numpy is still
  1.x, and flash-attn clears anemoi's own ≥2.6.0 gate. It does *not* import
  `flash_attn` — the extension dlopens the CUDA runtime and the build runner
  has no GPU — and it fetches no weights. A real load test has to run on an
  H100; see `SMOKE.md`.

- **Forecasts are not bitwise reproducible on GPU.** Upstream notes this
  explicitly. Determinism can be forced with
  `CUBLAS_WORKSPACE_CONFIG=:4096:8`, `torch.backends.cudnn.deterministic=True`,
  and `torch.use_deterministic_algorithms(True)`, at a significant runtime
  cost. If a forecast is going into a manuscript, decide which side of that
  trade you are on *before* running it, and record the choice.

- **Flash attention vs SDPA is not a free switch.** anemoi-models can fall back
  to PyTorch's scaled-dot-product attention, which is what makes CPU inference
  conceivable — but it uses far more memory
  ([ecmwf/anemoi-inference#119](https://github.com/ecmwf/anemoi-inference/issues/119))
  and is a different numerical path. This image ships flash-attn so the default
  path is the one the model was trained and validated with.
