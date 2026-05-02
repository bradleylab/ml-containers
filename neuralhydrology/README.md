# neuralhydrology

LSTM rainfall-runoff and streamflow prediction via the NeuralHydrology
library (Kratzert, Klotz, Gauch et al.). Pretrained CAMELS checkpoints
are tiny (LSTMs for streamflow are trivially small by modern ML
standards); CPU inference is essentially instant.

Laptop-runnable; the container provides a reproducible env (`pip
install neuralhydrology` also works). The Compute2 path is for
continental-scale training across thousands of basins, ensemble runs,
or retraining on a new forcing product — *training* is the GPU-bound
case, not inference. This image is CPU-only by design and targets
inference; for training jobs on H100, build a CUDA variant.

## Image tag

`ghcr.io/bradleylab/neuralhydrology:v1` (also `:latest`,
`:torch2.5-cpu`)

## Stack

- Base: `python:3.11-slim`
- PyTorch 2.5.1 (CPU wheels)
- `neuralhydrology >= 1.13`
- `xarray`, `netcdf4`, `h5py`, `numba`, `pandas`, `scipy` (transitive)
- Console scripts: `nh-run`, `nh-schedule-runs`, `nh-results-ensemble`

## Weights

Not baked. NeuralHydrology checkpoints are produced by `nh-run train`
and live in a `run_dir/` containing `model_epochXXX.pt` and
`config.yml`. For inference, bind-mount the run dir alongside the
forcing data:

```bash
docker run --rm -it \
  -v "$PWD/run_dir:/work/run_dir" \
  -v "$PWD/data:/work/data" \
  ghcr.io/bradleylab/neuralhydrology:v1
```

Pretrained CAMELS checkpoints from published runs are linked from
the [NeuralHydrology research blog](https://neuralhydrology.github.io/).

## Inference

```bash
# Inside the container:
nh-run evaluate --run-dir /work/run_dir --epoch 30
```

Or in Python:

```python
from pathlib import Path
from neuralhydrology.evaluation.tester import start_evaluation
from neuralhydrology.utils.config import Config

cfg = Config(Path("/work/run_dir/config.yml"))
start_evaluation(cfg, Path("/work/run_dir"), epoch=30, period="test")
```

See https://neuralhydrology.readthedocs.io for the full API and
config reference.

## Inputs

- **Forcing data**: meteorological forcings + streamflow observations
  in CAMELS layout (or compatible). Daymet, NLDAS, ERA5 are common
  sources for new basins outside CAMELS.
- **Run dir**: directory holding the trained checkpoint and the
  `config.yml` produced by `nh-run train`.

CAMELS US (~870 MB) is small enough to keep on a laptop. For
continental-scale work, mount the dataset onto the container's
`/data` (or wherever `config.yml` references).

## Run on Compute2

Inference is CPU-trivial; submit to `general-cpu`. Training (a
separate use case requiring a CUDA variant) goes to `general-gpu`.

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=4 \
       --mem=8G \
       --time=02:00:00 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+neuralhydrology+v1.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/run_dir:/work/run_dir,/scratch2/fs1/alexander.s.bradley/camels:/work/data \
         bash -lc "export PYTHONNOUSERSITE=1; nh-run evaluate --run-dir /work/run_dir --epoch 30"'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — see
`~/.claude/rules/research-infrastructure.md`.

## Limitations

- CPU-only. Training jobs (continental-scale, ensembles) need a CUDA
  variant; not on the roadmap until a use case lands.
- Pretrained CAMELS LSTMs are zero-shot on new basins given
  appropriate forcing inputs but performance varies; cross-basin
  generalization is the active research area in the
  NeuralHydrology team's publications.
