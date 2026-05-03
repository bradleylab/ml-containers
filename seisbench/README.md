# seisbench

Seismic phase picking from waveform data via SeisBench (Woollam et al.
2022). Bundles maintained reimplementations of PhaseNet (Zhu & Beroza
2019) and EQTransformer (Mousavi et al. 2020) plus the SeisBench model
zoo (zero-shot pretrained weights for many regions and networks).

Laptop-runnable; the container provides a reproducible env (`pip
install seisbench` also works). The Compute2 path is for
continental-scale catalogs where many station-days run in parallel.

## Image tag

`ghcr.io/bradleylab/seisbench:v2` (also `:latest`, `:torch2.5-cpu`).
`:v1` retained for rollback (amd64-only).

**Multi-arch.** v2 ships both `linux/amd64` and `linux/arm64` so
Apple-Silicon Mac users get native-arch images without qemu
emulation.

## Stack

- Base: `python:3.11-slim`
- PyTorch 2.5.1 (CPU wheels)
- `seisbench >= 0.7`
- `obspy >= 1.4` (waveform I/O via FDSN web services)
- `h5py`, `pandas`, `numpy` (transitive)

GPU is not required. SeisBench will use CUDA if torch sees one; on
this image torch is CPU-only by design (laptops are the primary
target). For dedicated GPU work, see `Dockerfile.cuda` if added later.

## Weights

Not baked. The first call to `Model.from_pretrained(...)` pulls from
the SeisBench model zoo (S3-hosted) into the cache directory set by
`SEISBENCH_CACHE_ROOT=/opt/seisbench-cache`. Bind-mount a persistent
host directory there to avoid re-downloading per job.

```bash
docker run --rm -it \
  -v "$PWD/seisbench-cache:/opt/seisbench-cache" \
  -v "$PWD/data:/data" \
  ghcr.io/bradleylab/seisbench:v2
```

(Each pretrained model is a few MB; first-run downloads are quick.)

## Inference

```python
import obspy
import seisbench.models as sbm

model = sbm.PhaseNet.from_pretrained("original")  # or "stead", "instance", etc.

st = obspy.read("/data/example.mseed")          # or use FDSN client
result = model.classify(st)
for p in result.picks:
    print(p.peak_time, p.peak_value, p.phase)
```

EQTransformer is the same shape:

```python
model = sbm.EQTransformer.from_pretrained("original")
result = model.classify(st)
```

See https://seisbench.readthedocs.io for the full API and the list of
available pretrained models per architecture.

## Inputs

- ObsPy `Stream` of seismic waveforms; 3-component data (Z/N/E or
  Z/1/2) preferred for EQTransformer, single-component acceptable for
  PhaseNet.
- Most pretrained models assume 100 Hz sampling; SeisBench resamples
  internally if the input differs.

## Run on Compute2

For continental or decadal catalogs, submit a CPU job array — the
model isn't the bottleneck, parallelism is. `general-cpu` partition
(no GPU needed):

```bash
sbatch -A compute2-alexander.s.bradley \
       -p general-cpu \
       --cpus-per-task=4 \
       --mem=8G \
       --time=04:00:00 \
       --array=0-99 \
       --wrap='srun \
         --container-image=/storage1/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+seisbench+v2.sqsh \
         --container-mounts=/scratch2/fs1/alexander.s.bradley/seisbench-cache:/opt/seisbench-cache,/scratch2/fs1/alexander.s.bradley/waveforms:/data \
         bash -lc "export PYTHONNOUSERSITE=1; python /scratch2/fs1/alexander.s.bradley/scripts/pick_station_day.py"'
```

`PYTHONNOUSERSITE=1` is required on Compute2 — enroot bind-mounts
`$HOME` into the container, so any stray `pip install --user` on the
login node would otherwise shadow the container's site-packages. See
`~/.claude/rules/research-infrastructure.md`.

## Limitations

- Pretrained models are zero-shot on new networks but not necessarily
  optimal — performance on a specific WUSTL or regional network may
  improve with transfer learning.
- The image is CPU-only. For real-time / streaming use cases a CUDA
  build would help; not on the roadmap until a use case lands.
