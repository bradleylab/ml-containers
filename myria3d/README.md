# myria3d — FRACTAL Lidar HD semantic segmentation (H100 container)

**Semantic segmentation of airborne lidar (ALS) point clouds into seven
classes**: other, ground, vegetation, building, water, bridge, permanent
structure. RandLA-Net, trained by IGN (Institut national de l'information
géographique et forestière) on the FRACTAL benchmark built from the French
Lidar HD programme.

This is the repo's first *general-purpose* ALS segmenter. Everything else
here that touches point clouds is forestry-specific — tree instances
(TreeLearn, ForAINet, ForestFormer3D), leaf vs wood (PointsToWood, FSCT),
crowns (AMS3D). None of them produce a ground / building / water / bridge
label, which is what a landscape-scale ALS tile usually needs first.

- Upstream code: https://github.com/IGNF/myria3d — BSD-3-Clause, v3.9.0
- Weights: https://huggingface.co/IGNF/FRACTAL-LidarHD_7cl_randlanet —
  Etalab Open Licence 2.0, ungated, 13 MB
- Architecture: RandLA-Net as implemented in myria3d
  (`myria3d.models.modules.pyg_randla_net.PyGRandLANet`)
- Training data: 80,000 patches of 50 × 50 m (200 km²) from FRACTAL, plus
  10,000 validation patches (25 km²). FRACTAL samples 250 km² from a
  17,440 km² area across five spatial domains in southern metropolitan
  France.

> ## Read this before using the output for anything scientific
>
> **The model was trained on French Lidar HD tiles at ~40 points/m²,
> colorized with RGB + near-infrared from IGN's BD ORTHO. Transfer to US
> 3DEP QL1/QL2, to UAV lidar, or to TLS is UNMEASURED.** Nobody in this
> lab has validated it outside its training domain, and the container does
> not make that problem go away. Validate against hand-labelled points on
> your own data before any result from this image enters a figure, a
> table, or a manuscript.
>
> Two specific failure modes are worth knowing about up front, because
> they are silent — the model will happily emit confident labels either
> way:
>
> 1. **Six of the nine input features come from colour.** The feature
>    vector is `Intensity, ReturnNumber, NumberOfReturns, Red, Green,
>    Blue, Infrared, rgb_avg, ndvi`
>    (`myria3d/pctl/points_pre_transform/lidar_hd.py`). When a LAS has no
>    RGB or NIR dimensions, myria3d **fills them with zeros and carries
>    on**, printing only `Color channel Red not found. Creating fake Red
>    filled with 0.` Uncolorized 3DEP tiles therefore run at two-thirds of
>    the input signal missing, with no error. Colorize first, or expect
>    the vegetation / water / permanent-structure classes to degrade the
>    most.
> 2. **Colour is normalized by 255 × 256 = 65,280**, i.e. 16-bit. A tile
>    whose RGB was written as 8-bit values (0–255) in 16-bit LAS fields
>    comes out ~256× too dark after normalization, again without an error.
>    Check the actual value range in your LAS before inference.
>
> The published metrics themselves show how much domain matters: 77.5 mIoU
> on the FRACTAL test split, but 60.8 mIoU on IGN's held-out `eval67` set —
> a 17-point drop *within France*, on the sensor the model was trained for.

## Image tag

`ghcr.io/bradleylab/myria3d:latest` (also `:v1`, `:torch2.4-cu124`)

## Contents

- `condaforge/miniforge3:26.1.1-3` base, with a conda-forge env at
  `/opt/env` (Python 3.12) carrying **PDAL 2.10 + python-pdal + GDAL**.
  Those are hard imports in myria3d and have no PyPI wheels — see the
  Dockerfile header for why this is not an NGC or `nvidia/cuda` base.
- **PyTorch 2.4.1 + CUDA 12.4** from the PyPI cu124 wheels (native H100
  sm_90), with the PyG companions pinned to the matching prebuilt wheels:
  `torch_scatter 2.1.2`, `torch_cluster 1.6.3`, `torch_sparse 0.6.18`,
  all `+pt24cu124 / cp312`. Nothing compiles at build time.
- `torch_geometric 2.6.1`, `pytorch-lightning 2.4.0`, `torchmetrics 1.4.3`,
  `hydra-core 1.3.2`, `laspy[lazrs] 2.5.4`, `ign-pdal-tools 1.16.0`.
- myria3d source at `/opt/myria3d`, pinned to commit
  `fcc22ac32f04380d971f99d6827edc4f0e376e58` (v3.9.0), plus the FRACTAL
  inference config at
  `/opt/myria3d/trained_model_assets/FRACTAL-LidarHD_7cl_randlanet-inference-Myria3DV3.8.yaml`.
- No wrapper CLI. Inference is upstream's `run.py` under hydra.

**Call the interpreter by absolute path — `/opt/env/bin/python`.** The
conda-forge base image carries its own `base` environment, and a login
shell (`bash -lc`, which every SLURM recipe here uses) can activate it and
put `/opt/conda/bin` ahead of the image env on `PATH`. Every example below
does this; a bare `python` may resolve to the base interpreter, which has
none of the packages.

## Weights

**Not baked into the image.** Upstream commits the checkpoint into its own
git repo; the Dockerfile deletes it after cloning, so the published image
carries BSD-3-Clause code only and the Etalab-licensed artifact is staged
separately.

Fetch once and reuse across jobs. `HF_HOME` defaults to
`/root/.cache/huggingface` inside the image and should be redirected to a
mounted directory so the cache survives the container:

```bash
srun --container-image=/storage3/fs1/alexander.s.bradley/Active/c2_jobs/bradleylab+myria3d+v1.sqsh \
     --container-mounts=/scratch2/fs1/alexander.s.bradley:/scratch2/fs1/alexander.s.bradley \
     bash -lc '
        export HF_HOME=/scratch2/fs1/alexander.s.bradley/hf-cache
        /opt/env/bin/hf download IGNF/FRACTAL-LidarHD_7cl_randlanet \
          FRACTAL-LidarHD_7cl_randlanet.ckpt \
          --revision 6ef0c46e11ab7fa9d20f3d9e39986c46dbd3814e
     '
```

`hf download` prints the resolved absolute path on stdout — that is what
goes into `predict.ckpt_path`. Run it inside the container rather than on
the login node: `hf` ships in the image and need not exist on Compute2.

The expected checkpoint is 13,631,452 bytes with

```
sha256  58baca3fbc00af2fa4af2a26cea345c08decbbba0215d21d6640c412a42e8cd1
```

which is also recorded in the image as `$MYRIA3D_CKPT_SHA256`. The copy
committed in the upstream git repo is byte-identical to the Hugging Face
copy (verified 2026-08-18), so either source works.

### Pre-staging on Storage3

Weights on `/scratch2` are subject to purge policy; the durable copy
belongs on Storage3 alongside the `.sqsh`:

```bash
STORAGE3=/storage3/fs1/alexander.s.bradley
SCRATCH=/scratch2/fs1/alexander.s.bradley
mkdir -p $STORAGE3/Active/weights/myria3d

srun --container-image=$STORAGE3/Active/c2_jobs/bradleylab+myria3d+v1.sqsh \
     --container-mounts=$SCRATCH:$SCRATCH,$STORAGE3:$STORAGE3 \
     bash -lc "
        export HF_HOME=$SCRATCH/hf-cache
        /opt/env/bin/hf download IGNF/FRACTAL-LidarHD_7cl_randlanet \
          FRACTAL-LidarHD_7cl_randlanet.ckpt \
          --revision 6ef0c46e11ab7fa9d20f3d9e39986c46dbd3814e \
          --local-dir $STORAGE3/Active/weights/myria3d
     "
```

`--local-dir` gives a flat, predictable path instead of the HF cache's
`hub/models--.../snapshots/<rev>/` layout. Mount that directory into the
job and point `predict.ckpt_path` at it. The checkpoint is 13 MB, so this
costs nothing to keep.

## Resource requirements

RandLA-Net at inference is small on the GPU and heavy on host RAM — the
`Interpolator` accumulates every subtile's logits on CPU, then allocates a
dense `(n_points, 7)` float32 buffer for the whole tile before writing the
output LAS (`myria3d/models/interpolation.py`).

| Resource | Value | Where it comes from |
|---|---|---|
| Checkpoint | 13 MB | HF repo file size |
| GPU memory | **not measured** | no lab run yet; start with one H100 and watch `nvidia-smi` |
| GPUs per job | 1 | inference is single-device (`predict.gpus=1`) |
| Host RAM, 1 km² tile at 40 pts/m² | **~10–16 GB, estimated** | 40 M points: dense logits 40e6 × 7 × 4 B ≈ 1.1 GB, plus the full PDAL point array with the added probability / entropy / predicted-class dimensions |
| Host RAM to request | 64 GB | lab default for this repo's point-cloud jobs; comfortably above the estimate |
| Shared memory | ≥ 2 GB | upstream's docker guidance for 1 km² Lidar HD clouds (dataloader workers) |
| `datamodule.batch_size` | 10 default, 50 suggested | config default; upstream docs say "N=50 works well, the larger the faster" |
| Wall time | **not measured** | see `SMOKE.md` for the first timing run |

Everything marked *not measured* stays that way until someone runs it here
and edits this table. Do not quote an estimate as a measurement.

## Running on Compute2 (Pyxis/enroot)

Import once on a login node, with enroot's scratch dirs pointed off the
50 GB home:

```bash
cd /storage3/fs1/alexander.s.bradley/Active/c2_jobs
XDG_CACHE_HOME=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_CACHE_PATH=/scratch2/fs1/alexander.s.bradley/enroot_cache \
ENROOT_RUNTIME_PATH=/scratch2/fs1/alexander.s.bradley/enroot_runtime \
  enroot import -o bradleylab+myria3d+v1.sqsh \
    'docker://ghcr.io#bradleylab/myria3d:v1'
```

Then submit a single-H100 job:

```bash
#!/bin/bash
#SBATCH -A compute2-alexander.s.bradley
#SBATCH -p general-gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH -J myria3d-predict

SCRATCH=/scratch2/fs1/alexander.s.bradley
STORAGE3=/storage3/fs1/alexander.s.bradley
CKPT=$STORAGE3/Active/weights/myria3d/FRACTAL-LidarHD_7cl_randlanet.ckpt

srun --container-image=$STORAGE3/Active/c2_jobs/bradleylab+myria3d+v1.sqsh \
     --container-mounts=$SCRATCH:$SCRATCH,$STORAGE3:$STORAGE3 \
     bash -lc "
        export PYTHONNOUSERSITE=1
        mkdir -p $SCRATCH/myria3d_runs && cd $SCRATCH/myria3d_runs
        /opt/env/bin/python /opt/myria3d/run.py \
          task.task_name=predict \
          predict.src_las=$SCRATCH/myria3d_in/tile.laz \
          predict.output_dir=$SCRATCH/myria3d_out \
          predict.ckpt_path=$CKPT \
          predict.gpus=1 \
          datamodule.batch_size=50 \
          datamodule.epsg=6344
     "
```

Three lines in there are easy to drop and each one bites:

- **`export PYTHONNOUSERSITE=1`** — enroot bind-mounts `$HOME` into the
  container, so a stray `pip install --user` on the login node would
  otherwise shadow the image's site-packages. (The image also sets this,
  but keep it in the job script; it survives shell re-execs and matches
  every other recipe here.)
- **`cd` into a writable directory first.** Hydra changes the working
  directory and writes an `outputs/<date>/<time>/` run directory under
  wherever the job started. `/opt/myria3d` is inside the read-only image.
  All paths you pass must be absolute for the same reason.
- **`datamodule.epsg=`** — required whenever the LAS has no EPSG in its
  metadata. The config defaults to 2154 (RGF93 / Lambert-93, i.e. France).
  For Missouri work that is `6344` (NAD83(2011) / UTM 15N). Getting this
  wrong does not error; it silently mislocates the tile.

## Minimal prediction, and what comes out

```bash
/opt/env/bin/python /opt/myria3d/run.py \
  task.task_name=predict \
  predict.src_las=/abs/path/tile.laz \
  predict.output_dir=/abs/path/out \
  predict.ckpt_path=/abs/path/FRACTAL-LidarHD_7cl_randlanet.ckpt \
  predict.gpus=1 \
  datamodule.batch_size=50
```

The output is a new LAS/LAZ in `predict.output_dir` sharing the input
basename, with the original points plus added dimensions:

| Dimension | Contents |
|---|---|
| `PredictedClassification` | argmax class, in LAS classification codes |
| `entropy` | Shannon entropy of the class probabilities — a rough, and only rough, uncertainty proxy |
| one dimension per class | probability, controlled by `predict.interpolator.probas_to_save` |

Class codes follow the LAS convention used by Lidar HD:

| Code | Class |
|---|---|
| 1 | unclassified / other |
| 2 | ground |
| 5 | vegetation |
| 6 | building |
| 9 | water |
| 17 | bridge |
| 64 | permanent structure (`lasting_above`) |

Options worth knowing:

- `predict.src_las` accepts a glob (`'/abs/path/*.laz'`), predicting on
  each match in turn. Quote it so the shell does not expand it first.
- `predict.interpolator.probas_to_save=[building,ground]` limits the
  probability dimensions written (no spaces). The FRACTAL config ships
  with exactly that pair; `all` writes seven.
- `predict.subtile_overlap=25` runs a 25 m sliding window over the 50 m
  receptive fields, smoothing object-level predictions. It multiplies
  inference time by roughly four.
- `datamodule.tile_width` defaults to 1000 (m). Set it to match your tile
  if you are not on 1 km² inputs.
- Points carrying classification code 65 are treated as acquisition
  artifacts and dropped from inference; they keep their original class and
  get null probabilities in the output. Map additional codes into 65 via
  `dataset_description.classification_preprocessing_dict` if you have
  known-bad classes.

## Reported accuracy

From the model card (IGN's own evaluation — not reproduced here):

| Class | IoU, FRACTAL test | IoU, `eval67` held-out |
|---|---|---|
| Other | 47.5 | 22.3 |
| Ground | 91.9 | 90.7 |
| Vegetation | 93.8 | 91.4 |
| Building | 90.4 | 86.9 |
| Water | 90.1 | 77.7 |
| Bridge | 65.2 | 38.0 |
| Permanent structure | 63.5 | 16.6 |
| **mIoU** | **77.5** | **60.8** |

Ground, vegetation and building hold up across both sets. Bridge and
permanent structure do not, and "other" is weak everywhere — treat the
three rare classes as indicative, not as measurements, even on French
data. The model card reports no overall accuracy.

## Licensing

Two different licences, and they stay separate on purpose:

- **Code** — myria3d is BSD-3-Clause. That is what the image itself ships
  and what the OCI `licenses` label records.
- **Weights** — the FRACTAL checkpoint is under
  [Etalab Open Licence 2.0](https://www.etalab.gouv.fr/licence-ouverte-open-licence/),
  a permissive, attribution-only licence. Attribution is a condition of
  use: credit IGN and name the source when publishing derived results.
  Because the checkpoint is fetched at run time rather than baked, the
  published image does not redistribute it.

The FRACTAL dataset itself derives from the French Lidar HD programme
(https://geoservices.ign.fr/lidarhd).

## Known limitations

- **Domain.** French Lidar HD, ~40 pts/m², RGB + NIR colorized from BD
  ORTHO, five southern-France spatial domains. See the boxed caveat at the
  top; nothing else in this README supersedes it.
- **ALS only.** IGN state the model is designed for aerial lidar and note
  that mobile and terrestrial clouds have different occlusion, density and
  scan-angle characteristics. For TLS/MLS use PointsToWood, TreeLearn or
  FSCT.
- **Semantic, not instance.** There are no per-object identities in the
  output — no individual buildings, no individual trees. Pair with
  TreeLearn or ForestFormer3D if instances are needed.
- **No turnkey validation.** The image ships inference only. Comparing
  `PredictedClassification` against a hand-labelled subset is your job and
  is the gate before any scientific use.

## Build notes

- **myria3d is installed editable, and that is not a style choice.**
  Upstream's `pyproject.toml` declares `packages = ["myria3d"]`, naming the
  top-level package and nothing beneath it. A regular install therefore copies
  four files into site-packages — `__init__.py`, `_version.py`, `predict.py`,
  `train.py` — and silently leaves `myria3d.models`, `myria3d.pctl` and the
  rest of the source tree out. Package metadata reads correctly and
  `import myria3d` succeeds, so the omission only shows up an import later, as
  `ModuleNotFoundError: No module named 'myria3d.models'`. Upstream never meets
  this because their own Dockerfile does not install the package at all; it
  copies the repo in and runs from that directory. `pip install -e` points the
  import machinery at `/opt/myria3d`, so the whole tree imports from any
  working directory while the metadata the smoke test reads still registers.
  If a future edit "tidies" this back to a plain install, the build fails at
  smoke test 1 — which is where it should fail.
- **Weights removal is deliberate, not an oversight.** Upstream commits
  the checkpoint at `trained_model_assets/`; the Dockerfile deletes it
  after cloning. If a future rebuild finds no `.ckpt` there, upstream may
  have moved it — the build-time smoke test asserts the directory is
  checkpoint-free, so a silently re-baked weight fails the build rather
  than shipping.
- **Do not bump torch past 2.5 casually.** torch 2.6 changed
  `torch.load`'s default to `weights_only=True`, which is the classic way
  to break `LightningModule.load_from_checkpoint` on checkpoints holding
  non-tensor hyperparameters. The FRACTAL checkpoint is one. Any bump
  needs a real `Model.load_from_checkpoint` test on an H100, not just a
  green build.
- **The PyG companion wheels are the fragile pin.** `torch_scatter`,
  `torch_cluster` and `torch_sparse` are compiled against one exact
  (torch minor, CUDA minor, CPython) triple. Changing torch, CUDA or the
  Python version means re-deriving all three from
  `https://data.pyg.org/whl/torch-<minor>+cu<ver>.html` — check the index
  actually lists a `cp3XX` wheel before editing the Dockerfile.
- **`anaconda` channel dropped.** Upstream's `environment.yml` pulls
  `mkl` from Anaconda's default channel, which carries commercial terms.
  This image installs from conda-forge only.
- **Build-time smoke tests are offline and cannot prove the GPU path.**
  They construct a RandLA-Net on CPU at the FRACTAL geometry (9 features,
  7 classes), import the PDAL/GDAL stack, and compose the hydra predict
  config — upstream's own build check. An sm_90 forward pass and a real
  checkpoint load happen on Compute2; see `SMOKE.md`.
