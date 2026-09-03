# afwizard

Spatially adaptive ground-point filtering. Heidelberg Scientific Software
Center. **MIT, no model weights** — this is software, not a trained model.

## The problem it solves

One ground filter applied across a whole survey is wrong whenever the survey
covers varied terrain. A setting that works on open slope oversmooths under
dense vegetation; a setting tuned for vegetation leaves noise on bare ground.

AFwizard lets you segment the area into polygons and apply a **different filter
and parameter set per segment**, producing one coherent DTM plus the pipelines
and metadata that made it — so the classification is auditable and repeatable,
which matters for `METHODS.md`.

## Why the forest tools are not a substitute

`fsct`, `forainet`, `forestformer3d` and `myria3d` all classify ground, and all
were built around forest *structure*: the goal is a clean canopy / stem / ground
split, and the ground surface is a means to an end.

Archaeological prospection wants the opposite emphasis — the anthropogenic
micro-relief **on** the ground surface. Burial mounds, house platforms and
borrow pits are 20–50 cm of relief, and an aggressive filter removes them as
noise before anything downstream ever sees them.

Doneus, Höfle, Kempf, Daskalakis & Shinoto (2022), *Archaeological Prospection*
29(4):503–524 ([doi:10.1002/arp.1873](https://doi.org/10.1002/arp.1873), CC-BY)
is the paper behind this tool and frames the problem in exactly those terms.

## It is not automatic

**You supply the polygon layer.** AFwizard does not segment the landscape by
terrain or vegetation for you — that is the human-in-the-loop part, by design.
Budget for someone drawing polygons over the survey area, informed by canopy
density and slope.

## When it does and does not help

| Situation | Use it? |
|---|---|
| Raw or unclassified point cloud, varied terrain, hunting micro-relief | **Yes** — this is the case it exists for |
| Dense drone lidar over forest (e.g. Tyson) | **Yes** |
| Already-classified ground returns from a provider (e.g. USGS 3DEP) | **No** — the filtering has already happened. Its parameters were generic, but you cannot revisit that without the unclassified cloud |
| Uniform open terrain, one filter clearly works | No — a single PDAL/CSF pass is simpler |

If you hold the **unclassified** version of a provider's cloud, re-filtering
with archaeology-aware parameters can recover micro-relief that generic
classification smoothed away. That is worth checking before assuming a
delivered ground classification is good enough.

## PDAL only — the other backends are proprietary

`afwizard.pdal`, `afwizard.lastools` and `afwizard.opals` all import, but only
**PDAL** is open source and installed here. LAStools and OPALS are commercial
and need licences this lab does not hold; upstream ships a separate
`docker/proprietary.dockerfile` for sites that do.

`set_lastools_directory()` and `set_opals_directory()` will find nothing in this
image. That is expected, not broken.

## Does it run headless? Two phases — one does, one does not

**No for the exploration phase, yes for the application phase.** Upstream's own
CLI help states the split:

> This CLI is used once you have finished the interactive exploration work with
> the AFwizard Jupyter UI. The CLI takes your dataset and the segmentation file
> created in Jupyter and executes the ground point filtering on the entire
> dataset.

| Phase | Headless? | What happens |
|---|---|---|
| Segment the area, tune and assign filters | **No** — JupyterLab widgets, and the polygons are drawn by a person | produces a segmentation GeoJSON |
| Apply that segmentation to the full dataset | **Yes** — `afwizard` CLI or `apply_adaptive_pipeline` | writes filtered LAS/LAZ + GeoTIFF |

So a SLURM job can do the filtering, but something has to hand it a
segmentation GeoJSON first. That file is the artifact the interactive phase
exists to produce, and it is reusable — segment a survey area once, then apply
it headlessly to as many tiles or repeat flights as you like.

### The headless CLI

```bash
afwizard --dataset cloud.laz --dataset-crs EPSG:26915          --segmentation segments.geojson --segmentation-crs EPSG:26915          --output-dir /work/out --resolution 1.0 --compress
```

`--library` can be given multiple times to add filter-library locations.
`--resolution` is the GeoTIFF meshing resolution in metres (default 0.5).
`--opals-dir` and `--lastools-dir` exist but have nothing to point at in this
image — see below.

A fully scripted run without any Jupyter is possible if you hand-author the
segmentation GeoJSON, or reuse one. The format is documented upstream; the
interactive UI is a convenience for producing it, not the only way.

## Batch use

The interactive apps (`pipeline_tuning`, `select_best_pipeline`,
`select_pipeline_from_library`) are JupyterLab widgets — a workstation
activity, not a SLURM one. Use them to *create* a segmentation and pipeline
assignment, then apply it here:

```python
from afwizard.execute import apply_adaptive_pipeline

apply_adaptive_pipeline(
    dataset=...,        # afwizard.dataset.DataSet
    segmentation=...,   # afwizard.segmentation.load_segmentation(<geojson>)
    output_dir="/work/out",
)
```

`afwizard.filter.load_filter` and `afwizard.library` reach the
community-contributed pipelines from the `afwizard-library` package, installed
alongside.

## Tuning for an objective — `afwizard-tune`

The interactive slider workflow asks a person to look at a filtered surface and
judge whether it is right *for this site and this purpose*. That judgment is two
jobs: knowing what "right" means for the case, and checking a result against
it. `afwizard-tune` splits them. **The caller** — an agent, usually — decides
what right means and writes it down as an objective. **The tuner** runs a
parameter search and scores every candidate against that objective. It never
encodes a purpose of its own.

```
afwizard-tune list-criteria                       # the vocabulary
afwizard-tune tune --objective obj.yaml --outdir run/
afwizard-tune describe --outdir run/              # re-read without re-running
afwizard-tune wire --objective obj.yaml --outdir run/ --apply
```

### The criteria are the vocabulary

Each is one number from a filtered cloud, and carries the prose a caller needs
to decide whether to use it: what it measures, which direction is better, what
failure it catches, and what kind of task cares. `list-criteria` prints all of
it. Read that before writing an objective; the descriptions say which pairings
are safe and which are traps (maximizing coverage with no commission check is
satisfied by labelling everything ground).

The one that carries the archaeological concern is `structured_variance`, and
it is parameterized by scale precisely so it is not archaeology-specific:
spatially coherent relief at 2–10 m is a mound, at 1–3 m a grave depression, at
20–50 m a terrace. Same criterion; the scale is the caller's input. It is also
sign-agnostic, which is why `relief_sign_ratio` exists separately.

### The objective is a file the caller writes

```yaml
constraints:
  recall_vendor_ground: {min: 0.99}                 # never drop the vendor's ground
  nugget: {max_ratio_to_reference: 1.2}            # no more speckle than the vendor DEM
maximize:
  structured_variance: {scale_lo_m: 2.0, scale_hi_m: 10.0}
report: [coverage, added_points, height_above_reference_p99]
```

Constraints prune. The maximize target ranks the survivors. Report entries are
computed and returned but never influence the pick. Two complete examples ship
in `examples/`: one that preserves features, one that smooths for hydrology —
same tuner, opposite purposes.

### What comes back

`evaluations.csv` — every parameter set tried, every criterion, feasible or
not and why. `summary.json` — the pick per segment, its scores, and the rule
that chose it. The pick is the least important row: the table is what lets a
caller re-reason and re-pick without re-running.

### Handing the result to AFwizard

`wire` writes each pick as an AFwizard filter, puts the pipeline hash into a
copy of the segmentation, and with `--apply` runs `apply_adaptive_pipeline` so
the applied filtering carries its provenance trail. One trap it handles: AFwizard
hashes a pipeline's **metadata only**, so two filters with different settings
and the same title collide. The settings are written into the title.

### Why the search drives PDAL directly

Every criterion scores a classified point cloud regardless of who classified
it. During the search that is the PDAL CLI, one call per candidate — filter,
height above the reference DEM, DTM, and the per-point columns all come out of
a single pipeline in a few seconds. AFwizard is used at the end, for the record,
not in the loop.

### Worked run — Greenwood Cemetery, USGS 3DEP 2017, two canopy segments

947,352 points, vendor classes 1 / 2 / 7 only (no vegetation classes — class 1
holds everything the vendor did not commit to, including about half the true
ground). Objective: `examples/feature_preservation.yaml` at 2–10 m, a small
search of 6 + 4 per segment, 81 s on a laptop with the PDAL binary.

| Segment | Feasible | Pick (window / threshold / slope / scalar) | recovered relief | speckle added | 99th-pct height above vendor DEM | vendor ground kept |
|---|---|---|---|---|---|---|
| open | 9 / 9 | 12.0 / 0.75 / 0.24 / 0.93 | 0.00068 m² | 0.36× vendor | 0.42 m | 100% |
| wooded | 2 / 9 | 20.6 / 0.32 / 0.40 / 1.44 | 0.00250 m² | 0.44× vendor | 0.50 m | 100% |

Two things the table shows that a single number would hide:

- **The commission caps are doing the work in the wooded zone.** Under a loose
  cap every wooded candidate passed and the pick was the one adding the most
  speckle and the tallest height tail (0.59 m). Under the shipped caps it is
  the moderate setting, and it sits exactly at the height cap — a boundary,
  not a plateau. The 60-evaluation run will say which.
- **On this site the settings that recover the most coherent relief also leak
  the most vegetation.** Every candidate ranks the same way on all three. That
  is what the Pareto front in `summary.json` is for: the pick is one point on
  it, and a caller who knows the site can move along it.

Coverage came back 0.98–1.00 at 1 m because ~3 points/m² fills nearly every
1 m cell; it becomes a discriminating criterion at `dtm_resolution_m: 0.5`,
which is where the original "30% of cells" figure was measured. It is
resolution-dependent by design.

### The same objective at full size, on Compute2 through `ml-jobs`

`mljob run afwizard tune greenwood_feature_preservation.yaml` — 40 + 20
evaluations per segment, 116 total, 10 minutes on a `general-cpu` node, then
`wire --apply` handing the picks to AFwizard's own `apply_adaptive_pipeline`,
which wrote the filtered LAS and GeoTIFF with its provenance log.

| Segment | Feasible | Pick (window / threshold / slope / scalar) | recovered relief | speckle added | 99th-pct height | Pareto front |
|---|---|---|---|---|---|---|
| open | 58 / 58 | 16.0 / **0.80** / 0.21 / 1.42 | 0.00073 m² | 0.39× | 0.44 m | 42 rows |
| wooded | 39 / 58 | 25.8 / 0.30 / **0.40** / 1.41 | 0.00249 m² | 0.44× | 0.50 m | 26 rows |

Two bold values sit at the edge of the search space — the open threshold at
its maximum 0.80, the wooded slope at its maximum 0.40. The optimum wants to
go further than the box allowed. That is the caller's knob (`search_space`),
and the table says so rather than hiding it.

**Against the hand-chosen settings the project started with** (open
8 / 0.20 / 0.12; wooded 18 / 0.45 / 0.18), scored on the same criteria:

| Segment | Hand-chosen recovered relief | Tuned | Hand-chosen speckle / tail | Tuned |
|---|---|---|---|---|
| open | 0.00054 m² | 0.00073 m² (+34%) | 0.28× / 0.26 m | 0.39× / 0.44 m |
| wooded | 0.00239 m² | 0.00249 m² (+4%) | 0.44× / 0.50 m | 0.44× / 0.50 m |

The wooded guess was already near-optimal under this objective. The open guess
was cleaner and recovered a third less; both are on the front. Neither result
is a surprise given the calibration warning above — and both are the kind of
answer the tuner exists to give, rather than a single number that hides it.

### What running it exposed that reading the code did not

Three criteria were wrong in their first form, and each looked plausible until
the numbers came back. They are recorded here so nobody re-derives them.

1. **Structured variance measured the hill, not the lumps.** On a site with
   24 m of relief, the raw semivariogram at 2–10 m is topography; candidate
   filters differed by 0.1%. Detrending fixes it — but detrending by filling
   gaps with a constant first puts a fake step at every polygon edge, and the
   open zone is 115 fragments. The moving mean has to be NaN-aware.
2. **The nugget extrapolated to zero.** On an interpolated surface the
   variogram grows faster than linearly, so the lag-0 intercept goes negative
   and floors, and a ratio constraint against it is vacuous. It is now the
   adjacent-cell semivariance, no extrapolation.
3. **Coverage read 1.0 for every candidate**, twice, for two different
   reasons: first because the rasterizer covers the crop's bounding box and
   the polygon fills a few percent of it; then, masked, because the IDW window
   fills every cell anyway. It is now counted from the points.

## Not built from upstream's Dockerfile

Upstream builds on `jupyter/base-notebook:2022-06-06` from a git checkout,
because the tool's primary interface is a notebook widget. That base is pinned
to a 2022 image and carries a JupyterLab stack this image has no use for on a
batch scheduler.

`afwizard` 1.0.1 and `afwizard-library` are both published on **conda-forge**,
upstream's own distribution channel, so this installs from there: same
software, current dependencies, no notebook server.

`PROJ_LIB` is set for the same reason upstream sets it — without it the conda
PROJ database is not found and reprojection fails at runtime rather than at
import. Upstream calls the underlying situation a bug; the workaround is
theirs.

## GPU

None. This is PDAL-based geometric filtering — no neural network, no CUDA path.
Labelled `bradleylab.model.gpu="none"` so the Compute2 harness does not report
it as a GPU failure. Run it on `general-cpu`.

## Status

**Experimental — not yet run on real data by this lab.** The build asserts that
the batch entry point resolves and that PROJ has a working database behind it,
but no filtering run has been scored. See `SMOKE.md`.
