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
