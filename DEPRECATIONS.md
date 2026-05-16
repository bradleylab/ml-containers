# Deprecations

Searchable record of GHCR images that were retired, why, and what
replaces each one. New work should not depend on anything listed here.

## `bradleylab/tree-analysis` (deprecated 2026-05-01)

A 7.27 GB kitchen-sink image bundling SegmentAnyTree + ForAINet +
DeepForest + lidR. Built ad-hoc on 2026-04-03 with no source recipe
in this repo. Failed the one-model-per-container convention adopted
2026-04-30.

**Replaced by:**

| Part of `tree-analysis` | Replacement |
|------------------------|------------|
| SegmentAnyTree | `ghcr.io/bradleylab/segment-any-tree-h100:v2` (and `:v2-defaults`) |
| ForAINet | `ghcr.io/bradleylab/forainet:v1` (experimental H100 port) |
| DeepForest | `ghcr.io/bradleylab/deepforest:v1` |
| lidR | Install locally from CRAN / use `rocker/geospatial:4.4.1`. lidR is a library, not a model — does not belong in `ml-containers`. |

**Action items:**

1. Update SLURM scripts that reference
   `bradleylab+tree-analysis+latest.sqsh` to point at the per-model
   sqsh files above.
2. Operator: delete the `bradleylab/tree-analysis` GHCR package (or
   tag it `:deprecated` if there is a reason to keep it pullable).
3. Delete the `bradleylab+tree-analysis+latest.sqsh` cache from
   `/storage1/fs1/alexander.s.bradley/Active/c2_jobs/` once no jobs
   are using it.

## `bradleylab/multispec-species` (deprecated 2026-05-01)

A Python + GDAL + xgboost environment image bundled with two scripts
(`extract_crown_spectra.py`, `train_xgboost.py`). Failed the boundary
test "could a reviewer name a specific model whose inference this
container runs?" — the image shipped only data-prep + fitting, no
inference entrypoint, no shipped/fetched weights. Closed without
merge in PR #6.

**Replaced by:** the canonical scripts in
`tyson-forest-linkage/multispec/scripts/` (which include `apply_model.py`,
`extract_crown_texture.py`, run_*.sbatch wrappers, plus newer
`train_xgboost.py` with Approach 4 fusion + genus-collapse modes).
Run from a uv venv on a `general-cpu` slot. If a Compute2 sqsh is
still required, build a stock python+GDAL+xgboost env image inside
the analysis repo — not here, since there is no model to ship.

**Action items:**

1. Operator: delete the `bradleylab/multispec-species` GHCR package.
2. Update SLURM scripts that reference
   `bradleylab+multispec-species+v1.sqsh` to either run from a uv
   venv or mount a stock GDAL+xgboost env image.
3. Delete the `bradleylab+multispec-species+v1.sqsh` cache from
   Compute2.

## `bradleylab/sar-amplitude-rtc` (deprecated 2026-05-09)

A SNAP-based ICEYE GRD → calibrated speckle-filtered terrain-flattened
γ⁰_T-in-dB pipeline (Calibration → Refined Lee 5×5 →
Range-Doppler Terrain-Correction → LinearToFromdB), built on top of
`mundialis/esa-snap:latest`. Source lived in
`bradleylab/stl-tornado:containers/sar-amplitude-rtc/` and was never
moved to this repo or published to GHCR; deleted from that repo in
the same change.

**Why retired:** two upstream SNAP s1tbx bugs make the container
unusable for absolute radiometry until upstream patches:

1. `IceyeCalibrator` double-applies the per-scene calibration factor K
   when the graph runs `Calibration → Terrain-Correction` with
   `applyRadiometricNormalization=true`. The output raster is
   `K²·DN²·sin(θ_loc)` instead of `K·DN²/cos(θ_loc)`, a constant offset
   of ~46–50 dB from γ⁰_T (verified empirically at +49.2 dB on ICEYE
   X30 frame 950694619).
2. `RangeDopplerGeocodingOp` silently writes an all-zero raster for
   off-broadside SpotlightHigh acquisitions because the zero-Doppler
   intercept search has no solution outside the acquisition window.

Both are documented in the combined STEP forum draft at
`bradleylab/stl-tornado:scratch/s1tbx-patch/STEP_FORUM_POST_combined.md`.

**Replaced by:** the Tier 1 RPC-based Python geocoder at
`bradleylab/stl-tornado:scripts/grd_to_gamma0_python.py` (uses ICEYE's
own RPC block + GDAL + a DEM, bypassing s1tbx entirely). Sub-pixel
agreement with ICEYE QUICKORTHO across all four STL tornado frames.

**Action items:**

1. None for GHCR — image was never published.
2. None for Compute2 caches — image was never imported.
3. If upstream s1tbx merges the `IceyeCalibrator` fix and someone
   wants the SNAP-based path back, the recipe is recoverable from
   `bradleylab/stl-tornado` git history (commit prior to the deletion
   on 2026-05-09).
