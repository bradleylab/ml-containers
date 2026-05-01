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
