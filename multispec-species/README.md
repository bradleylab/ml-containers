# multispec-species

Per-tree species classification from Altum-PT 5-band reflectance
mosaics, using lidar-derived crown polygons as masks and ForestGEO
census species labels as supervision.

This container is **Approach 1** of the Tyson multispec plan:
XGBoost on per-crown zonal spectral stats. CPU-only; run on
Compute2 `general-cpu`.

## Image

`ghcr.io/bradleylab/multispec-species:v1`

## Scripts

- `extract_crown_spectra.py` — zonal stats + vegetation indices per crown
- `train_xgboost.py` — spatial-split train/test + XGBoost multiclass

See `tyson-forest-linkage/multispec/docs/PLAN.md` for the full
analytical plan (regions, labels, train/test discipline, metrics).
