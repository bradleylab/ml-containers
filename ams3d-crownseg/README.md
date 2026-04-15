# ams3d-crownseg

AMS3D (Ferraz et al. 2016) individual tree crown segmentation for airborne
UAV lidar, operating directly on the 3D point cloud (no CHM projection).

- Base: `rocker/geospatial:4.4.1` (R 4.4 + GDAL/PROJ/sf/terra)
- Core R package: [`crownsegmentr`](https://cran.r-project.org/package=crownsegmentr)
  (CRAN implementation of the Ferraz adaptive mean shift 3D algorithm)
- Also: `lidR`, `data.table`, `arrow`, `optparse`
- Python venv for post-processing: `laspy`, `scipy`, `geopandas`, `rasterio`, `pyarrow`
- PDAL for COPC writing

Pull:
```
ghcr.io/bradleylab/ams3d-crownseg:latest
```

Built automatically via GitHub Actions on push to `ams3d-crownseg/`.

## Usage

Per-tile run on Compute2 (Pyxis/enroot):
```
sbatch --container-image=/storage1/.../bradleylab+ams3d-crownseg+v1.sqsh \
       --export=ALL,CD2TH_VAL=0.4,CD2TH_TAG=40,MANIFEST=/path/tiles_manifest.tsv \
       run_ams3d_array.sbatch
```

Per-tile entry point `/opt/crownseg/process_tile.sh` consumes env vars:
- `TILE_NAME`, `GRID_X`, `GRID_Y` — which tile and its grid position
- `CD2TH_VAL` — crown diameter / tree height ratio (default 0.4)
- `CONVERGENCE` — mean shift convergence distance (default 0.3 = fast mode)
- `IN_DIR`, `OUT_DIR` — LAZ locations

## Outputs per tile

- `$OUT_DIR/laz/tile_X_Y.laz` — segmented LAZ with `treeID` extra dim
- `$OUT_DIR/parquet/tile_X_Y_trees.parquet` — per-tree attributes
- `$OUT_DIR/logs/tile_X_Y.json` — timing + tree count + status

## Scripts available inside the container

In `/opt/crownseg/`:
- `run_tile_ams3d.R` — per-tile R worker
- `process_tile.sh` — entrypoint wrapper
- `merge_and_assemble.py` — union parquets, NN relabel, write COPC
- `render_random_colors_u64.py` — bake RGB per global tree ID
- `forest_structure_products.py` — tree attributes, density/gap/vertical rasters
- `build_manifests.py` — build SLURM array manifests from S3 listing
- `stage_tiles.sh` — aws s3 sync + manifest build on Compute2
