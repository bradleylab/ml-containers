#!/bin/bash
# Container entrypoint — runs AMS3D on one tile.
#
# Environment variables (set by the SLURM array wrapper):
#   TILE_NAME      e.g. "tile_-10_8"
#   GRID_X, GRID_Y integer tile grid indices (parsed from TILE_NAME if unset)
#   CD2TH_VAL      crown_diameter_to_tree_height ratio (default 0.4)
#   AMS3D_MODE     unused (reserved for future voxel mode)
#   CONVERGENCE    centroid_convergence_distance (default 0.3 — benchmark-chosen
#                  fast mode, 8x speedup vs 0.01 with 94.5% tree count match)
#   IN_DIR         directory containing input LAZ files
#   OUT_DIR        directory to write outputs (created if missing)

set -euo pipefail

TILE_NAME="${TILE_NAME:?TILE_NAME not set}"
CD2TH_VAL="${CD2TH_VAL:-0.4}"
CONVERGENCE="${CONVERGENCE:-0.3}"
IN_DIR="${IN_DIR:?IN_DIR not set}"
OUT_DIR="${OUT_DIR:?OUT_DIR not set}"

# Parse grid_x/grid_y from tile_X_Y pattern if not provided.
if [[ -z "${GRID_X:-}" || -z "${GRID_Y:-}" ]]; then
  IFS='_' read -r _ GRID_X GRID_Y <<< "$(basename "${TILE_NAME}" .laz)"
fi

echo "=== AMS3D tile processing ==="
echo "  tile: ${TILE_NAME}  grid: (${GRID_X}, ${GRID_Y})"
echo "  CD2TH=${CD2TH_VAL}  convergence=${CONVERGENCE}"
echo "  in:  ${IN_DIR}"
echo "  out: ${OUT_DIR}"

mkdir -p "${OUT_DIR}"/{laz,parquet,logs}

Rscript /opt/crownseg/run_tile_ams3d.R \
  --tile-name "${TILE_NAME}" \
  --grid-x "${GRID_X}" --grid-y "${GRID_Y}" \
  --cd2th "${CD2TH_VAL}" --convergence "${CONVERGENCE}" \
  --in-dir "${IN_DIR}" --out-dir "${OUT_DIR}"
