#!/bin/bash
# Stage all 745 Tyson tiles from S3 to Compute2 scratch.
#
# Run this once on a Compute2 login node or interactive allocation
# (single-shot sync, ~15 min wall for ~6 GB). Skips tiles that already
# exist locally (rsync-style sync is what `aws s3 sync` does by default).

set -euo pipefail

BUCKET="${BUCKET:-s3://bradleylab-public/pointclouds/tyson_2025-11-04_tiles_100m}"
DEST="${DEST:-/scratch2/fs1/alexander.s.bradley/tyson_ams3d/tiles}"

mkdir -p "${DEST}"

echo "Staging tiles:"
echo "  source: ${BUCKET}"
echo "  dest:   ${DEST}"

aws s3 sync "${BUCKET}" "${DEST}" --only-show-errors

N_TILES=$(ls -1 "${DEST}"/*.laz 2>/dev/null | wc -l)
echo "Staged ${N_TILES} LAZ files."

# Build manifests
MANIFEST_DIR="${DEST%/tiles}/manifests"
echo
echo "Building manifests in ${MANIFEST_DIR}"
python3 /opt/crownseg/build_manifests.py \
  --source "${DEST}" \
  --out-dir "${MANIFEST_DIR}"

ls -la "${MANIFEST_DIR}"
