#!/bin/bash
# Place GeoLG-3DFaultNet's pretrained checkpoint where inference.py expects it.
#
# Weights are not baked into the image. The single published checkpoint is a
# GitHub Release asset:
#
#   repo    letsfly27/GeoLG-3DFaultNet
#   tag     v1.0  ("Pre-trained Model Weights", published 2026-03-30)
#   asset   best_model.pth
#   size    71,617,015 bytes
#   sha256  67cb47ab5ebfd7da50eb874011c36ba1c35473c4a94d4fd8da9a0a560c76229a
#
# The size and digest above are recorded from the GitHub Releases API on
# 2026-08-18. GitHub lets a maintainer delete and re-upload an asset under the
# same tag, so this script fails closed on a digest mismatch: a silently
# swapped checkpoint is exactly the failure this pins against, and with a
# single-contributor upstream it is not a hypothetical.
#
# Two modes, same command:
#   - online   — asset absent from WEIGHTS_DIR: download, verify, link.
#   - offline  — asset already in WEIGHTS_DIR (pre-staged to Storage3, or a
#                cache mount from an earlier job): verify, link, no network.
#
# Usage:
#   fetch_weights.sh [WEIGHTS_DIR] [LINK_DIR]
#
#   WEIGHTS_DIR  where the checkpoint lives.   Default: $GEOLG_WEIGHTS_DIR
#                                                       (/opt/geolg-weights)
#   LINK_DIR     where to symlink it as ./best_model.pth, which is the path
#                inference.py resolves against the working directory.
#                Default: $PWD
set -euo pipefail

WEIGHTS_DIR="${1:-${GEOLG_WEIGHTS_DIR:-/opt/geolg-weights}}"
LINK_DIR="${2:-$PWD}"

RELEASE_TAG="v1.0"
ASSET="best_model.pth"
ASSET_URL="https://github.com/letsfly27/GeoLG-3DFaultNet/releases/download/${RELEASE_TAG}/${ASSET}"
EXPECTED_SHA256="67cb47ab5ebfd7da50eb874011c36ba1c35473c4a94d4fd8da9a0a560c76229a"
EXPECTED_BYTES=71617015

TARGET="${WEIGHTS_DIR}/${ASSET}"

mkdir -p "$WEIGHTS_DIR" "$LINK_DIR"

if [ ! -f "$TARGET" ]; then
    echo "[weights] $TARGET absent; fetching ${RELEASE_TAG}/${ASSET} (68 MiB)"
    if ! curl --fail --location \
              --connect-timeout 30 --max-time 900 \
              --retry 5 --retry-delay 10 --retry-connrefused \
              -o "${TARGET}.part" "$ASSET_URL"; then
        rm -f "${TARGET}.part"
        echo "ERROR: download failed: $ASSET_URL" >&2
        echo "  If this is a compute node without egress, pre-stage the file on a" >&2
        echo "  login node and bind-mount its directory at \$GEOLG_WEIGHTS_DIR." >&2
        exit 1
    fi
    mv "${TARGET}.part" "$TARGET"
else
    echo "[weights] $TARGET already present; verifying without download"
fi

ACTUAL_BYTES=$(wc -c < "$TARGET" | tr -d ' ')
ACTUAL_SHA256=$(sha256sum "$TARGET" | awk '{print $1}')

if [ "$ACTUAL_BYTES" != "$EXPECTED_BYTES" ] || [ "$ACTUAL_SHA256" != "$EXPECTED_SHA256" ]; then
    echo "ERROR: checkpoint does not match the recorded release asset." >&2
    echo "  expected  ${EXPECTED_BYTES} bytes  sha256 ${EXPECTED_SHA256}" >&2
    echo "  actual    ${ACTUAL_BYTES} bytes  sha256 ${ACTUAL_SHA256}" >&2
    echo "  The asset under tag ${RELEASE_TAG} has changed, or the file is truncated." >&2
    echo "  Do not use it. Re-verify upstream before updating these constants." >&2
    exit 1
fi
echo "[weights] sha256 verified: $ACTUAL_SHA256"

ln -sfn "$TARGET" "${LINK_DIR}/${ASSET}"
echo "[weights] linked ${LINK_DIR}/${ASSET} -> ${TARGET}"
