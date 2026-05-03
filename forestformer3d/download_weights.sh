#!/bin/bash
# Download ForestFormer3D pretrained weights from Zenodo record 16742708
# with retry/backoff. The weights ship inside `clean_forestformer.zip`
# (~198 MB), which extracts to a directory containing
# `epoch_3000_fix.pth` plus auxiliary files. We unzip into the
# destination so the resulting layout matches what the upstream
# config expects:
#
#   <DEST>/clean_forestformer/epoch_3000_fix.pth
#
# Upstream test command (from FF3D readme.md):
#   CUDA_VISIBLE_DEVICES=0 python tools/test.py \
#       configs/oneformer3d_qs_radius16_qp300_2many.py \
#       work_dirs/clean_forestformer/epoch_3000_fix.pth
#
# So a typical Compute2 invocation bind-mounts the destination at
# /workspace/work_dirs/ inside the container.
#
# Usage:
#   download_weights.sh [DEST_DIR]
# Default DEST_DIR is /workspace/work_dirs (image-local; lost on
# container teardown). Pass a bind-mounted scratch path for
# persistence across SLURM jobs.
set -euo pipefail

DEST="${1:-/workspace/work_dirs}"
mkdir -p "$DEST"

ZENODO_URL="https://zenodo.org/api/records/16742708/files/clean_forestformer.zip/content"
ZIP_PATH="$DEST/clean_forestformer.zip"
EXPECTED_MD5="553d67379331966509076f3fbb409e57"
TARGET_FILE="$DEST/clean_forestformer/epoch_3000_fix.pth"

if [ -f "$TARGET_FILE" ]; then
    echo "[weights] $TARGET_FILE already present; skipping download"
    ls -la "$DEST/clean_forestformer/"
    exit 0
fi

SUCCESS=0
for attempt in 1 2 3 4 5 6 7 8 9 10; do
    echo "[weights] attempt $attempt: GET $ZENODO_URL -> $ZIP_PATH"
    if curl --fail --location \
            --connect-timeout 30 --max-time 1800 \
            --retry 3 --retry-delay 10 \
            -o "$ZIP_PATH" "$ZENODO_URL"; then
        echo "[weights] download OK"
        SUCCESS=1
        break
    fi
    echo "[weights] attempt $attempt failed; sleeping $((attempt * 30))s"
    rm -f "$ZIP_PATH"
    sleep $((attempt * 30))
done

if [ "$SUCCESS" -ne 1 ]; then
    echo "ERROR: clean_forestformer.zip not downloaded after 10 attempts."
    echo "Zenodo record: https://zenodo.org/records/16742708"
    echo "If Zenodo is persistently down, fetch manually from a browser."
    exit 1
fi

# Verify checksum (md5 reported by Zenodo's record JSON).
ACTUAL_MD5=$(md5sum "$ZIP_PATH" | awk '{print $1}')
if [ "$ACTUAL_MD5" != "$EXPECTED_MD5" ]; then
    echo "ERROR: md5 mismatch on $ZIP_PATH"
    echo "  expected: $EXPECTED_MD5"
    echo "  actual:   $ACTUAL_MD5"
    exit 1
fi
echo "[weights] md5 verified: $ACTUAL_MD5"

# Extract. The zip top-level is `clean_forestformer/`.
echo "[weights] extracting into $DEST"
if ! command -v unzip >/dev/null 2>&1; then
    echo "[weights] unzip not on PATH; falling back to python zipfile"
    python3 -c "import zipfile, sys; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
        "$ZIP_PATH" "$DEST"
else
    unzip -q -o "$ZIP_PATH" -d "$DEST"
fi

if [ ! -f "$TARGET_FILE" ]; then
    echo "ERROR: $TARGET_FILE missing after extraction. Zip layout may have changed."
    echo "Zip contents:"
    if command -v unzip >/dev/null 2>&1; then unzip -l "$ZIP_PATH"; fi
    exit 1
fi

# Optional cleanup of zip to save scratch space (toggle off by setting KEEP_ZIP=1).
if [ "${KEEP_ZIP:-0}" = "0" ]; then
    rm -f "$ZIP_PATH"
fi

echo "[weights] done. Contents of $DEST/clean_forestformer/:"
ls -la "$DEST/clean_forestformer/"
echo "[weights] checkpoint path: $TARGET_FILE"
