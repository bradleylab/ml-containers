#!/bin/bash
# Download TreeLearn pretrained weights from Göttingen dataverse with
# per-variant retry. Accepts success if at least one variant lands, so
# transient dataverse 500s on specific file IDs don't block us.
#
# Usage:
#   download_weights.sh [DEST_DIR]
# Default DEST_DIR is /opt/TreeLearn/data/model_weights (image-local,
# lost on container teardown). Pass a bind-mounted path on the host for
# persistence.
set -euo pipefail

DEST="${1:-/opt/TreeLearn/data/model_weights}"
mkdir -p "$DEST"
cd /opt/TreeLearn

SUCCESS=0
for name in \
    model_weights_with_small_20241213 \
    model_weights_20241213 \
    model_weights_diverse_training_data; do
    for attempt in 1 2 3 4 5 6 7 8 9 10; do
        echo "[weights] attempt $attempt for $name -> $DEST"
        if python tree_learn/util/download.py \
            --dataset_name "$name" \
            --root_folder "$DEST"; then
            echo "[weights] $name OK"
            SUCCESS=$((SUCCESS + 1))
            break
        fi
        echo "[weights] $name failed (attempt $attempt); sleeping 30s"
        sleep 30
    done
done

echo "[weights] $SUCCESS of 3 variants downloaded to $DEST"
ls -la "$DEST"

if [ "$SUCCESS" -lt 1 ]; then
    echo "ERROR: no weights downloaded — Göttingen dataverse likely down."
    echo "Retry later, or fetch manually via curl:"
    echo "  BASE=https://data.goettingen-research-online.de/api/access/datafile/:persistentId?persistentId=doi:10.25625/VPMPID/"
    echo "  curl -L -o $DEST/model_weights_with_small_20241213.pth \"\${BASE}TYZJ4E\""
    echo "  curl -L -o $DEST/model_weights_20241213.pth            \"\${BASE}IMHF3G\""
    echo "  curl -L -o $DEST/model_weights_diverse_training_data.pth \"\${BASE}1JMEQV\""
    exit 1
fi
