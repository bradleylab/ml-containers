#!/bin/bash
# Thin wrapper around `tools/test.py` that fills in the upstream
# defaults for an "out-of-the-box" inference run with the FF3D
# pretrained checkpoint. Intended for the common case: one or more
# .ply tiles in $TEST_DATA, weights at $CKPT, output written by
# FF3D's predict() under the configured work_dir.
#
# Environment variables (with defaults):
#   FF3D_CONFIG     configs/oneformer3d_qs_radius16_qp300_2many.py
#   CKPT            /workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth
#   GPU             0
#
# Pre-requisites at run time:
#   - Test files placed in /workspace/data/ForAINetV2/test_data/
#     (.ply preferred; non-.ply requires upstream-readme code edits)
#   - Test list listing those file basenames at
#     /workspace/data/ForAINetV2/meta_data/test_list.txt
#   - Pre-processed data created via:
#       cd /workspace/data/ForAINetV2 && python batch_load_ForAINetV2_data.py
#       cd /workspace        && python tools/create_data_forainetv2.py forainetv2
#
# This script does NOT run preprocessing — that requires writes inside
# the container layout and is best done in the same SLURM job before
# this script is invoked. See the README "Run on Compute2" section.
set -euo pipefail

cd /workspace

FF3D_CONFIG="${FF3D_CONFIG:-configs/oneformer3d_qs_radius16_qp300_2many.py}"
CKPT="${CKPT:-/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth}"
GPU="${GPU:-0}"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    echo "Run download_weights.sh against a bind-mounted scratch path"
    echo "and bind that path at /workspace/work_dirs inside the container."
    exit 1
fi

if [ ! -f "$FF3D_CONFIG" ]; then
    echo "ERROR: config not found at $FF3D_CONFIG (cwd: $(pwd))"
    exit 1
fi

export PYTHONPATH="${PYTHONPATH:-/workspace}"
export PYTHONNOUSERSITE=1   # required by ml-containers C2 enroot rule
export CUDA_VISIBLE_DEVICES="$GPU"

echo "[ff3d] config:  $FF3D_CONFIG"
echo "[ff3d] ckpt:    $CKPT"
echo "[ff3d] cuda:    $CUDA_VISIBLE_DEVICES"
python tools/test.py "$FF3D_CONFIG" "$CKPT"
