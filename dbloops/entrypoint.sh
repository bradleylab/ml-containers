#!/bin/bash
# Entrypoint for the dbloops container.
#
# The MATLAB Compiler-generated wrapper expects the MATLAB Runtime root
# as its first positional argument. We discover it at runtime so the
# image is robust to MCR_VERSION bumps without needing to hardcode the
# release directory name (e.g., /opt/mcr/R2024b).
#
# Our wrapper (run_dbloops_patch.m) reads PATCH_XYZ, PATCH_OUT, NP_VAL,
# ESFA, ESFB from the environment, so additional arguments are simply
# forwarded for forward compatibility.

set -euo pipefail

MCR_ROOT=$(ls -d /opt/mcr/R* 2>/dev/null | head -1)
if [[ -z "${MCR_ROOT}" ]]; then
    echo "error: no MATLAB Runtime found under /opt/mcr/" >&2
    exit 1
fi

exec /opt/dbloops/bin/run_run_dbloops_patch.sh "$MCR_ROOT" "$@"
