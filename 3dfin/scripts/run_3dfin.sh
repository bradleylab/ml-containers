#!/bin/bash
# Invoke 3DFin on a TLS LAZ. Inputs + output dir supplied by the caller.
#
# Minimum invocation discovered at build/runtime via `3DFin cli --help`.
# If a specific arg name changes upstream, the wrapper fails fast and
# the sbatch log surfaces the help text.
set -euo pipefail

INPUT="${1:?usage: run_3dfin.sh <input.laz> <output_dir> [extra 3DFin cli args...]}"
OUTDIR="${2:?usage: run_3dfin.sh <input.laz> <output_dir>}"
shift 2

mkdir -p "${OUTDIR}"

echo "=== 3DFin ==="
echo "  input:  ${INPUT}"
echo "  output: ${OUTDIR}"
3DFin --version 2>&1 | head -3 || true
echo "--- CLI help ---"
3DFin cli --help 2>&1 | head -80

echo "--- running ---"
3DFin cli --input "${INPUT}" --output "${OUTDIR}" "$@"
echo "=== done ==="
ls -la "${OUTDIR}"
