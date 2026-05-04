#!/usr/bin/env bash
# Compile DBloops to a standalone binary using MATLAB Compiler (`mcc`).
#
# This must run on a host with MATLAB Compiler R2024b licensed. We
# develop on WashU Compute2 where the matlab/R2024b module includes
# Compiler under the campus TAH licence. Run from inside the patched
# DBloops source tree (see bradleylab/rock_glaciers/scripts/matlab/
# for the patches).
#
# Usage (from C2 with the source at /storage1/.../dbloops/repo):
#     bash compile.sh /storage1/.../dbloops/repo /storage1/.../dbloops/_compile
#
# Output: <out-dir>/run_dbloops_patch{,sh}, plus a tarball at
# <out-dir>/../dbloops_bin_${MCR_VERSION}_linux.tar.gz that the docker
# build context expects as bin/.

set -euo pipefail

MCR_VERSION=${MCR_VERSION:-R2024b}

if [[ $# -ne 2 ]]; then
    echo "usage: $0 <dbloops-repo-root> <out-dir>" >&2
    exit 2
fi

REPO_DIR="$1"
OUT_DIR="$2"
GSD_DIR="$REPO_DIR/GSD code"

if [[ ! -f "$GSD_DIR/run_dbloops_patch.m" ]]; then
    echo "error: $GSD_DIR/run_dbloops_patch.m missing — run apply_patches.sh first" >&2
    exit 1
fi

# shellcheck disable=SC1091
[[ -f /etc/profile.d/lmod.sh ]] && source /etc/profile.d/lmod.sh
module load ris
module load "matlab/${MCR_VERSION}"
which mcc

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

cd "$GSD_DIR"
mcc -m run_dbloops_patch.m \
    -a DBloops -a Terpunkto -a G3point \
    -d "$OUT_DIR" \
    -v 2>&1 | tail -40

echo
echo "=== output ==="
ls -lh "$OUT_DIR"
echo
echo "=== required toolboxes ==="
cat "$OUT_DIR/requiredMCRProducts.txt" 2>/dev/null || echo "(none)"

TAR="$(dirname "$OUT_DIR")/dbloops_bin_${MCR_VERSION}_linux.tar.gz"
echo
echo "=== packaging $TAR ==="
tar -czf "$TAR" -C "$OUT_DIR" .
ls -lh "$TAR"
sha256sum "$TAR"
