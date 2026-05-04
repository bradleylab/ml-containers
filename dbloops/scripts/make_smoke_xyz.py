#!/usr/bin/env python3
"""Synthesize a small XYZ patch with clusterable boulders for the build-time smoke test.

Uniform random points caused the DBloops smoke test to fail: at our
auto-tuned epsilon every point was a DBSCAN noise point, so DBloops
returned with zero clusters and the wrapper's 6-output destructure
errored with MATLAB:unassignedOutputs.

Synthesize 30 boulders as upper-hemisphere point clouds (radius ~0.10-0.25 m,
60-100 points each) plus a sparse matrix layer. DBloops should find
~30 clusters and complete normally.
"""

from __future__ import annotations

import sys
import numpy as np


def main(out_path: str) -> None:
    rng = np.random.default_rng(42)
    patch_size = 5.0  # metres

    n_boulders = 30
    blocks: list[np.ndarray] = []

    for _ in range(n_boulders):
        cx, cy = rng.uniform(0.5, patch_size - 0.5, size=2)
        cz = rng.uniform(0.0, 0.3)
        radius = rng.uniform(0.10, 0.25)
        n_pts = int(rng.integers(60, 101))

        theta = rng.uniform(0.0, 2 * np.pi, n_pts)
        phi = rng.uniform(0.0, np.pi / 2, n_pts)  # upper hemisphere only
        jitter = rng.normal(0.0, 0.005, size=(n_pts, 3))

        px = cx + radius * np.sin(phi) * np.cos(theta) + jitter[:, 0]
        py = cy + radius * np.sin(phi) * np.sin(theta) + jitter[:, 1]
        pz = cz + radius * np.cos(phi) + jitter[:, 2]
        blocks.append(np.column_stack([px, py, pz]))

    # Sparse matrix layer between boulders so DBloops sees an outlier population too.
    n_matrix = 250
    mx = rng.uniform(0.0, patch_size, n_matrix)
    my = rng.uniform(0.0, patch_size, n_matrix)
    mz = rng.uniform(0.0, 0.05, n_matrix)
    blocks.append(np.column_stack([mx, my, mz]))

    pts = np.vstack(blocks)
    np.savetxt(out_path, pts, fmt="%.6f")
    print(f"wrote {pts.shape[0]} points ({n_boulders} boulders + {n_matrix} matrix) to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: make_smoke_xyz.py <out.xyz>", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
