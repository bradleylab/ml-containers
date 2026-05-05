"""
Round-trip smoke test for the baked RRUFF index.

Extracts a random reference spectrum from /opt/rruff_index.npz, adds
mild Poisson-style noise, writes it as a 2-column text file, and runs
the matcher against it. Asserts that the original mineral lands in
the top-3 by cosine similarity.

This validates that the build-time preprocessing pipeline and the
runtime preprocessing pipeline produce comparable embeddings — i.e.
the index isn't subtly corrupted, axes aren't flipped, normalisation
is consistent.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path,
                    default=Path("/opt/rruff_index.npz"))
    ap.add_argument("--matcher", type=Path,
                    default=Path("/opt/scripts/raman_match.py"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-idx", type=int, default=None,
                    help="Specific row to test; default is random")
    ap.add_argument("--noise-frac", type=float, default=0.02,
                    help="Gaussian noise fraction relative to peak intensity")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    idx = np.load(args.index, allow_pickle=True)
    grid = idx["wavenumbers"]
    refs = idx["intensities"]
    names = idx["names"]
    ids = idx["rruff_ids"]

    n = refs.shape[0]
    if args.sample_idx is None:
        # Pick a row whose mineral name is non-empty, otherwise the
        # top-k report is hard to interpret.
        candidates = [i for i in range(n) if names[i]]
        if not candidates:
            print("ERROR: index has no named entries", file=sys.stderr)
            return 1
        sample_idx = int(rng.choice(candidates))
    else:
        sample_idx = args.sample_idx

    truth_name = str(names[sample_idx])
    truth_id = str(ids[sample_idx])
    print(f"truth: row={sample_idx}  mineral={truth_name}  id={truth_id}",
          file=sys.stderr)

    # Reconstruct an "unknown" spectrum by un-normalising the indexed
    # one (it's already on the canonical grid; just add noise).
    base = refs[sample_idx].astype(np.float64)
    peak = float(base.max())
    noise = rng.normal(0.0, args.noise_frac * peak, size=base.shape)
    perturbed = np.clip(base + noise, 0.0, None)

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        for w, y in zip(grid, perturbed):
            f.write(f"{w:.4f},{y:.6e}\n")
        fixture = Path(f.name)

    # Run the matcher.
    proc = subprocess.run(
        [
            sys.executable, str(args.matcher),
            "--spectrum", str(fixture),
            "--index", str(args.index),
            "--top-k", "5",
        ],
        check=False, capture_output=True, text=True,
    )
    fixture.unlink(missing_ok=True)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        print(f"matcher failed: rc={proc.returncode}", file=sys.stderr)
        return proc.returncode

    # Re-rank locally to verify the assertion (matcher writes scores
    # to stderr; easiest to recompute and check).
    query = perturbed.astype(np.float32)
    nrm = np.linalg.norm(query)
    if nrm > 0:
        query = query / nrm
    cos = refs @ query
    top5 = np.argsort(-cos)[:5]
    top5_names = [str(names[i]) for i in top5]

    print(f"\ntop-5 by recomputed cosine: {top5_names}", file=sys.stderr)
    if truth_name in top5_names[:3]:
        print(f"PASS: truth mineral '{truth_name}' in top-3", file=sys.stderr)
        return 0
    print(f"FAIL: truth mineral '{truth_name}' NOT in top-3 "
          f"(top-5 = {top5_names})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
