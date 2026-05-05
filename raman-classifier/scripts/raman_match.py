"""
Match an unknown Raman spectrum against the bundled RRUFF reference
library and return top-k mineral candidates by cosine similarity.

Pipeline:
    1. Load unknown spectrum (CSV/txt with wavenumber, intensity).
    2. Apply the same preprocessing as build_rruff_index.py:
       Whitaker-Hayes despike -> SavGol denoise -> ASLS baseline ->
       L2 normalisation.
    3. Resample onto the index's wavenumber grid (default 100-1500 cm^-1
       at 1 cm^-1).
    4. Compute cosine similarity against every reference; return top-k.

Cosine similarity is reported because the references are stored
L2-normalised and the unknown is L2-normalised at preprocess time;
cos = u . v directly. Spectral Angle Distance (SAD = arccos(cos)) is
order-equivalent.

Usage:
    python /opt/scripts/raman_match.py \\
      --spectrum /work/unknown.txt \\
      --top-k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import ramanspy as rp


def load_spectrum_text(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a 2-column wavenumber-intensity text file. Tries comma
    then whitespace; skips header lines starting with '#' or '##'."""
    for sep in (",", None):
        try:
            data = np.loadtxt(path, delimiter=sep, comments="#")
        except ValueError:
            continue
        if data.ndim == 2 and data.shape[1] >= 2:
            return data[:, 0].astype(np.float64), data[:, 1].astype(np.float64)
    raise ValueError(
        f"could not parse {path} as a 2-column wavenumber/intensity file"
    )


def build_pipeline() -> rp.preprocessing.Pipeline:
    """Identical to build_rruff_index.py — must stay in sync so
    library and unknown are processed the same way."""
    return rp.preprocessing.Pipeline([
        rp.preprocessing.despike.WhitakerHayes(),
        rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
        rp.preprocessing.baseline.ASLS(),
        rp.preprocessing.normalise.Vector(),
    ])


def preprocess_unknown(
    wavenumber: np.ndarray,
    intensity: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    pipeline = build_pipeline()
    spec = rp.Spectrum(intensity, wavenumber)
    processed = pipeline.apply(spec)
    wn = np.asarray(processed.spectral_axis, dtype=np.float64)
    inten = np.asarray(processed.spectral_data, dtype=np.float64).squeeze()
    if wn[0] > wn[-1]:
        wn = wn[::-1]
        inten = inten[::-1]
    resampled = np.interp(grid, wn, inten, left=0.0, right=0.0)
    norm = np.linalg.norm(resampled)
    if norm == 0 or not np.isfinite(norm):
        raise ValueError("processed spectrum has zero norm")
    return (resampled / norm).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spectrum", type=Path, required=True,
                    help="Unknown spectrum (2-column text: wavenumber, intensity)")
    ap.add_argument("--index", type=Path,
                    default=Path("/opt/rruff_index.npz"),
                    help="Pre-computed RRUFF index npz")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--out", type=Path, default=None,
                    help="Optional CSV output (otherwise stdout table)")
    args = ap.parse_args()

    if not args.spectrum.exists():
        print(f"ERROR: spectrum not found: {args.spectrum}", file=sys.stderr)
        return 2
    if not args.index.exists():
        print(f"ERROR: index not found: {args.index}", file=sys.stderr)
        return 2

    idx = np.load(args.index, allow_pickle=True)
    grid = idx["wavenumbers"]
    refs = idx["intensities"]
    names = idx["names"]
    ids = idx["rruff_ids"]
    lasers = idx["lasers"]

    wn, inten = load_spectrum_text(args.spectrum)
    print(f"unknown spectrum: {len(wn)} points, "
          f"{wn.min():.0f}-{wn.max():.0f} cm^-1", file=sys.stderr)

    query = preprocess_unknown(wn, inten, grid)

    cos = refs @ query  # references are pre-normalised
    order = np.argsort(-cos)[: args.top_k]

    print(f"\ntop {args.top_k} matches:", file=sys.stderr)
    rows = []
    for rank, i in enumerate(order, start=1):
        row = {
            "rank": rank,
            "mineral": str(names[i]),
            "rruff_id": str(ids[i]),
            "laser": str(lasers[i]),
            "cosine": float(cos[i]),
            "sad_rad": float(np.arccos(np.clip(cos[i], -1.0, 1.0))),
        }
        rows.append(row)
        print(
            f"  {rank:2d}. {row['cosine']:.4f}  "
            f"{row['mineral']:<28s}  {row['rruff_id']}  laser={row['laser']}",
            file=sys.stderr,
        )

    if args.out:
        import csv
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
