"""
Pre-compute a numpy index of preprocessed RRUFF reference Raman
spectra. Runs at Docker build time so runtime matching is fast and
offline.

Output: /opt/rruff_index.npz with keys:
    wavenumbers : (W,) common wavenumber grid (cm^-1)
    intensities : (N, W) preprocessed + L2-normalized reference spectra
    names       : (N,)   mineral names parsed from RRUFF headers
    rruff_ids   : (N,)   RRUFF IDs (e.g. R040125)
    lasers      : (N,)   laser wavelengths (e.g. "532")

The "excellent_unoriented" archive (229 MB) is the canonical curated
reference set. Other archives can be added via additional --dataset
flags but are not pulled by default to keep the image size in check.

We download the archives directly from rruff.net and parse them
in-house rather than via `ramanspy.datasets.rruff`. ramanspy's
upstream parser is strict about line format (split-on-comma without
fallback) and chokes on real archive files that contain occasional
non-data lines. Our parser tolerates them.

Citation for the underlying data:
    Lafuente B, Downs RT, Yang H, Stone N (2015). The power of databases:
    the RRUFF project. In: Highlights in Mineralogical Crystallography,
    T Armbruster & RM Danisi, eds., De Gruyter, Berlin, 1-30.
"""

from __future__ import annotations

import argparse
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import ramanspy as rp


RRUFF_BASE_URL = "https://www.rruff.net/zipped_data_files/raman"
HEADER_RE = re.compile(r"^##([A-Z_]+)=(.*)$")


# Common wavenumber grid for the Raman fingerprint region. 100-1500
# cm^-1 at 1 cm^-1 resolution covers the diagnostic peaks for nearly
# all rock-forming minerals; the OH stretch region (3000-3700) is
# excluded to keep the index focused on lattice modes. Override via
# --wavenumber-min / --wavenumber-max if you need a different window.
WAVENUMBER_MIN_DEFAULT = 100.0
WAVENUMBER_MAX_DEFAULT = 1500.0
WAVENUMBER_STEP_DEFAULT = 1.0


def build_pipeline() -> rp.preprocessing.Pipeline:
    """Standard preprocessing pipeline applied identically to library
    and unknown spectra.

    Pipeline rationale:
    - WhitakerHayes despike : remove cosmic rays (always safe)
    - SavGol denoise        : modest smoothing without distorting peaks
    - ASLS baseline         : robust fluorescence baseline removal
    - Vector normalise      : L2 normalisation so cosine similarity is
                              well-defined and laser-power-independent
    """
    return rp.preprocessing.Pipeline([
        rp.preprocessing.despike.WhitakerHayes(),
        rp.preprocessing.denoise.SavGol(window_length=7, polyorder=3),
        rp.preprocessing.baseline.ASLS(),
        rp.preprocessing.normalise.Vector(),
    ])


def resample_to_grid(
    intensity: np.ndarray,
    wavenumber: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate a single spectrum onto the common grid.
    Wavenumber regions outside the spectrum's range are zero-filled,
    which is appropriate after L2 normalisation (zero contribution
    rather than fabricated signal)."""
    return np.interp(grid, wavenumber, intensity, left=0.0, right=0.0)


def parse_rruff_metadata(header: dict) -> dict:
    """Normalize the parsed ##KEY=value headers to a stable schema."""
    name = header.get("NAMES") or header.get("MINERAL") or ""
    rruff_id = header.get("RRUFFID") or header.get("ID") or ""
    laser = header.get("LASER_WAVELENGTH") or header.get("LASER") or ""
    return {"name": str(name).strip(),
            "rruff_id": str(rruff_id).strip(),
            "laser": str(laser).strip()}


def download_rruff_zip(dataset: str, dest_zip: Path) -> Path:
    """Fetch an RRUFF zip archive if not already on disk."""
    url = f"{RRUFF_BASE_URL}/{dataset}.zip"
    if dest_zip.exists() and dest_zip.stat().st_size > 0:
        print(f"  cache hit: {dest_zip}", file=sys.stderr)
        return dest_zip
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dest_zip}", file=sys.stderr)
    with urllib.request.urlopen(url, timeout=300) as resp, \
            dest_zip.open("wb") as out:
        while chunk := resp.read(1 << 20):
            out.write(chunk)
    return dest_zip


def parse_rruff_text(text: str) -> tuple[dict, np.ndarray, np.ndarray] | None:
    """Parse a single RRUFF .txt file. Returns (header, wavenumber,
    intensity) or None if the file does not yield a valid spectrum.

    RRUFF format:
      lines starting with '##KEY=...' carry metadata
      data lines are 'wavenumber, intensity' (or whitespace-separated)
      blank lines and stray non-numeric lines are silently skipped
    """
    header: dict[str, str] = {}
    wn: list[float] = []
    inten: list[float] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = HEADER_RE.match(line)
        if m:
            header[m.group(1)] = m.group(2).strip()
            continue
        if line.startswith("##"):  # malformed header — skip
            continue
        # Try comma, then whitespace.
        for sep in (",", None):
            parts = [p for p in (line.split(sep) if sep else line.split()) if p]
            if len(parts) >= 2:
                try:
                    a = float(parts[0])
                    b = float(parts[1])
                except ValueError:
                    break
                wn.append(a)
                inten.append(b)
                break
    if len(wn) < 16:  # too short to be a usable spectrum
        return None
    return header, np.asarray(wn, dtype=np.float64), np.asarray(inten, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        action="append",
        default=None,
        help=(
            "RRUFF dataset name (ramanspy convention). Pass multiple "
            "times to combine. Default: ['excellent_unoriented']."
        ),
    )
    ap.add_argument("--out", type=Path, required=True,
                    help="Output .npz path")
    ap.add_argument("--data-dir", type=Path,
                    default=Path("/opt/rruff/raw"),
                    help="Where ramanspy unpacks the RRUFF archives")
    ap.add_argument("--wavenumber-min", type=float,
                    default=WAVENUMBER_MIN_DEFAULT)
    ap.add_argument("--wavenumber-max", type=float,
                    default=WAVENUMBER_MAX_DEFAULT)
    ap.add_argument("--wavenumber-step", type=float,
                    default=WAVENUMBER_STEP_DEFAULT)
    args = ap.parse_args()

    datasets = args.dataset or ["excellent_unoriented"]
    args.data_dir.mkdir(parents=True, exist_ok=True)
    grid = np.arange(args.wavenumber_min,
                     args.wavenumber_max + args.wavenumber_step / 2,
                     args.wavenumber_step)

    print(f"datasets   : {datasets}", file=sys.stderr)
    print(f"grid       : {grid[0]:.0f}-{grid[-1]:.0f} cm^-1, "
          f"{len(grid)} points", file=sys.stderr)
    print(f"data_dir   : {args.data_dir}", file=sys.stderr)

    pipeline = build_pipeline()

    all_intens: list[np.ndarray] = []
    all_names: list[str] = []
    all_ids: list[str] = []
    all_lasers: list[str] = []
    n_failed = 0

    for ds in datasets:
        print(f"\n=== {ds} ===", file=sys.stderr)
        zip_path = download_rruff_zip(ds, args.data_dir / f"{ds}.zip")
        with zipfile.ZipFile(zip_path) as zf:
            members = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            print(f"  archive members: {len(members)}", file=sys.stderr)
            for i, member in enumerate(members):
                try:
                    text = zf.read(member).decode("utf-8", errors="replace")
                except Exception as e:
                    n_failed += 1
                    if n_failed <= 5:
                        print(f"  skip {member}: read error {e}",
                              file=sys.stderr)
                    continue

                parsed = parse_rruff_text(text)
                if parsed is None:
                    n_failed += 1
                    continue
                header, wn, inten = parsed

                try:
                    spec = rp.Spectrum(inten, wn)
                    processed = pipeline.apply(spec)
                    pwn = np.asarray(processed.spectral_axis, dtype=np.float64)
                    pinten = np.asarray(
                        processed.spectral_data, dtype=np.float64
                    ).squeeze()
                    if pinten.ndim != 1:
                        n_failed += 1
                        continue
                    if pwn[0] > pwn[-1]:
                        pwn = pwn[::-1]
                        pinten = pinten[::-1]
                    resampled = resample_to_grid(pinten, pwn, grid)
                    norm = np.linalg.norm(resampled)
                    if norm == 0 or not np.isfinite(norm):
                        n_failed += 1
                        continue
                    resampled = resampled / norm
                except Exception as e:
                    n_failed += 1
                    if n_failed <= 5:
                        print(f"  skip {member}: pipeline error {e}",
                              file=sys.stderr)
                    continue

                md = parse_rruff_metadata(header)
                all_intens.append(resampled.astype(np.float32))
                all_names.append(md["name"])
                all_ids.append(md["rruff_id"])
                all_lasers.append(md["laser"])

                if (i + 1) % 200 == 0:
                    print(f"  processed {i + 1}/{len(members)}",
                          file=sys.stderr)

    if not all_intens:
        print("ERROR: no spectra processed; index would be empty",
              file=sys.stderr)
        return 1

    intensities = np.stack(all_intens, axis=0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        wavenumbers=grid.astype(np.float32),
        intensities=intensities,
        names=np.array(all_names, dtype=object),
        rruff_ids=np.array(all_ids, dtype=object),
        lasers=np.array(all_lasers, dtype=object),
    )
    print(
        f"\nwrote {args.out}: {intensities.shape}, "
        f"{n_failed} failed",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
