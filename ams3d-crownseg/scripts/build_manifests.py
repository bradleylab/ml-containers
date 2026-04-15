#!/usr/bin/env python3
"""Build tile manifests for SLURM array submission.

Takes a list of tile filenames (either from an S3 bucket or a local
directory) and writes tab-separated manifests that the SLURM array script
reads with `sed -n "$((TASK_ID+1))p"`.

Manifest format (no header):
  tile_name<TAB>grid_x<TAB>grid_y

Produces multiple manifests:
  smoke_5.tsv       — 5 existing test tiles
  pilot_25.tsv      — 5x5 contiguous block around a central tile
  rehearsal_100.tsv — 10x10 contiguous block
  full_745.tsv      — all tiles
  sensitivity_30.tsv — 30 tiles stratified by point count
                       (computed after the full run completes, using
                        per-tile point counts from the full run logs)
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

TILE_RE = re.compile(r"^tile_(-?\d+)_(-?\d+)\.laz$")

# Five existing test tiles (already used throughout the project).
SMOKE_TILES = [
    ("tile_-10_8", -10, 8),
    ("tile_-10_10", -10, 10),
    ("tile_-8_10", -8, 10),
    ("tile_-4_16", -4, 16),
    ("tile_-2_12", -2, 12),
]

# Pilot 5x5 block: dense canopy area around tile_-10_8.
# We'll include all (-12..-8) × (6..10) = 25 tiles if they exist.
PILOT_RANGE_X = range(-12, -7)
PILOT_RANGE_Y = range(6, 11)

# Rehearsal 10x10: extend to (-14..-5) × (5..14) = 100 tiles.
REHEARSAL_RANGE_X = range(-14, -4)
REHEARSAL_RANGE_Y = range(5, 15)


def load_tiles_list(source: Path | str) -> list[tuple[str, int, int]]:
    """Return list of (tile_name, grid_x, grid_y) for tiles in source.

    Source can be a local directory or a pre-saved tile listing file
    (e.g. `aws s3 ls ... > tyson_tiles.txt`).
    """
    if isinstance(source, str) and source.startswith("s3://"):
        raise NotImplementedError("direct s3:// reads not supported; "
                                   "pipe `aws s3 ls ...` to a file first")
    source = Path(source)
    tiles: list[tuple[str, int, int]] = []
    if source.is_dir():
        for p in sorted(source.glob("tile_*.laz")):
            m = TILE_RE.match(p.name)
            if m:
                tiles.append((p.stem, int(m.group(1)), int(m.group(2))))
    elif source.is_file():
        for line in source.read_text().splitlines():
            # Handle both "2026-04-03 20:55:06    7504602 tile_-10_8.laz"
            # and bare "tile_-10_8.laz".
            name = line.strip().split()[-1]
            m = TILE_RE.match(name)
            if m:
                tiles.append((name[:-4], int(m.group(1)), int(m.group(2))))
    else:
        raise SystemExit(f"source not found: {source}")
    return tiles


def write_manifest(out: Path, tiles: list[tuple[str, int, int]]) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for name, gx, gy in tiles:
            f.write(f"{name}\t{gx}\t{gy}\n")
    print(f"  wrote {out} ({len(tiles)} tiles)")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", required=True,
                   help="directory of tiles or s3-listing text file")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()

    available = load_tiles_list(args.source)
    avail_set = {name for name, _, _ in available}
    print(f"discovered {len(available)} tiles in {args.source}")

    # Smoke: 5 existing test tiles
    smoke = [t for t in SMOKE_TILES if t[0] in avail_set]
    write_manifest(args.out_dir / "smoke_5.tsv", smoke)

    # Pilot: 5x5 block that exists
    pilot = [(f"tile_{x}_{y}", x, y)
             for x in PILOT_RANGE_X for y in PILOT_RANGE_Y
             if f"tile_{x}_{y}" in avail_set]
    write_manifest(args.out_dir / f"pilot_{len(pilot)}.tsv", pilot)

    # Rehearsal: 10x10 block that exists
    rehearsal = [(f"tile_{x}_{y}", x, y)
                 for x in REHEARSAL_RANGE_X for y in REHEARSAL_RANGE_Y
                 if f"tile_{x}_{y}" in avail_set]
    write_manifest(args.out_dir / f"rehearsal_{len(rehearsal)}.tsv", rehearsal)

    # Full: everything
    write_manifest(args.out_dir / f"full_{len(available)}.tsv", available)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
