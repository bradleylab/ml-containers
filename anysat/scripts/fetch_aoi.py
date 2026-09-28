#!/usr/bin/env python
"""Fetch NAIP, a Sentinel-2 L2A time series and NLCD over one square AOI.

NAIP and Sentinel-2 come from Microsoft Planetary Computer's STAC API, signed
anonymously with ``planetary_computer``; NLCD 2021 Land Cover comes from the
MRLC GeoServer WCS, since Planetary Computer does not carry it. Everything is
written on one grid in WGS 84 / UTM 15N, ready for ``anysat_features.py``:

  naip.tif        4 bands (R, G, B, NIR) at 1.25 m, AnySat's NAIP resolution
  s2.tif          date-major stack, 10 bands per date in AnySat's order, 10 m
  s2_dates.txt    one ISO date per line, in stack order
  nlcd.tif        NLCD 2021 class codes on the 10 m grid (nearest neighbor)
  aoi.json        bounds, CRS, the STAC item IDs used and the NLCD source

The default AOI is a 3840 m square centered on the Tyson Research Center
ForestGEO plot (38.5178 N, 90.5575 W, from forestgeo.si.edu), which is four
by four 960 m tiles.
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
import rasterio.errors
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
NLCD_WCS = "https://www.mrlc.gov/geoserver/mrlc_download/wcs"
NLCD_COVERAGE = "mrlc_download__NLCD_2021_Land_Cover_L48"
NLCD_CRS = "EPSG:5070"

AOI_CRS = "EPSG:32615"  # Sentinel-2 tile 15SYC's CRS, so 10 m bands need no resampling
TYSON_LAT, TYSON_LON = 38.5178, -90.5575  # ForestGEO site page, Tyson Research Center
AOI_EDGE_M = 3840
S2_GRID_M = 10  # Sentinel-2 10 m pixel edges fall on multiples of 10 m in UTM

S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]  # AnySat order
S2_10M = {"B02", "B03", "B04", "B08"}
# Scene-level prefilter only; the per-date decision is the SCL check below.
MAX_SCENE_CLOUD_PCT = 20
# A date is kept when at most this fraction of AOI pixels carries a rejected
# SCL class. A choice, not an upstream value: requiring zero keeps too few
# dates for a time series, and flagged pixels on kept dates are left as
# observed rather than filled.
MAX_FLAGGED_FRACTION = 0.01
# Scene Classification Layer codes that make a date unusable over the AOI
# (L2A SCL legend): 0 no data, 1 saturated or defective, 3 cloud shadows,
# 8 and 9 cloud medium and high probability, 10 thin cirrus. Class 2
# (topographic or dark-feature shadow) is terrain, not atmosphere, and is kept.
SCL_REJECT = [0, 1, 3, 8, 9, 10]
# GDAL's own HTTP retries do not cover a failed DNS lookup, which ends a
# fetch of many remote reads as surely as a real error.
READ_ATTEMPTS = 4


@dataclass(frozen=True)
class Grid:
    crs: str
    bounds: tuple[float, float, float, float]
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        w, s, e, n = self.bounds
        return round((n - s) / self.res), round((e - w) / self.res)

    @property
    def transform(self):
        return from_origin(self.bounds[0], self.bounds[3], self.res, self.res)

    def at(self, res: float) -> Grid:
        return Grid(self.crs, self.bounds, res)


def tyson_grid(edge_m: int = AOI_EDGE_M) -> Grid:
    x, y = Transformer.from_crs("EPSG:4326", AOI_CRS, always_xy=True).transform(TYSON_LON, TYSON_LAT)
    x, y = (S2_GRID_M * round(v / S2_GRID_M) for v in (x, y))
    h = edge_m / 2
    return Grid(AOI_CRS, (x - h, y - h, x + h, y + h), float(S2_GRID_M))


def lonlat_bbox(grid: Grid) -> list[float]:
    t = Transformer.from_crs(grid.crs, "EPSG:4326", always_xy=True)
    w, s, e, n = grid.bounds
    return list(t.transform_bounds(w, s, e, n))


def warp(href: str, grid: Grid, resampling: Resampling, indexes=None) -> tuple[np.ndarray, np.ndarray]:
    """Read a raster onto ``grid``; returns (data, valid mask).

    The alpha band marks where the source actually covers the grid: NAIP has
    no nodata value, so without it a quarter-quad reads as valid zeros
    everywhere outside its own footprint.
    """
    for attempt in range(READ_ATTEMPTS):
        try:
            return _warp_once(href, grid, resampling, indexes)
        except rasterio.errors.RasterioIOError:
            if attempt == READ_ATTEMPTS - 1:
                raise
            time.sleep(2**attempt)


def _warp_once(href: str, grid: Grid, resampling: Resampling, indexes) -> tuple[np.ndarray, np.ndarray]:
    h, w = grid.shape
    with (
        rasterio.open(href) as src,
        WarpedVRT(
            src,
            crs=grid.crs,
            transform=grid.transform,
            width=w,
            height=h,
            resampling=resampling,
            add_alpha=True,
        ) as vrt,
    ):
        data = vrt.read(indexes if indexes is not None else list(range(1, src.count + 1)))
        valid = vrt.read(vrt.count) > 0
    return data, valid


def fetch_naip(catalog, grid: Grid, year: int) -> tuple[np.ndarray, list[str]]:
    """Mosaic every NAIP quarter-quad of ``year`` that touches the AOI.

    The NAIP items over the AOI are 0.6 m; area-averaging to 1.25 m matches
    the resolution AnySat's NAIP projector was trained at.
    """
    items = list(
        catalog.search(
            collections=["naip"], bbox=lonlat_bbox(grid), datetime=f"{year}-01-01/{year}-12-31"
        ).items()
    )
    g = grid.at(1.25)
    mosaic, filled = np.zeros((4, *g.shape), np.float32), np.zeros(g.shape, bool)
    for item in items:
        data, valid = warp(item.assets["image"].href, g, Resampling.average)
        take = valid & ~filled
        mosaic[:, take] = data[:, take]
        filled |= take
    if not filled.all():
        raise RuntimeError(f"NAIP {year} leaves {(~filled).sum()} pixels of the AOI empty")
    return mosaic, [i.id for i in items]


def boa_offset(item) -> float:
    """BOA_ADD_OFFSET from the product metadata, or 0 where there is none.

    Products from processing baseline 04.00 on record BOA_ADD_OFFSET = -1000;
    03.00 products have no such field. The 2021 items over Tyson mix
    baselines 02.12, 03.00 and 04.00, so the offset has to come off before
    the dates are comparable.
    """
    with urllib.request.urlopen(item.assets["product-metadata"].href, timeout=120) as r:
        found = re.findall(r"<BOA_ADD_OFFSET[^>]*>(-?\d+)<", r.read().decode())
    if len(set(found)) > 1:
        raise RuntimeError(f"{item.id}: band-dependent BOA_ADD_OFFSET {set(found)}")
    return float(found[0]) if found else 0.0


def fetch_s2(catalog, grid: Grid, year: int) -> tuple[np.ndarray, list[str], list[str]]:
    """Every clear date in ``year``, one item per date.

    Where a date has more than one item (reprocessed copies), the one with
    the fewest flagged AOI pixels is used.
    """
    items = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=lonlat_bbox(grid),
        datetime=f"{year}-01-01/{year}-12-31",
        filter={"op": "<=", "args": [{"property": "eo:cloud_cover"}, MAX_SCENE_CLOUD_PCT]},
        filter_lang="cql2-json",
    ).items()
    best = {}
    for item in items:
        scl, _ = warp(item.assets["SCL"].href, grid, Resampling.nearest, 1)
        flagged = float(np.isin(scl, SCL_REJECT).mean())
        day = item.datetime.date().isoformat()
        if flagged <= MAX_FLAGGED_FRACTION and (day not in best or flagged < best[day][0]):
            best[day] = (flagged, item)
    stack, dates, ids = [], [], []
    for day, (_, item) in sorted(best.items()):
        offset = boa_offset(item)
        bands = [
            warp(item.assets[b].href, grid, Resampling.nearest if b in S2_10M else Resampling.bilinear, 1)[0]
            for b in S2_BANDS
        ]
        stack.append(np.stack(bands).astype(np.float32) + offset)
        dates.append(day)
        ids.append(item.id)
    if not stack:
        raise RuntimeError(f"no clear Sentinel-2 date over the AOI in {year}")
    return np.concatenate(stack), dates, ids


def fetch_nlcd(grid: Grid, workdir: Path) -> np.ndarray:
    """NLCD 2021 from the MRLC WCS (Albers, 30 m), resampled onto the grid."""
    t = Transformer.from_crs(grid.crs, NLCD_CRS, always_xy=True)
    w, s, e, n = t.transform_bounds(*grid.bounds)
    pad = 60  # two NLCD pixels, so the nearest-neighbor warp has no edge gaps
    query = [
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("request", "GetCoverage"),
        ("coverageid", NLCD_COVERAGE),
        ("format", "image/geotiff"),
        ("subset", f"X({w - pad:.0f},{e + pad:.0f})"),
        ("subset", f"Y({s - pad:.0f},{n + pad:.0f})"),
    ]
    raw = workdir / "nlcd_2021_albers.tif"
    with urllib.request.urlopen(f"{NLCD_WCS}?{urllib.parse.urlencode(query)}", timeout=300) as r:
        raw.write_bytes(r.read())
    return warp(str(raw), grid, Resampling.nearest, 1)[0]


def write(path: Path, data: np.ndarray, grid: Grid, dtype: str) -> None:
    data = data if data.ndim == 3 else data[None]
    h, w = grid.shape
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=w,
        height=h,
        count=data.shape[0],
        dtype=dtype,
        crs=grid.crs,
        transform=grid.transform,
        compress="deflate",
        tiled=True,
    ) as dst:
        dst.write(data.astype(dtype))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--naip-year", type=int, default=2020)
    p.add_argument("--s2-year", type=int, default=2021)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    grid = tyson_grid()
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2"
    ):
        naip, naip_ids = fetch_naip(catalog, grid, args.naip_year)
        s2, dates, s2_ids = fetch_s2(catalog, grid, args.s2_year)
        nlcd = fetch_nlcd(grid, args.out)
    write(args.out / "naip.tif", naip, grid.at(1.25), "float32")
    write(args.out / "s2.tif", s2, grid, "float32")
    write(args.out / "nlcd.tif", nlcd, grid, "uint8")
    (args.out / "s2_dates.txt").write_text("\n".join(dates) + "\n")
    (args.out / "aoi.json").write_text(
        json.dumps(
            {
                "crs": grid.crs,
                "bounds": grid.bounds,
                "center_lat_lon": [TYSON_LAT, TYSON_LON],
                "naip_items": naip_ids,
                "s2_items": s2_ids,
                "s2_dates": dates,
                "nlcd": {"service": NLCD_WCS, "coverage": NLCD_COVERAGE},
            },
            indent=2,
        )
    )
    print(f"NAIP {naip.shape} from {len(naip_ids)} items; S2 {len(dates)} clear dates; NLCD {nlcd.shape}")


if __name__ == "__main__":
    main()
