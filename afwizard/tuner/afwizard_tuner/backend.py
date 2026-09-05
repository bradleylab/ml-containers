"""Run one ground-filter evaluation through PDAL and read the results back.

Everything a criterion could want comes out of ONE `pdal pipeline` call:

    readers.las
    filters.ferry   Classification => VendorClass   (SMRF overwrites it)
    <prefilter stages, fixed>
    <the filter under search>
    filters.hag_dem against the reference DEM
    writers.gdal    a DTM from the points now labelled ground
    writers.text    VendorClass, Classification, HeightAboveGround, X, Y, Z

Point-level criteria read the CSV; surface-level criteria read the DTM. The
CLI is used rather than python-pdal so the same code runs on a laptop with
only the PDAL binary and inside the container, and so the scorer stays
backend-agnostic: hand it any classified cloud and a reference DEM and it can
score it, whoever produced the classification.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from shapely import wkt as shapely_wkt
from shapely.geometry import shape

GROUND_CLASS = 2
CSV_COLUMNS = ("VendorClass", "Classification", "HeightAboveGround", "X", "Y", "Z")
DTM_NODATA = -9999.0


class PdalError(RuntimeError):
    """PDAL exited non-zero; message carries the tail of its stderr."""


@dataclass(frozen=True)
class Segment:
    """One polygon of the segmentation, with its cloud cropped and cached."""

    name: str
    wkt: str
    cloud: Path  # LAS cropped to this polygon, produced once


@dataclass
class Evaluation:
    """Raw outputs of one filter run, before any criterion is computed."""

    vendor_class: np.ndarray
    classification: np.ndarray
    height_above_reference: np.ndarray
    xyz: np.ndarray
    dtm: np.ndarray  # 2-D, NaN where nodata AND NaN outside the segment polygon
    mask: np.ndarray  # 2-D bool, True inside the segment polygon
    reference: np.ndarray  # vendor DEM resampled onto the DTM grid, same masking
    dtm_transform: rasterio.Affine
    dtm_resolution_m: float
    seconds: float

    @property
    def is_ground(self) -> np.ndarray:
        return self.classification == GROUND_CLASS

    @property
    def was_vendor_ground(self) -> np.ndarray:
        return self.vendor_class == GROUND_CLASS


def _python_pdal():
    """PDAL's Python bindings, if this environment has them (the container does)."""
    try:
        import pdal  # noqa: F401
        return pdal
    except ImportError:
        return None


def run_pdal(stages: list[dict], label: str) -> None:
    """Execute a pipeline given as a list of stage dicts."""
    proc = subprocess.run(
        ["pdal", "pipeline", "--stdin"],
        input=json.dumps({"pipeline": stages}),
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise PdalError(f"pdal failed on {label}:\n{proc.stderr[-2000:]}")


def crop_segments(
    cloud: Path,
    segmentation_geojson: Path,
    workdir: Path,
    name_property: str = "zone",
) -> list[Segment]:
    """Cut the cloud once per segment polygon; each evaluation reads its crop.

    The segmentation is expected in the cloud's CRS. Cropping once rather than
    per evaluation is what makes a 60-evaluation search cost minutes instead
    of an hour on a large cloud.
    """
    features = json.loads(segmentation_geojson.read_text())["features"]
    segments = []
    for index, feature in enumerate(features):
        name = str(feature.get("properties", {}).get(name_property, f"segment{index}"))
        wkt = shape(feature["geometry"]).wkt
        out = workdir / f"segment_{name}.las"
        if not out.exists():
            run_pdal(
                [
                    {"type": "readers.las", "filename": str(cloud)},
                    {"type": "filters.crop", "polygon": wkt},
                    {"type": "writers.las", "filename": str(out), "forward": "all"},
                ],
                label=f"crop {name}",
            )
        segments.append(Segment(name=name, wkt=wkt, cloud=out))
    return segments


def evaluate(
    segment: Segment,
    filter_stage: dict,
    prefilter: list[dict],
    reference_dem: Path,
    dtm_resolution_m: float,
    scratch: Path | None = None,
) -> Evaluation:
    """Filter one segment with one parameter set and read everything back."""
    import time

    scratch_ctx = tempfile.TemporaryDirectory(dir=scratch) if scratch else tempfile.TemporaryDirectory()
    with scratch_ctx as tmp:
        tmp = Path(tmp)
        dtm_path = tmp / "dtm.tif"
        csv_path = tmp / "pts.csv"
        stages = [
            {"type": "readers.las", "filename": str(segment.cloud)},
            {"type": "filters.ferry", "dimensions": "Classification=>VendorClass"},
            *prefilter,
            filter_stage,
            {"type": "filters.hag_dem", "raster": str(reference_dem), "zero_ground": False},
            {
                "type": "writers.gdal",
                "filename": str(dtm_path),
                "resolution": dtm_resolution_m,
                "output_type": "idw",
                # 3-cell IDW window: fills one-cell gaps from neighbours, does
                # not paper over larger holes, which coverage must still see.
                "window_size": 3,
                "gdaldriver": "GTiff",
                "where": f"Classification == {GROUND_CLASS}",
                "nodata": DTM_NODATA,
            },
            {
                "type": "writers.text",
                "filename": str(csv_path),
                "order": ",".join(CSV_COLUMNS),
                "keep_unspecified": False,
                "write_header": True,
                "precision": 3,
            },
        ]
        started = time.perf_counter()
        pdal_py = _python_pdal()
        if pdal_py is not None:
            # In-process: the point arrays come back as numpy without a CSV
            # round-trip. On a 30 M-point tile the text write plus np.loadtxt
            # was most of a 348 s evaluation; the filter itself is ~2 min.
            stages_np = [st for st in stages if st.get("type") != "writers.text"]
            pipe = pdal_py.Pipeline(json.dumps(stages_np))
            pipe.execute()
            arr = pipe.arrays[0]
            table = np.column_stack([arr[c].astype(float) for c in CSV_COLUMNS])
        else:
            run_pdal(stages, label=f"evaluate {segment.name} {filter_stage}")
            table = np.loadtxt(csv_path, delimiter=",", skiprows=1, ndmin=2)
        seconds = time.perf_counter() - started
        with rasterio.open(dtm_path) as src:
            dtm = src.read(1).astype(float)
            dtm[dtm == src.nodata] = np.nan
            transform = src.transform
            # writers.gdal rasterizes the bounding box of the cropped points.
            # A segment that is 115 small polygons fills a few percent of its
            # own bbox, so every surface criterion must see only the inside.
            mask = geometry_mask([shapely_wkt.loads(segment.wkt)], out_shape=dtm.shape,
                                 transform=transform, invert=True)
            dtm[~mask] = np.nan
        # The vendor's surface on this exact grid: criteria that ask "what did
        # the filter change relative to the vendor" difference against it.
        reference = _reference_window(reference_dem, dtm.shape, transform)
        reference[~mask] = np.nan

    return Evaluation(
        vendor_class=table[:, 0].astype(int),
        classification=table[:, 1].astype(int),
        height_above_reference=table[:, 2],
        xyz=table[:, 3:6],
        dtm=dtm,
        mask=mask,
        reference=reference,
        dtm_transform=transform,
        dtm_resolution_m=dtm_resolution_m,
        seconds=seconds,
    )


def _reference_window(reference_dem: Path, shape: tuple[int, int], transform: rasterio.Affine) -> np.ndarray:
    """Read the reference DEM over the candidate DTM's footprint, onto its grid."""
    from rasterio.windows import from_bounds

    height, width = shape
    bounds = (transform.c, transform.f + transform.e * height,
              transform.c + transform.a * width, transform.f)
    with rasterio.open(reference_dem) as src:
        window = from_bounds(*bounds, transform=src.transform)
        ref = src.read(1, window=window, boundless=True, fill_value=src.nodata,
                       out_shape=shape).astype(float)
        if src.nodata is not None:
            ref[ref == src.nodata] = np.nan
    return ref


def read_reference_dem(path: Path) -> tuple[np.ndarray, rasterio.Affine]:
    """The vendor's own terrain model, NaN where nodata, for comparison criteria."""
    with rasterio.open(path) as src:
        dem = src.read(1).astype(float)
        if src.nodata is not None:
            dem[dem == src.nodata] = np.nan
        return dem, src.transform
