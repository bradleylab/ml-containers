"""The vocabulary an agent composes an objective from.

Every criterion is one number computed from an Evaluation, and carries the
prose an agent needs to decide whether to use it: what it measures, which
direction is better, which failure it catches, and what kind of task cares.
That metadata is the product as much as the arithmetic -- `list_criteria()`
is what a caller reads before writing an objective.

Surface criteria work on the DTM the filter produced; point criteria on the
per-point columns. Nothing here knows what "good" means.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .backend import Evaluation

# Semivariogram lags are integer cell offsets; this many cells is enough to
# reach a feature-scale sill on a 1 m grid without walking off small segments.
MAX_LAG_CELLS = 40
# A surface criterion looks at relief RELATIVE to a smooth trend, else it
# measures the hill the site sits on rather than the lumps on it. The trend
# window defaults to this multiple of the feature scale: wide enough that a
# feature is not itself absorbed into the trend, narrow enough to follow the
# terrain. Verified on the Greenwood vendor DEM: raw gamma(2..10 m) is
# 0.01-0.22 m2 of topography; detrended at 30 m it is 0.028 m2 of relief.
DETREND_SCALE_FACTOR = 3.0


@dataclass(frozen=True)
class Criterion:
    name: str
    direction: str  # "higher" | "lower" | "report"
    measures: str
    catches: str
    cares: str
    # name -> {"doc": str, "default": value or None}; a param with a default may
    # be omitted from an objective, one without must be given.
    params: dict = field(default_factory=dict)
    surface: bool = False  # computed from the DTM (has a vendor-DEM counterpart) vs from points
    compute: Callable[..., float] = field(repr=False, compare=False, default=None)

    def defaults(self) -> dict:
        return {k: v["default"] for k, v in self.params.items() if v.get("default") is not None}

    def describe(self) -> dict:
        return {
            "name": self.name,
            "direction": self.direction,
            "measures": self.measures,
            "catches": self.catches,
            "cares": self.cares,
            "surface": self.surface,
            "params": self.params,
        }


REGISTRY: dict[str, Criterion] = {}


def _register(criterion: Criterion) -> Criterion:
    REGISTRY[criterion.name] = criterion
    return criterion


def _semivariogram(grid: np.ndarray, max_lag: int) -> np.ndarray:
    """Isotropic empirical semivariogram of a 2-D grid with NaN holes.

    gamma(h) = 0.5 * mean((z(x) - z(x+h))^2) over valid pairs, averaged across
    the two axes. Returned as an array indexed by lag in cells; index 0 is 0.
    """
    gamma = np.zeros(max_lag + 1)
    for lag in range(1, max_lag + 1):
        pairs = []
        for a, b in ((grid[:, lag:], grid[:, :-lag]), (grid[lag:, :], grid[:-lag, :])):
            diff = a - b
            pairs.append(diff[np.isfinite(diff)])
        joined = np.concatenate(pairs)
        gamma[lag] = 0.5 * np.mean(joined**2) if joined.size else np.nan
    return gamma


def _nugget(gamma: np.ndarray) -> float:
    """Adjacent-cell semivariance, gamma at one cell.

    Not extrapolated to lag 0: on an interpolated DTM gamma grows faster than
    linearly, so the intercept goes negative and floors at 0 -- which made the
    nugget constraint vacuous on the first run. gamma(1) is the speckle
    measure itself and is what a vegetation leak inflates.
    """
    return float(gamma[1])


def _moving_mean_nan(grid: np.ndarray, size: int) -> np.ndarray:
    """Moving mean over VALID cells only (normalized convolution).

    Filling holes with a constant before smoothing puts a fake step at every
    hole edge. On a segment that is 115 polygon fragments, most cells are near
    an edge, and the residual then measures the fill rather than the terrain --
    which is what made the first two runs' structured variance flat.
    """
    from scipy.ndimage import uniform_filter

    valid = np.isfinite(grid).astype(float)
    total = uniform_filter(np.where(valid > 0, grid, 0.0), size=size, mode="constant", cval=0.0)
    count = uniform_filter(valid, size=size, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def _detrend_grid(grid: np.ndarray, detrend_m: float, resolution_m: float) -> np.ndarray:
    size = max(3, int(round(detrend_m / resolution_m)) | 1)  # odd
    return grid - _moving_mean_nan(grid, size)


def _detrend(evaluation: Evaluation, detrend_m: float) -> np.ndarray:
    return _detrend_grid(evaluation.dtm, detrend_m, evaluation.dtm_resolution_m)


def _difference_from_reference(evaluation: Evaluation) -> np.ndarray:
    """Candidate minus vendor surface: what the filter changed, cell by cell."""
    return evaluation.dtm - evaluation.reference


def _resolve_detrend(scale_hi_m: float, detrend_m: float | None) -> float:
    return DETREND_SCALE_FACTOR * scale_hi_m if detrend_m is None else detrend_m


def _lags_for_scale(evaluation: Evaluation, scale_lo_m: float, scale_hi_m: float) -> tuple[int, int]:
    res = evaluation.dtm_resolution_m
    lo = max(1, int(round(scale_lo_m / res)))
    hi = min(MAX_LAG_CELLS, int(round(scale_hi_m / res)))
    if hi < lo:
        raise ValueError(f"scale window {scale_lo_m}..{scale_hi_m} m is empty at {res} m cells")
    return lo, hi


def recall_vendor_ground(evaluation: Evaluation) -> float:
    vendor = evaluation.was_vendor_ground
    if not vendor.any():
        return float("nan")
    return float((vendor & evaluation.is_ground).sum() / vendor.sum())


def added_points(evaluation: Evaluation) -> float:
    return float((evaluation.is_ground & ~evaluation.was_vendor_ground).sum())


def coverage(evaluation: Evaluation) -> float:
    """Fraction of in-polygon cells that hold at least one ground POINT.

    Counted from the points, not read off the DTM: the rasterizer's IDW window
    fills every cell at 1 m from 1.4 pts/m2, so a DTM-based count saturated at
    1.0 for every candidate and measured the interpolator rather than the data.
    """
    ground = evaluation.is_ground
    if not ground.any() or not evaluation.mask.any():
        return float("nan")
    t = evaluation.dtm_transform
    x, y = evaluation.xyz[ground, 0], evaluation.xyz[ground, 1]
    col = np.floor((x - t.c) / t.a).astype(int)
    row = np.floor((y - t.f) / t.e).astype(int)
    rows, cols = evaluation.dtm.shape
    ok = (row >= 0) & (row < rows) & (col >= 0) & (col < cols)
    hit = np.zeros(evaluation.dtm.shape, dtype=bool)
    hit[row[ok], col[ok]] = True
    return float(hit[evaluation.mask].mean())


def _hag_percentile(evaluation: Evaluation, p: float) -> float:
    added = evaluation.is_ground & ~evaluation.was_vendor_ground
    if not added.any():
        return float("nan")
    return float(np.percentile(evaluation.height_above_reference[added], p))


def height_above_reference_p90(evaluation: Evaluation) -> float:
    return _hag_percentile(evaluation, 90)


def height_above_reference_p99(evaluation: Evaluation) -> float:
    return _hag_percentile(evaluation, 99)


def nugget(evaluation: Evaluation, detrend_m: float) -> float:
    return _nugget(_semivariogram(_detrend(evaluation, detrend_m), 1))


def structured_variance(evaluation: Evaluation, scale_lo_m: float, scale_hi_m: float,
                        detrend_m: float | None = None) -> float:
    lo, hi = _lags_for_scale(evaluation, scale_lo_m, scale_hi_m)
    gamma = _semivariogram(_detrend(evaluation, _resolve_detrend(scale_hi_m, detrend_m)), hi)
    return float(np.nanmean(gamma[lo : hi + 1]) - _nugget(gamma))


def recovered_variance(evaluation: Evaluation, scale_lo_m: float, scale_hi_m: float) -> float:
    """Coherent relief at feature scale in (candidate - vendor).

    Zero for the vendor's own surface by construction. Positive when the filter
    added spatially structured relief the vendor lacked -- a mound or hollow
    recovered from points the vendor left unclassified. Speckle goes into the
    nugget instead, so this does not reward vegetation.
    """
    lo, hi = _lags_for_scale(evaluation, scale_lo_m, scale_hi_m)
    gamma = _semivariogram(_difference_from_reference(evaluation), hi)
    return float(np.nanmean(gamma[lo : hi + 1]) - _nugget(gamma))


def added_nugget(evaluation: Evaluation) -> float:
    """Adjacent-cell semivariance of (candidate - vendor): speckle the filter added."""
    return _nugget(_semivariogram(_difference_from_reference(evaluation), 1))


def added_nugget_ratio(evaluation: Evaluation, detrend_m: float) -> float:
    """Speckle the filter added, as a fraction of the speckle the vendor surface already carries.

    Dimensionless, so a caller can constrain it without knowing the site's
    units: 0 means no incoherent change from the vendor surface, 1 means the
    filter added as much adjacent-cell noise as the vendor DEM already had.
    """
    vendor = _nugget(_semivariogram(_detrend_grid(evaluation.reference, detrend_m, evaluation.dtm_resolution_m), 1))
    if not vendor > 0:
        return float("nan")
    return added_nugget(evaluation) / vendor


def relief_sign_ratio(evaluation: Evaluation, scale_lo_m: float, scale_hi_m: float,
                      detrend_m: float | None = None) -> float:
    """Positive over negative residual relief at feature scale.

    Ratio > 1 means the retained structure is mostly lumps; < 1 mostly hollows.
    Reported, not optimized: the caller knows which sign the target has, the
    tuner does not.
    """
    residual = _detrend(evaluation, _resolve_detrend(scale_hi_m, detrend_m))
    valid = residual[np.isfinite(residual)]
    pos = np.sum(np.clip(valid, 0, None) ** 2)
    neg = np.sum(np.clip(valid, None, 0) ** 2)
    return float(pos / neg) if neg > 0 else float("inf")


_register(Criterion(
    "recall_vendor_ground", "higher",
    "Fraction of the vendor's ground points the filter also labels ground.",
    "A filter more conservative than the vendor -- it is dropping points a careful classifier was sure of.",
    "Almost every task WHERE THE INPUT CARRIES A VENDOR GROUND CLASS (USGS 3DEP does). NaN on unclassified clouds -- raw drone products have no class 2 -- and a NaN constraint rejects every candidate. Omit it there; the reference-DEM criteria carry the burden instead.",
    compute=recall_vendor_ground,
))
_register(Criterion(
    "added_points", "report",
    "Count of points labelled ground that the vendor did not.",
    "Nothing on its own. A large number is either recovered ground or leaked vegetation; other criteria say which.",
    "Report only. Never maximize this: labelling everything ground wins.",
    compute=added_points,
))
_register(Criterion(
    "coverage", "higher",
    "Fraction of in-polygon cells at the working resolution holding at least one ground point (counted from points, not the interpolated DTM).",
    "A sparse surface that will be interpolated across.",
    "Hydrology, flood modelling, anything wanting a dense terrain model. Pair with a commission constraint or it is gameable.",
    compute=coverage,
))
_register(Criterion(
    "height_above_reference_p90", "lower",
    "90th percentile height of ADDED points above the reference DEM.",
    "Vegetation leaking into the ground class, as a fat upper tail.",
    "Smooth-surface tasks. UNRELIABLE for micro-relief tasks: a 30 cm mound between reference points reads exactly like a bush.",
    compute=height_above_reference_p90,
))
_register(Criterion(
    "height_above_reference_p99", "lower",
    "99th percentile height of ADDED points above the reference DEM.",
    "The worst of the leaked vegetation.",
    "Same caveat as p90; useful as a report alongside it.",
    compute=height_above_reference_p99,
))
_register(Criterion(
    "nugget", "lower",
    "Adjacent-cell semivariance of the DETRENDED filtered DTM: variance between neighbouring cells with no spatial structure.",
    "Speckle. Vegetation leaking in is spatially incoherent and shows up here; real relief does not.",
    "Any micro-relief task, as a constraint relative to the reference DEM's own nugget (max_ratio_to_reference).",
    params={"detrend_m": {"doc": "moving-mean window removed before measuring, metres", "default": 30.0}},
    surface=True,
    compute=nugget,
))
_register(Criterion(
    "structured_variance", "higher",
    "Semivariogram variance across a lag window, above the nugget, on the DETRENDED DTM: spatially coherent relief at feature scale.",
    "Over-smoothing. A filter that flattens mounds or pits collapses this. Without detrending it would measure the hill, not the lumps.",
    "Feature-preservation tasks. Set the scale to the feature: graves 1-3 m, mounds 5-30 m, terraces 20-50 m. Sign-agnostic.",
    params={"scale_lo_m": {"doc": "lower edge of feature size, metres", "default": None},
            "scale_hi_m": {"doc": "upper edge, metres", "default": None},
            "detrend_m": {"doc": "trend window, metres; default 3 x scale_hi_m", "default": None}},
    surface=True,
    compute=structured_variance,
))
_register(Criterion(
    "recovered_variance", "higher",
    "Semivariogram variance at feature scale, above nugget, of (candidate DTM minus vendor DEM).",
    "A filter that recovers no coherent relief beyond the vendor's surface scores 0; over-smoothing cannot score above it.",
    "The sharpest feature-preservation target: it isolates what the filter CHANGED, so terrain and the vendor's own relief cancel out. Prefer it over structured_variance when a reference DEM exists.",
    params={"scale_lo_m": {"doc": "lower edge of feature size, metres", "default": None},
            "scale_hi_m": {"doc": "upper edge, metres", "default": None}},
    surface=True,
    compute=recovered_variance,
))
_register(Criterion(
    "added_nugget", "lower",
    "Adjacent-cell semivariance of (candidate DTM minus vendor DEM): speckle the filter introduced.",
    "Vegetation leaking in, measured directly as incoherent change from the vendor surface.",
    "Pair with recovered_variance as an absolute-units constraint (m2); 0 means the filter changed nothing incoherently.",
    surface=True,
    compute=added_nugget,
))
_register(Criterion(
    "added_nugget_ratio", "lower",
    "added_nugget divided by the vendor DEM's own (detrended) nugget: speckle introduced, relative to speckle already present.",
    "Vegetation leaking in, in units a caller can reason about without knowing the site.",
    "The recommended commission constraint for feature-preservation tasks, e.g. max 0.5. Pair with recovered_variance.",
    params={"detrend_m": {"doc": "trend window for the vendor nugget, metres", "default": 30.0}},
    surface=True,
    compute=added_nugget_ratio,
))
_register(Criterion(
    "relief_sign_ratio", "report",
    "Positive over negative residual relief at feature scale (>1 lumps, <1 hollows).",
    "Whether what survived is mounds or depressions.",
    "Report when the task knows the sign of its target. The tuner does not decide this.",
    params={"scale_lo_m": {"doc": "metres", "default": None},
            "scale_hi_m": {"doc": "metres", "default": None},
            "detrend_m": {"doc": "trend window, metres; default 3 x scale_hi_m", "default": None}},
    surface=True,
    compute=relief_sign_ratio,
))


def list_criteria() -> list[dict]:
    return [c.describe() for c in REGISTRY.values()]


def compute(name: str, evaluation: Evaluation, **params) -> float:
    if name not in REGISTRY:
        raise KeyError(f"unknown criterion {name!r}; known: {sorted(REGISTRY)}")
    crit = REGISTRY[name]
    return crit.compute(evaluation, **{**crit.defaults(), **params})
