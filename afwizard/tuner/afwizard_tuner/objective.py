"""The objective is data the caller writes. This module validates and applies it.

A constraint prunes; the maximize target ranks the survivors; report entries
are computed and returned but never influence the pick. `max_ratio_to_reference`
lets a constraint be stated relative to the vendor DEM, so a threshold need
not be guessed in absolute units.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from . import criteria

ALLOWED_TOP_KEYS = {
    "dataset", "segmentation", "reference_dem", "dtm_resolution_m", "prefilter",
    "filter", "search_space", "fixed", "constraints", "maximize", "report", "search", "crs",
}


class ObjectiveError(ValueError):
    """The objective file is malformed or names something the tuner does not know."""


@dataclass(frozen=True)
class CriterionRef:
    name: str
    params: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        if not self.params:
            return self.name
        inner = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.name}({inner})"


@dataclass(frozen=True)
class Constraint:
    ref: CriterionRef
    minimum: float | None = None
    maximum: float | None = None
    max_ratio_to_reference: float | None = None


@dataclass(frozen=True)
class Objective:
    dataset: Path
    segmentation: Path
    reference_dem: Path
    dtm_resolution_m: float
    # Optional. apply_adaptive_pipeline refuses a segmentation with no spatial
    # reference; this is used when the GeoJSON carries no `crs` member itself.
    crs: str | None
    prefilter: list[dict]
    filter_type: str
    search_space: dict[str, tuple[float, float]]
    fixed: dict
    constraints: list[Constraint]
    maximize: CriterionRef
    report: list[CriterionRef]
    initial_samples: int
    refine_samples: int
    seed: int

    def all_refs(self) -> list[CriterionRef]:
        seen: dict[str, CriterionRef] = {}
        for ref in [c.ref for c in self.constraints] + [self.maximize] + list(self.report):
            seen.setdefault(ref.key, ref)
        return list(seen.values())


def _parse_ref(item) -> CriterionRef:
    if isinstance(item, str):
        return CriterionRef(item)
    if isinstance(item, dict) and len(item) == 1:
        (name, params), = item.items()
        return CriterionRef(name, dict(params or {}))
    raise ObjectiveError(f"cannot read criterion reference {item!r}")


def _check_known(ref: CriterionRef) -> None:
    """A param without a default is required; one with a default may be omitted;
    anything the criterion does not declare is an error."""
    if ref.name not in criteria.REGISTRY:
        raise ObjectiveError(f"unknown criterion {ref.name!r}; run `afwizard-tune list-criteria`")
    spec = criteria.REGISTRY[ref.name].params
    given = set(ref.params)
    unknown = given - set(spec)
    if unknown:
        raise ObjectiveError(f"{ref.name} does not take {sorted(unknown)}; it takes {sorted(spec)}")
    # detrend_m on scale-parameterized criteria is derived from scale_hi_m when
    # absent, so a None default there means optional rather than required.
    required = {k for k, v in spec.items() if v.get("default") is None and k != "detrend_m"}
    missing = required - given
    if missing:
        raise ObjectiveError(f"{ref.name} requires {sorted(missing)}")


def load(path: Path) -> Objective:
    raw = yaml.safe_load(Path(path).read_text())
    unknown = set(raw) - ALLOWED_TOP_KEYS
    if unknown:
        raise ObjectiveError(f"unknown top-level keys {sorted(unknown)}")
    for required in ("dataset", "segmentation", "reference_dem", "filter", "search_space", "maximize"):
        if required not in raw:
            raise ObjectiveError(f"objective is missing {required!r}")

    constraints = []
    for name, spec in (raw.get("constraints") or {}).items():
        ref = _parse_ref({name: spec.pop("params", {})} if "params" in spec else name)
        _check_known(ref)
        constraints.append(Constraint(
            ref=ref,
            minimum=spec.get("min"),
            maximum=spec.get("max"),
            max_ratio_to_reference=spec.get("max_ratio_to_reference"),
        ))

    maximize = _parse_ref(raw["maximize"])
    _check_known(maximize)
    if criteria.REGISTRY[maximize.name].direction == "report":
        raise ObjectiveError(f"{maximize.name} is report-only and cannot be maximized")

    report = [_parse_ref(r) for r in (raw.get("report") or [])]
    for ref in report:
        _check_known(ref)

    space = {}
    for param, bounds in raw["search_space"].items():
        space[param] = (float(bounds["min"]), float(bounds["max"]))

    search = raw.get("search") or {}
    return Objective(
        dataset=Path(raw["dataset"]),
        segmentation=Path(raw["segmentation"]),
        reference_dem=Path(raw["reference_dem"]),
        dtm_resolution_m=float(raw.get("dtm_resolution_m", 1.0)),
        crs=raw.get("crs"),
        prefilter=list(raw.get("prefilter") or []),
        filter_type=raw["filter"],
        search_space=space,
        fixed=dict(raw.get("fixed") or {}),
        constraints=constraints,
        maximize=maximize,
        report=report,
        initial_samples=int(search.get("initial_samples", 40)),
        refine_samples=int(search.get("refine_samples", 20)),
        seed=int(search.get("seed", 0)),
    )


def is_feasible(objective: Objective, scores: dict[str, float], reference_scores: dict[str, float]) -> tuple[bool, list[str]]:
    """Apply every constraint; return the verdict and the reasons for any failure."""
    reasons = []
    for c in objective.constraints:
        value = scores[c.ref.key]
        if value != value:  # NaN never satisfies a constraint
            reasons.append(f"{c.ref.key} is NaN")
            continue
        if c.minimum is not None and value < c.minimum:
            reasons.append(f"{c.ref.key}={value:.4g} < min {c.minimum}")
        if c.maximum is not None and value > c.maximum:
            reasons.append(f"{c.ref.key}={value:.4g} > max {c.maximum}")
        if c.max_ratio_to_reference is not None:
            ref_value = reference_scores.get(c.ref.key)
            if ref_value is None or ref_value != ref_value:
                reasons.append(f"{c.ref.key}: no reference value to compare against")
            elif value > c.max_ratio_to_reference * ref_value:
                reasons.append(
                    f"{c.ref.key}={value:.4g} > {c.max_ratio_to_reference} x reference {ref_value:.4g}"
                )
    return (not reasons), reasons


def objective_value(objective: Objective, scores: dict[str, float]) -> float:
    """The number the pick is made on; sign-flipped so higher is always better."""
    value = scores[objective.maximize.key]
    direction = criteria.REGISTRY[objective.maximize.name].direction
    return value if direction == "higher" else -value
