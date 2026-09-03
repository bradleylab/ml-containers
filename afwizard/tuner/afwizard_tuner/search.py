"""Parameter search, per segment, against an objective.

Latin-hypercube start, constraint pruning, rank by the maximize target, then a
local refinement around the best feasible candidates. Every evaluation is kept
and written out -- the pick is one row of a table the caller is expected to
re-read and possibly overrule. The optimizer is deliberately plain: at a few
seconds per evaluation the objective, not the search, is the hard part.
"""

from __future__ import annotations

import csv
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from . import criteria
from .backend import Evaluation, Segment, crop_segments, evaluate
from .objective import Objective, is_feasible, objective_value

# Refinement draws around each seed within this fraction of each parameter's
# full range; wide enough to leave a poor local optimum, narrow enough to be a
# refinement rather than a second global sweep.
REFINE_SPREAD = 0.15
# How many of the best feasible candidates seed the refinement.
REFINE_SEEDS = 3


@dataclass
class Row:
    segment: str
    phase: str  # "initial" | "refine"
    params: dict
    scores: dict
    feasible: bool
    reasons: list[str]
    objective: float
    seconds: float


@dataclass
class SegmentResult:
    segment: str
    rows: list[Row] = field(default_factory=list)
    reference_scores: dict = field(default_factory=dict)
    pick: Row | None = None


def _filter_stage(objective: Objective, params: dict) -> dict:
    return {"type": objective.filter_type, **objective.fixed, **params}


def _score(objective: Objective, evaluation: Evaluation) -> dict[str, float]:
    return {ref.key: criteria.compute(ref.name, evaluation, **ref.params) for ref in objective.all_refs()}


def _reference_scores(objective: Objective, segment_eval: Evaluation) -> dict[str, float]:
    """Surface criteria of the vendor DEM over the candidate's grid and mask.

    Point criteria have no vendor counterpart, so only surface criteria appear;
    `max_ratio_to_reference` on a point criterion is reported infeasible with a
    reason rather than passing silently. Difference-based criteria are zero here
    by construction.
    """
    ref_eval = Evaluation(
        vendor_class=np.empty(0, int), classification=np.empty(0, int),
        height_above_reference=np.empty(0), xyz=np.empty((0, 3)),
        dtm=segment_eval.reference, mask=segment_eval.mask, reference=segment_eval.reference,
        dtm_transform=segment_eval.dtm_transform,
        dtm_resolution_m=segment_eval.dtm_resolution_m, seconds=0.0,
    )
    return {
        ref_.key: criteria.compute(ref_.name, ref_eval, **ref_.params)
        for ref_ in objective.all_refs()
        if criteria.REGISTRY[ref_.name].surface
    }


def _lhs(objective: Objective, n: int, rng: np.random.Generator) -> list[dict]:
    names = list(objective.search_space)
    lo = np.array([objective.search_space[k][0] for k in names])
    hi = np.array([objective.search_space[k][1] for k in names])
    unit = qmc.LatinHypercube(d=len(names), seed=rng).random(n)
    return [dict(zip(names, (lo + u * (hi - lo)).round(4).tolist())) for u in unit]


def _perturb(objective: Objective, seed: dict, n: int, rng: np.random.Generator) -> list[dict]:
    out = []
    for _ in range(n):
        p = {}
        for k, (lo, hi) in objective.search_space.items():
            step = rng.normal(0.0, REFINE_SPREAD * (hi - lo))
            p[k] = float(np.clip(seed[k] + step, lo, hi).round(4))
        out.append(p)
    return out


def _run_one(objective: Objective, segment: Segment, params: dict, phase: str,
             reference_scores: dict, scratch: Path) -> tuple[Row, Evaluation]:
    evaluation = evaluate(
        segment, _filter_stage(objective, params), objective.prefilter,
        objective.reference_dem, objective.dtm_resolution_m, scratch=scratch,
    )
    scores = _score(objective, evaluation)
    feasible, reasons = is_feasible(objective, scores, reference_scores)
    row = Row(segment.name, phase, params, scores, feasible, reasons,
              objective_value(objective, scores) if feasible else float("-inf"), evaluation.seconds)
    return row, evaluation


def tune_segment(objective: Objective, segment: Segment, scratch: Path, log) -> SegmentResult:
    rng = np.random.default_rng(objective.seed)
    result = SegmentResult(segment=segment.name)

    # Reference scores need one candidate DTM's grid to align to; the first
    # initial sample provides it, and is then scored like any other.
    initial = _lhs(objective, objective.initial_samples, rng)
    first_row, first_eval = _run_one(objective, segment, initial[0], "initial", {}, scratch)
    result.reference_scores = _reference_scores(objective, first_eval)
    feasible, reasons = is_feasible(objective, first_row.scores, result.reference_scores)
    first_row.feasible, first_row.reasons = feasible, reasons
    first_row.objective = objective_value(objective, first_row.scores) if feasible else float("-inf")
    result.rows.append(first_row)
    log(f"[{segment.name}] reference: " + ", ".join(f"{k}={v:.4g}" for k, v in result.reference_scores.items()))
    log(f"[{segment.name}] 1/{objective.initial_samples} {first_row.params} -> {'ok' if feasible else 'X'} {first_row.objective:.4g} ({first_row.seconds:.1f}s)")

    for i, params in enumerate(initial[1:], start=2):
        row, _ = _run_one(objective, segment, params, "initial", result.reference_scores, scratch)
        result.rows.append(row)
        log(f"[{segment.name}] {i}/{objective.initial_samples} {params} -> {'ok' if row.feasible else 'X'} {row.objective:.4g} ({row.seconds:.1f}s)")

    seeds = sorted([r for r in result.rows if r.feasible], key=lambda r: -r.objective)[:REFINE_SEEDS]
    if seeds and objective.refine_samples:
        per_seed = max(1, objective.refine_samples // len(seeds))
        for seed in seeds:
            for params in _perturb(objective, seed.params, per_seed, rng):
                row, _ = _run_one(objective, segment, params, "refine", result.reference_scores, scratch)
                result.rows.append(row)
                log(f"[{segment.name}] refine {params} -> {'ok' if row.feasible else 'X'} {row.objective:.4g} ({row.seconds:.1f}s)")

    feasible_rows = [r for r in result.rows if r.feasible]
    result.pick = max(feasible_rows, key=lambda r: r.objective) if feasible_rows else None
    return result


def tune(objective: Objective, outdir: Path, log=print) -> list[SegmentResult]:
    outdir.mkdir(parents=True, exist_ok=True)
    scratch = outdir / "scratch"
    scratch.mkdir(exist_ok=True)
    started = time.perf_counter()

    segments = crop_segments(objective.dataset, objective.segmentation, scratch)
    log(f"segments: {[s.name for s in segments]}")
    results = [tune_segment(objective, s, scratch, log) for s in segments]

    write_results(objective, results, outdir, time.perf_counter() - started)
    return results


def _pareto_front(objective: Objective, rows: list[Row]) -> list[Row]:
    """Feasible rows not dominated on (maximize target, every 'lower'-direction criterion in the objective).

    The pick is one point on this front. Showing the whole front is how a caller
    sees that, on some sites, the settings recovering the most relief are also
    the ones leaking the most vegetation -- and chooses accordingly.
    """
    axes = [(objective.maximize.key, +1)]
    for ref in objective.all_refs():
        if ref.key != objective.maximize.key and criteria.REGISTRY[ref.name].direction == "lower":
            axes.append((ref.key, -1))

    def vec(r: Row):
        return [sign * r.scores[k] for k, sign in axes]

    feasible = [r for r in rows if r.feasible]
    front = []
    for r in feasible:
        v = vec(r)
        dominated = any(
            all(a >= b for a, b in zip(vec(o), v)) and any(a > b for a, b in zip(vec(o), v))
            for o in feasible if o is not r
        )
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: -r.objective)


def write_results(objective: Objective, results: list[SegmentResult], outdir: Path, seconds: float) -> None:
    keys = [ref.key for ref in objective.all_refs()]
    params = list(objective.search_space)
    with (outdir / "evaluations.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["segment", "phase", *params, *keys, "feasible", "objective", "seconds", "reasons"])
        for res in results:
            for r in res.rows:
                w.writerow([r.segment, r.phase, *[r.params[p] for p in params],
                            *[r.scores[k] for k in keys], int(r.feasible), r.objective, round(r.seconds, 2),
                            "; ".join(r.reasons)])

    summary = {
        "objective": {
            "filter": objective.filter_type, "fixed": objective.fixed,
            "maximize": objective.maximize.key,
            "constraints": [{"criterion": c.ref.key, "min": c.minimum, "max": c.maximum,
                              "max_ratio_to_reference": c.max_ratio_to_reference} for c in objective.constraints],
            "report": [r.key for r in objective.report],
        },
        "elapsed_seconds": round(seconds, 1),
        "segments": [],
    }
    for res in results:
        n_feasible = sum(r.feasible for r in res.rows)
        entry = {
            "segment": res.segment,
            "evaluations": len(res.rows),
            "feasible": n_feasible,
            "reference_scores": res.reference_scores,
            "pick": None,
        }
        if res.pick:
            entry["pick"] = {"params": {**objective.fixed, **res.pick.params}, "scores": res.pick.scores,
                             "objective": res.pick.objective,
                             "rule": f"max {objective.maximize.key} among feasible"}
            entry["pareto_front"] = [
                {"params": {**objective.fixed, **r.params}, "scores": r.scores, "objective": r.objective}
                for r in _pareto_front(objective, res.rows)
            ]
        else:
            entry["why_no_pick"] = "no evaluation satisfied every constraint; loosen one or widen the search space"
            entry["closest"] = [
                {"params": r.params, "reasons": r.reasons}
                for r in sorted(res.rows, key=lambda r: len(r.reasons))[:3]
            ]
        summary["segments"].append(entry)
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
