"""Hand the picks to AFwizard so the applied filtering carries its provenance.

AFwizard locates a segment's pipeline by a SHA-1 of the pipeline's *metadata*
only (`afwizard.library.metadata_hash`), not of its parameters. Two filters
with different SMRF settings and the same title collide and AFwizard refuses
with "Ambiguous pipeline metadata". So the parameters are written into the
title, which makes every pick's hash distinct and its settings legible in the
segmentation file.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .objective import Objective

PIPELINE_SCHEMA_MAJOR = 0
PIPELINE_SCHEMA_MINOR = 0


def _metadata_hash(metadata: dict) -> str:
    """Replica of afwizard.library.metadata_hash, so this runs without afwizard installed.

    Uses afwizard's own implementation when it is importable, because the hash
    must match what apply_adaptive_pipeline computes.
    """
    try:
        from afwizard.library import metadata_hash
        import pyrsistent

        class _Shim:
            config = pyrsistent.freeze({"metadata": metadata})

        return metadata_hash(_Shim())
    except ImportError:
        mrepr = repr({k: metadata[k] for k in sorted(metadata)})
        return hashlib.sha1(mrepr.encode()).hexdigest()


def _filter_document(objective: Objective, segment: str, params: dict) -> dict:
    settings = " ".join(f"{k}={v}" for k, v in sorted(params.items()))
    filters = [{"_backend": "pdal", **stage} for stage in objective.prefilter]
    filters.append({"_backend": "pdal", "type": objective.filter_type, **params})
    return {
        "_backend": "pipeline",
        "_major": PIPELINE_SCHEMA_MAJOR,
        "_minor": PIPELINE_SCHEMA_MINOR,
        "_variability": [],
        "filters": filters,
        "metadata": {
            "author": "afwizard-tuner",
            "title": f"{objective.filter_type} [{segment}] {settings}",
            "description": f"Chosen by afwizard-tune maximizing {objective.maximize.key}; see summary.json in the run directory.",
            "example_data_url": "",
            "keywords": ["tuned", segment],
        },
    }


def wire(objective: Objective, outdir: Path, apply: bool = False) -> dict:
    summary = json.loads((outdir / "summary.json").read_text())
    library = outdir / "filters"
    library.mkdir(exist_ok=True)

    hashes = {}
    for seg in summary["segments"]:
        if not seg["pick"]:
            continue
        doc = _filter_document(objective, seg["segment"], seg["pick"]["params"])
        (library / f"{seg['segment']}.json").write_text(json.dumps(doc, indent=2))
        hashes[seg["segment"]] = _metadata_hash(doc["metadata"])

    segmentation = json.loads(objective.segmentation.read_text())
    unmatched = []
    for feature in segmentation["features"]:
        name = str(feature.get("properties", {}).get("zone", ""))
        if name in hashes:
            feature.setdefault("properties", {})["pipeline"] = hashes[name]
        else:
            unmatched.append(name)
    wired_path = outdir / "segmentation_with_pipelines.geojson"
    wired_path.write_text(json.dumps(segmentation, indent=2))

    out = {"filter_library": str(library), "segmentation": str(wired_path),
           "pipelines": hashes, "segments_without_pick": unmatched, "applied": None}

    if apply:
        out["applied"] = _apply(objective, library, wired_path, outdir / "applied")
    return out


def _segmentation_crs(objective: Objective, segmentation_geojson: dict) -> str:
    """apply_adaptive_pipeline raises on a segmentation with no spatial reference.

    Precedence: the objective's `crs`, then the GeoJSON's own (non-standard but
    common) `crs.properties.name`. Neither is an error the compute node should
    be the first to report.
    """
    if objective.crs:
        return objective.crs
    named = segmentation_geojson.get("crs", {}).get("properties", {}).get("name")
    if named:
        return named
    raise ValueError(
        "segmentation has no crs member and the objective sets no `crs:`; "
        "apply_adaptive_pipeline needs one (e.g. crs: EPSG:6344)"
    )


def _apply(objective: Objective, library: Path, segmentation: Path, output_dir: Path) -> str:
    from afwizard import DataSet, add_filter_library, apply_adaptive_pipeline, load_segmentation

    crs = _segmentation_crs(objective, json.loads(segmentation.read_text()))
    add_filter_library(str(library))
    dataset = DataSet(filename=str(objective.dataset), spatial_reference=crs)
    seg = load_segmentation(str(segmentation), spatial_reference=crs)
    apply_adaptive_pipeline(dataset=dataset, segmentation=seg, output_dir=str(output_dir),
                            resolution=objective.dtm_resolution_m)
    return str(output_dir)
