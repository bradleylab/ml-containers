"""`afwizard-tune`: the surface an agent drives.

    afwizard-tune list-criteria [--json]      what can go in an objective
    afwizard-tune tune --objective F --outdir D
    afwizard-tune wire --objective F --outdir D [--apply]   hand picks to AFwizard
    afwizard-tune describe --outdir D         re-read a finished run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import criteria, objective as objective_mod, search, wire


def cmd_list_criteria(args) -> int:
    items = criteria.list_criteria()
    if args.json:
        print(json.dumps(items, indent=2))
        return 0
    for c in items:
        print(f"{c['name']}  [{c['direction']}]")
        print(f"  measures: {c['measures']}")
        print(f"  catches : {c['catches']}")
        print(f"  cares   : {c['cares']}")
        if c["params"]:
            print(f"  params  : {c['params']}")
        print()
    return 0


def cmd_tune(args) -> int:
    obj = objective_mod.load(Path(args.objective))
    outdir = Path(args.outdir)
    log_path = outdir / "tune.log"
    outdir.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as fh:
        def log(msg: str) -> None:
            print(msg, flush=True)
            fh.write(msg + "\n")
        results = search.tune(obj, outdir, log=log)
    return 0 if all(r.pick for r in results) else 2


def cmd_wire(args) -> int:
    obj = objective_mod.load(Path(args.objective))
    out = wire.wire(obj, Path(args.outdir), apply=args.apply)
    print(json.dumps(out, indent=2))
    return 0


def cmd_describe(args) -> int:
    summary = json.loads((Path(args.outdir) / "summary.json").read_text())
    print(f"maximize {summary['objective']['maximize']}; elapsed {summary['elapsed_seconds']} s")
    for seg in summary["segments"]:
        print(f"\n[{seg['segment']}] {seg['feasible']}/{seg['evaluations']} feasible")
        if seg["pick"]:
            print(f"  pick   : {seg['pick']['params']}")
            print(f"  scores : " + ", ".join(f"{k}={v:.4g}" for k, v in seg["pick"]["scores"].items()))
            front = seg.get("pareto_front", [])
            print(f"  pareto : {len(front)} non-dominated (maximize vs every 'lower' criterion)")
            for r in front[:5]:
                print(f"     {r['params']}  obj={r['objective']:.4g}")
        else:
            print(f"  no pick: {seg['why_no_pick']}")
            for r in seg.get("closest", []):
                print(f"     nearest: {r['params']} -- {'; '.join(r['reasons'])}")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="afwizard-tune", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    lc = sub.add_parser("list-criteria", help="print the criteria vocabulary")
    lc.add_argument("--json", action="store_true")
    lc.set_defaults(func=cmd_list_criteria)

    tn = sub.add_parser("tune", help="run the search")
    tn.add_argument("--objective", required=True)
    tn.add_argument("--outdir", required=True)
    tn.set_defaults(func=cmd_tune)

    wr = sub.add_parser("wire", help="write picks as AFwizard filters and a segmentation with pipeline hashes")
    wr.add_argument("--objective", required=True)
    wr.add_argument("--outdir", required=True)
    wr.add_argument("--apply", action="store_true", help="also run apply_adaptive_pipeline (needs afwizard)")
    wr.set_defaults(func=cmd_wire)

    ds = sub.add_parser("describe", help="summarise a finished run")
    ds.add_argument("--outdir", required=True)
    ds.set_defaults(func=cmd_describe)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
