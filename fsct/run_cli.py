#!/usr/bin/env python3
"""Headless CLI wrapper for FSCT (Forest Structural Complexity Tool).

Replaces the upstream `scripts/run.py` tkinter file-picker with argparse,
so FSCT can be invoked from sbatch scripts inside a container.

Builds the `parameters` dict from CLI args (with sensible defaults for
a dense TLS survey), calls the programmatic `FSCT(...)` entrypoint from
`run_tools.py`. All files live at /opt/FSCT.

Usage:
    python run_cli.py --input /path/to.las --output-dir /path/to/out \\
        [--plot-centre X Y] [--plot-radius R] [--batch-size N] [--gpu]
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from run_tools import FSCT  # noqa: E402  — provided by FSCT scripts dir
from other_parameters import other_parameters  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path,
                   help="Input .las / .laz file")
    p.add_argument("--output-dir", required=True, type=Path,
                   help="Directory to receive FSCT outputs")
    p.add_argument("--plot-centre", nargs=2, type=float, default=None,
                   metavar=("X", "Y"),
                   help="Optional plot centre coords (m)")
    p.add_argument("--plot-radius", type=float, default=0.0,
                   help="If > 0, cylindrically crop around plot_centre")
    p.add_argument("--plot-radius-buffer", type=float, default=0.0,
                   help="Buffer added to plot_radius for tree-aware cropping")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Inference batch size (default 2)")
    p.add_argument("--num-cpu-cores", type=int, default=0,
                   help="Number of CPU cores for multiprocessing (0 = all)")
    p.add_argument("--gpu", action="store_true",
                   help="Use CUDA (default: CPU-only, matches container build)")
    p.add_argument("--slice-thickness", type=float, default=0.15)
    p.add_argument("--slice-increment", type=float, default=0.05)
    p.add_argument("--tree-base-cutoff-height", type=float, default=5.0,
                   help="Min tree-base height above DTM to be kept (m)")
    p.add_argument("--ground-veg-cutoff-height", type=float, default=3.0,
                   help="Points below this are understory, not assigned to trees")
    p.add_argument("--skip-report", action="store_true",
                   help="Skip the make_report step (faster for batch use)")
    return p


def main() -> int:
    args = build_parser().parse_args()
    if not args.input.exists():
        print(f"ERROR: input not found: {args.input}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # FSCT writes outputs alongside the input file (creates <stem>_FSCT_output/
    # in the parent dir). To redirect, stage the input into output_dir.
    staged = args.output_dir / args.input.name
    if not staged.exists():
        shutil.copy2(args.input, staged)

    parameters = dict(
        point_cloud_filename=str(staged),
        plot_centre=args.plot_centre,
        plot_radius=args.plot_radius,
        plot_radius_buffer=args.plot_radius_buffer,
        batch_size=args.batch_size,
        num_cpu_cores=args.num_cpu_cores,
        use_CPU_only=not args.gpu,
        slice_thickness=args.slice_thickness,
        slice_increment=args.slice_increment,
        sort_stems=1,
        height_percentile=100,
        tree_base_cutoff_height=args.tree_base_cutoff_height,
        generate_output_point_cloud=1,
        ground_veg_cutoff_height=args.ground_veg_cutoff_height,
        veg_sorting_range=1.5,
        stem_sorting_range=1.0,
        taper_measurement_height_min=0,
        taper_measurement_height_max=30,
        taper_measurement_height_increment=0.2,
        taper_slice_thickness=0.4,
        delete_working_directory=True,
        minimise_output_size_mode=0,
    )
    parameters.update(other_parameters)

    FSCT(
        parameters=parameters,
        preprocess=1,
        segmentation=1,
        postprocessing=1,
        measure_plot=1,
        make_report=0 if args.skip_report else 1,
        clean_up_files=0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
