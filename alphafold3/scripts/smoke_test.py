"""Offline build-time checks for the AlphaFold 3 software image."""

import shutil
import subprocess
import sys
from importlib import metadata, resources
from pathlib import Path

import alphafold3.constants.converters

REQUIRED_DISTRIBUTIONS = (
    "alphafold3",
    "jax",
    "rdkit",
    "zstandard",
)
EXPECTED_ALPHAFOLD_VERSION = "3.0.4"
REQUIRED_EXECUTABLES = (
    "hmmalign",
    "hmmbuild",
    "hmmsearch",
    "jackhmmer",
    "nhmmer",
)
REQUIRED_CHEMICAL_DATA = (
    "ccd.pickle",
    "chemical_component_sets.pickle",
)
REQUIRED_LICENSE_FILES = (
    Path("/usr/share/licenses/hmmer/HMMER-LICENSE"),
    Path("/usr/share/licenses/hmmer/EASEL-LICENSE"),
    Path("/usr/share/licenses/hmmer/LIBDIVSUFSORT-COPYING"),
)


def main() -> None:
    """Verify package metadata, executables, CLI help, and chemical data."""
    for distribution in REQUIRED_DISTRIBUTIONS:
        print(distribution, metadata.version(distribution))

    alphafold_version = metadata.version("alphafold3")
    if alphafold_version != EXPECTED_ALPHAFOLD_VERSION:
        raise RuntimeError(
            f"Expected AlphaFold {EXPECTED_ALPHAFOLD_VERSION}, got {alphafold_version}"
        )

    for executable in REQUIRED_EXECUTABLES:
        executable_path = shutil.which(executable)
        if executable_path is None:
            raise RuntimeError(f"Missing required executable: {executable}")
        print(executable, executable_path)

    for license_path in REQUIRED_LICENSE_FILES:
        if not license_path.is_file() or license_path.stat().st_size == 0:
            raise RuntimeError(
                f"Missing required redistribution notice: {license_path}"
            )
        print("license", license_path, license_path.stat().st_size)

    converter_data = resources.files(alphafold3.constants.converters)
    for filename in REQUIRED_CHEMICAL_DATA:
        resource = converter_data.joinpath(filename)
        with resources.as_file(resource) as resource_path:
            path = Path(resource_path)
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing generated chemical data: {filename}")
            print(filename, path.stat().st_size)

    help_result = subprocess.run(
        [sys.executable, "run_alphafold.py", "--help"],
        check=False,
        cwd="/app/alphafold",
        capture_output=True,
        text=True,
        timeout=60,
    )
    help_output = help_result.stdout + help_result.stderr
    if help_result.returncode not in (0, 1):
        raise RuntimeError(
            f"run_alphafold.py --help failed: {help_result.stderr[-2000:]}"
        )
    if "--model_dir" not in help_output:
        raise RuntimeError("AlphaFold CLI help is missing --model_dir")
    print("run_alphafold.py --help ok")


if __name__ == "__main__":
    main()
