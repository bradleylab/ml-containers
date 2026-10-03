"""Tests for bin/adaf-contract that need neither the model nor the image.

They cover what the program decides on its own: the three options, the
parameters, finding the DTM, locating ADAF's results folder, and the manifest.
Running ADAF itself needs the image and is left to the executor's fixture.

Run from the adaf/ directory with ``python3 -m unittest discover tests``.
"""

import hashlib
import importlib.machinery
import importlib.util
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
from pathlib import Path

PROGRAM = Path(__file__).resolve().parents[1] / "bin" / "adaf-contract"


def load_program():
    loader = importlib.machinery.SourceFileLoader("adaf_contract", str(PROGRAM))
    spec = importlib.util.spec_from_loader("adaf_contract", loader)
    module = importlib.util.module_from_spec(spec)
    sys.modules["adaf_contract"] = module
    loader.exec_module(module)
    return module


contract = load_program()


def write_params(root: Path, params) -> Path:
    path = root / "params.json"
    path.write_text(json.dumps(params))
    return path


class ParamsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def read(self, params):
        return contract.read_params(write_params(self.root, params))

    def test_defaults_are_the_gui_defaults(self):
        params = self.read({})
        self.assertEqual(params, contract.Params("segmentation", "AO", 40.0, 0.5))

    def test_values_arrive_as_strings(self):
        params = self.read(
            {
                "method": "object_detection",
                "feature_class": "barrow",
                "min_area_m2": "25",
                "min_roundness": "0.75",
            }
        )
        self.assertEqual(params, contract.Params("object detection", "barrow", 25.0, 0.75))

    def test_json_numbers_are_accepted(self):
        self.assertEqual(self.read({"min_area_m2": 0}).min_area_m2, 0.0)

    def test_bounds_are_inclusive(self):
        self.assertEqual(self.read({"min_area_m2": "100"}).min_area_m2, 100.0)
        self.assertEqual(self.read({"min_roundness": "0.95"}).min_roundness, 0.95)

    def test_refusals(self):
        cases = {
            "unknown": ({"threshold": "0.5"}, "unknown parameters"),
            "method": ({"method": "Segmentation"}, "method must be one of"),
            "class": ({"feature_class": "mound"}, "feature_class must be one of"),
            "area_high": ({"min_area_m2": "101"}, "min_area_m2 must be between"),
            "area_negative": ({"min_area_m2": "-5"}, "min_area_m2 must be between"),
            "roundness_high": ({"min_roundness": "0.96"}, "min_roundness must be between"),
            "not_a_number": ({"min_area_m2": "forty"}, "min_area_m2 must be a number"),
            "nan": ({"min_roundness": "nan"}, "min_roundness must be between"),
            "not_an_object": (["method"], "must hold a JSON object"),
        }
        for name, (params, message) in cases.items():
            with self.subTest(name):
                with self.assertRaisesRegex(contract.Refused, message):
                    self.read(params)

    def test_unreadable_file_is_refused(self):
        with self.assertRaisesRegex(contract.Refused, "cannot read the parameters file"):
            contract.read_params(self.root / "missing.json")


class FindDtmTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.input_dir = Path(self.tmp.name)
        self.primary = self.input_dir / "primary"
        self.primary.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def touch(self, relative: str) -> Path:
        path = self.primary / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
        return path

    def test_one_geotiff_is_found_beside_its_sidecars(self):
        dtm = self.touch("site/dtm.TIF")
        self.touch("site/dtm.TIF.aux.xml")
        self.assertEqual(contract.find_dtm(self.input_dir), dtm)

    def test_none_or_two_are_refused(self):
        with self.assertRaisesRegex(contract.Refused, "found 0"):
            contract.find_dtm(self.input_dir)
        self.touch("a.tif")
        self.touch("b.tiff")
        with self.assertRaisesRegex(contract.Refused, "found 2"):
            contract.find_dtm(self.input_dir)

    def test_missing_primary_is_refused(self):
        with self.assertRaisesRegex(contract.Refused, "found 0"):
            contract.find_dtm(self.input_dir / "elsewhere")


class OptionsTest(unittest.TestCase):
    def test_exactly_three_options(self):
        args = contract.parse_args(["--input-dir", "i", "--output-dir", "o", "--params-json", "p"])
        self.assertEqual((args.input_dir, args.output_dir), (Path("i"), Path("o")))
        for argv in (
            ["--input-dir", "i", "--output-dir", "o"],
            ["--input-dir", "i", "--output-dir", "o", "--params-json", "p", "--gpu", "0"],
            ["--input", "i", "--output-dir", "o", "--params-json", "p"],
        ):
            with self.subTest(argv=argv), self.assertRaises(SystemExit) as caught:
                contract.parse_args(argv)
            self.assertEqual(caught.exception.code, 2)


class RunDirTest(unittest.TestCase):
    def test_the_one_new_folder_is_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            (output_dir / "older_run").mkdir()
            before = set(output_dir.iterdir())
            with self.assertRaisesRegex(RuntimeError, "found 0"):
                contract.new_run_dir(output_dir, before)
            run_dir = output_dir / "dtm_20260926_101500_seg"
            run_dir.mkdir()
            (output_dir / "stray.txt").write_text("")
            self.assertEqual(contract.new_run_dir(output_dir, before), run_dir)


class DetectionWarningTest(unittest.TestCase):
    def test_messages(self):
        seg = contract.Params("segmentation", "barrow", 40.0, 0.5)
        obj = contract.Params("object detection", "AO", 40.0, 0.5)
        self.assertIn("wrote no vector file", contract.detection_warning(None, seg))
        self.assertIn("min_roundness", contract.detection_warning(0, seg))
        self.assertNotIn("min_roundness", contract.detection_warning(0, obj))
        self.assertIsNone(contract.detection_warning(3, seg))


class ManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output_dir = Path(self.tmp.name)
        self.run_dir = self.output_dir / "dtm_20260926_101500_seg"
        self.run_dir.mkdir()
        (self.run_dir / "logfile.txt").write_text("ADAF log\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_detections_and_log(self):
        gpkg = self.run_dir / "semantic_segmentation.gpkg"
        gpkg.write_bytes(b"not really a geopackage")
        manifest = contract.build_manifest(
            self.output_dir, self.run_dir, "segmentation", ["a warning"]
        )
        self.assertEqual(manifest["contract_version"], 1)
        self.assertEqual(manifest["warnings"], ["a warning"])
        by_role = {entry["role"]: entry for entry in manifest["outputs"]}
        self.assertEqual(set(by_role), {"detections", "log"})
        self.assertEqual(
            by_role["detections"]["path"], "dtm_20260926_101500_seg/semantic_segmentation.gpkg"
        )
        self.assertEqual(by_role["detections"]["format"], "GPKG")
        self.assertNotIn("format", by_role["log"])
        for entry in manifest["outputs"]:
            digest = hashlib.sha256((self.output_dir / entry["path"]).read_bytes()).hexdigest()
            self.assertEqual(entry["sha256"], digest)

    def test_other_methods_file_is_not_listed(self):
        (self.run_dir / "semantic_segmentation.gpkg").write_bytes(b"")
        manifest = contract.build_manifest(self.output_dir, self.run_dir, "object detection", [])
        self.assertEqual([entry["role"] for entry in manifest["outputs"]], ["log"])
        self.assertNotIn("warnings", manifest)

    def test_missing_log_is_an_error(self):
        (self.run_dir / "logfile.txt").unlink()
        with self.assertRaisesRegex(RuntimeError, "logfile.txt"):
            contract.build_manifest(self.output_dir, self.run_dir, "segmentation", [])


class FakeADAFInput:
    """Stands in for adaf_utils.ADAFInput, but refuses a name the real one lacks."""

    # The attributes ADAFInput.__init__ sets at the pinned ADAF commit.
    FIELDS = {
        "input_file_list",
        "vis_exist_ok",
        "save_vis",
        "ml_type",
        "labels",
        "ml_model_custom",
        "custom_model_pth",
        "roundness",
        "min_area",
        "save_ml_output",
        "out_dir",
        "tiles_to_vrt",
        "dem_path",
    }

    def __init__(self):
        self.values = {}

    def update(self, **kwargs):
        unknown = set(kwargs) - self.FIELDS
        if unknown:
            raise AssertionError(f"ADAFInput has no {sorted(unknown)}")
        self.values.update(kwargs)


class RunTest(unittest.TestCase):
    """run() around a stand-in main_routine that writes what the real one writes."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dtm = self.root / "input" / "primary" / "site_dtm.tif"
        self.dtm.parent.mkdir(parents=True)
        self.dtm.write_bytes(b"")
        self.output_dir = self.root / "output"
        self.received = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_main_routine(self, with_vector):
        def main_routine(adaf_input):
            self.received.append(adaf_input.values)
            suffix = "_obj" if adaf_input.values["ml_type"] == "object detection" else "_seg"
            run_dir = Path(adaf_input.values["out_dir"]) / f"site_dtm_20260926_101500{suffix}"
            run_dir.mkdir()
            (run_dir / "logfile.txt").write_text("ADAF log\n")
            if with_vector:
                name = contract.VECTOR_FILES[adaf_input.values["ml_type"]]
                (run_dir / name).write_bytes(b"polygons")
            return ""

        return main_routine

    def run_with(self, params, with_vector, feature_count):
        patches = [
            unittest.mock.patch.object(
                contract,
                "load_adaf",
                return_value=(self.fake_main_routine(with_vector), FakeADAFInput),
            ),
            unittest.mock.patch.object(contract, "check_dtm", return_value=["cell size note"]),
            unittest.mock.patch.object(contract, "count_features", return_value=feature_count),
        ]
        for patch in patches:
            patch.start()
            self.addCleanup(patch.stop)
        contract.run(self.dtm, self.output_dir, params)
        return json.loads((self.output_dir / "run.json").read_text())

    def test_segmentation_with_detections(self):
        params = contract.Params("segmentation", "barrow", 40.0, 0.5)
        manifest = self.run_with(params, with_vector=True, feature_count=2)
        self.assertEqual(
            self.received,
            [
                {
                    "dem_path": str(self.dtm),
                    "out_dir": str(self.output_dir),
                    "vis_exist_ok": False,
                    "save_vis": False,
                    "ml_type": "segmentation",
                    "labels": ["barrow"],
                    "ml_model_custom": "ADAF model",
                    "roundness": 0.5,
                    "min_area": 40.0,
                    "save_ml_output": False,
                }
            ],
        )
        self.assertEqual(
            [(entry["path"], entry["role"]) for entry in manifest["outputs"]],
            [
                ("site_dtm_20260926_101500_seg/semantic_segmentation.gpkg", "detections"),
                ("site_dtm_20260926_101500_seg/logfile.txt", "log"),
            ],
        )
        self.assertEqual(manifest["warnings"], ["cell size note"])

    def test_object_detection_with_nothing_found(self):
        params = contract.Params("object detection", "AO", 40.0, 0.5)
        manifest = self.run_with(params, with_vector=False, feature_count=None)
        self.assertEqual([entry["role"] for entry in manifest["outputs"]], ["log"])
        self.assertEqual(len(manifest["warnings"]), 2)
        self.assertIn("wrote no vector file", manifest["warnings"][1])


class ProgramTest(unittest.TestCase):
    """The program as the executor and the Docker build check invoke it."""

    def run_program(self, *args):
        return subprocess.run(
            [sys.executable, "-Es", str(PROGRAM), *args],
            capture_output=True,
            text=True,
            check=False,  # the test judges the exit status
        )

    def test_help_needs_nothing_from_the_model(self):
        result = self.run_program("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--params-json", result.stdout)

    def test_refusal_exits_2_without_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "input" / "primary").mkdir(parents=True)
            (root / "input" / "primary" / "dtm.tif").write_bytes(b"")
            params = write_params(root, {"threshold": "0.9"})
            result = self.run_program(
                "--input-dir",
                str(root / "input"),
                "--output-dir",
                str(root / "output"),
                "--params-json",
                str(params),
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("unknown parameters", result.stderr)
            self.assertFalse((root / "output" / "run.json").exists())

    def test_run_path_exposes_load_adaf(self):
        # The Dockerfile's empty-environment check loads the program this way.
        self.assertTrue(callable(runpy.run_path(str(PROGRAM))["load_adaf"]))


if __name__ == "__main__":
    unittest.main()
