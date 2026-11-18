import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
PACKAGE = HERE.parents[1] / "deployment/kaggle" / HERE.name


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


PRE = module(HERE / "prelaunch_check.py", "r5_live_prelaunch")
LAUNCH = module(PACKAGE / "launch_once.py", "r5_live_launch")


class TotalAuthorizationPredicate(unittest.TestCase):
    def valid(self):
        return {
            "schema": "one-shot-kaggle-authorization-v1", "identity": HERE.name,
            "manifest_sha256": "m", "card_sha256": "c", "notebook_sha256": "n",
            "target": "owner/target", "used": False, "maximum_pushes": 1, "retries": 0,
            "push_timeout_seconds": 3600, "displayed_gpu_hour_cap": 1.0,
        }

    def accepted(self, value, consumed=False):
        return PRE.authorization_valid(value, "m", "c", "n", "owner/target", consumed)

    def test_valid_exact_binding(self):
        self.assertTrue(self.accepted(self.valid()))

    def test_missing_empty_wrong_type_and_consumed_rejected(self):
        for value in (None, {}, [], "authorization", 1, True):
            self.assertFalse(self.accepted(value))
        self.assertFalse(self.accepted(self.valid(), consumed=True))

    def test_wrong_field_types_rejected(self):
        for key, value in (("used", 0), ("maximum_pushes", True), ("retries", False), ("push_timeout_seconds", 3600.0), ("displayed_gpu_hour_cap", "1.0")):
            candidate = self.valid(); candidate[key] = value
            self.assertFalse(self.accepted(candidate), key)

    def test_every_binding_and_cap_is_required(self):
        for key in self.valid():
            candidate = self.valid(); candidate.pop(key)
            self.assertFalse(self.accepted(candidate), key)
        for key in ("identity", "manifest_sha256", "card_sha256", "notebook_sha256", "target", "schema"):
            candidate = self.valid(); candidate[key] = "wrong"
            self.assertFalse(self.accepted(candidate), key)

    def test_malformed_document_is_total(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "AUTHORIZATION.json"
            path.write_text("{")
            self.assertEqual(PRE.load_document(path), (None, "MALFORMED"))
            path.unlink()
            self.assertEqual(PRE.load_document(path), (None, "MISSING"))

    def test_final_review_requires_exact_document(self):
        exact = PRE.review_expected("m", "c", "n")
        self.assertTrue(PRE.review_valid(exact, "m", "c", "n"))
        for key in exact:
            candidate = dict(exact); candidate[key] = "wrong"
            self.assertFalse(PRE.review_valid(candidate, "m", "c", "n"), key)
        self.assertFalse(PRE.review_valid({**exact, "extra": True}, "m", "c", "n"))


class OneShotLaunchControls(unittest.TestCase):
    def setup_launch(self, temporary):
        root = Path(temporary)
        package = root / "deployment/kaggle" / HERE.name
        analysis = root / "analysis" / HERE.name
        package.mkdir(parents=True); (analysis / "evidence").mkdir(parents=True)
        meta = json.loads((HERE / "SERVER-REFERENCE.json").read_text())
        (package / "kernel-metadata.json").write_text(json.dumps(meta))
        (analysis / "SERVER-REFERENCE.json").write_text(json.dumps(meta))
        (analysis / "AUTHORIZATION.json").write_text("{}")
        (analysis / "REVIEW-RESOLVED.json").write_text("{}")
        return root, package, analysis, meta

    def receipts(self, meta, push_stdout):
        return [
            {"returncode": 0, "stdout": '[{"resource":"GPU","remaining":"2.0h","used":"1.0h"}]', "stderr": "", "arguments": []},
            {"returncode": 0, "stdout": "[]", "stderr": "", "arguments": []},
            {"returncode": 0, "stdout": push_stdout, "stderr": "", "arguments": [], "timed_out": False},
        ]

    def run_main(self, push_stdout, target_present=False):
        temporary = tempfile.TemporaryDirectory()
        root, package, analysis, meta = self.setup_launch(temporary.name)
        rows = self.receipts(meta, push_stdout)
        if target_present:
            rows[1]["stdout"] = json.dumps([{"ref": meta["id"]}])
        calls = []
        def invoke(args, timeout=120):
            calls.append(args); return rows.pop(0)
        patches = [mock.patch.object(LAUNCH, "HERE", package), mock.patch.object(LAUNCH, "REPO", root), mock.patch.object(LAUNCH, "ANALYSIS", analysis), mock.patch.object(LAUNCH, "invoke", side_effect=invoke), mock.patch.object(LAUNCH, "local_gate", return_value={"status": "APPROVED_NOT_LAUNCHED", "manifest_sha256": "m"})]
        for patch in patches: patch.start()
        try:
            try: LAUNCH.main(); error = None
            except Exception as caught: error = caught
        finally:
            for patch in reversed(patches): patch.stop()
        return temporary, analysis, calls, error

    def test_exact_one_push_and_returned_identity(self):
        slug = "pranavagrawai/ls20-construction-policy-live-20260914t093351z"
        url = "https://www.kaggle.com/code/" + slug
        temporary, analysis, calls, error = self.run_main("Kernel version 1 successfully pushed.  Please check progress at " + url + "\n")
        try:
            self.assertIsNone(error)
            self.assertEqual(sum(call[:2] == ["kernels", "push"] for call in calls), 1)
            self.assertEqual(json.loads((analysis / "evidence/LAUNCH-RESULT.json").read_text())["returned_identity"], slug)
        finally: temporary.cleanup()

    def test_target_exists_blocks_before_consumption(self):
        temporary, analysis, calls, error = self.run_main("unused", target_present=True)
        try:
            self.assertIn("FRESH_TARGET_NOT_ABSENT", str(error))
            self.assertFalse((analysis / "LAUNCH-CONSUMED.json").exists())
            self.assertFalse(any(call[:2] == ["kernels", "push"] for call in calls))
        finally: temporary.cleanup()

    def test_current_package_prepared_not_approved(self):
        result = PRE.check(write=False)
        self.assertEqual(result["status"], "PREPARED_NOT_APPROVED")
        self.assertEqual(result["final_review"], "MISSING")
        self.assertEqual(result["pushes"], 0)


if __name__ == "__main__":
    unittest.main()
