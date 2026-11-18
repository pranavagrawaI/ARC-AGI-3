"""Reuse retained full-corpus harness against the generated successor runner."""

import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import runner

HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / "actor-mtp-20260906"
sys.path.append(str(DONOR))
spec = importlib.util.spec_from_file_location(
    "retained_replay_tests", DONOR / "test_local.py"
)
retained = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retained)


def fake_guarded_call(call, process):
    response, elapsed = call()
    # The retained HTTP fixture predates speculative metadata. Add only that
    # fixture field; watchdog behavior has its own real-process test.
    response["choices"][0]["meta_info"] = {"spec_verify_ct": 1}
    return response, elapsed


if __name__ == "__main__":
    suite = unittest.TestSuite(
        [retained.Checks("test_full_runner_keeps_invalid_results_and_runs_both_arms")]
    )
    with (
        patch.object(runner, "install", return_value={}),
        patch.object(runner, "guarded_call", side_effect=fake_guarded_call),
    ):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
