"""Test the tests: break one property, require the suite to notice.

The four prototypes verify themselves, so "the prototype passed" only means
something if a prototype that should fail does. This probe breaks one named
property at a time and records whether anything catches it.

A mutant is **caught** if any listed prototype exits nonzero or the unit test
module fails. A **survivor** is a property that is asserted somewhere but whose
assertion cannot fail — a vacuous check, not a verified one.

Source files are edited in place and restored in ``finally``; the probe verifies
every digest afterwards and reports loudly if a restore failed. It is deliberately
not part of the test suite: it mutates tracked source, and it refuses to start while
another run holds its lock.

Run: ``uv run python3 experiments/agents/0006-capability-environment/p9/mutation_probe.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


P9_ROOT = Path(__file__).resolve().parent
REPO = P9_ROOT.parents[3]
TEST_MODULE = "experiments.agents.tests.test_p9_shared_arm"


@dataclass(frozen=True)
class Mutant:
    name: str
    breaks: str
    filename: str
    before: str
    after: str
    prototypes: tuple[str, ...]


MUTANTS = (
    Mutant(
        name="omission-notice-frozen-at-first-eviction",
        breaks="the notice names every evicted step, not just the first",
        filename="conversation.py",
        before="        self._evicted_steps.extend(evicted)",
        after="        self._evicted_steps.extend(evicted[:1] if not self._evicted_steps else [])",
        prototypes=("p9_2_context_loss.py",),
    ),
    Mutant(
        name="eviction-notice-never-shown-to-model",
        breaks="the omission is explicit rather than silent",
        filename="conversation.py",
        before="        if notice is not None:\n            output.insert(1, notice)",
        after="        if False:\n            output.insert(1, notice)",
        prototypes=("p9_2_context_loss.py",),
    ),
    Mutant(
        name="model-authored-files-are-wiped-between-rounds",
        breaks="the cumulative workspace loop across action boundaries",
        filename="agent.py",
        before='        payload["environment_observation"]["observations_glob"] = OBSERVATIONS_GLOB\n',
        after=(
            '        payload["environment_observation"]["observations_glob"] = OBSERVATIONS_GLOB\n'
            "        _reserved = ('WORKSPACE.md', 'current_frame.png',\n"
            "                     'current_observation.json', 'public_history.jsonl')\n"
            "        for _stale in self.workspace.root.iterdir():\n"
            "            if _stale.is_file() and _stale.name not in _reserved:\n"
            "                _stale.unlink()\n"
        ),
        prototypes=("p9_1_cumulative_loop.py", "p9_2_context_loss.py", "p9_4_harness_task.py"),
    ),
    Mutant(
        name="observation-filename-width-misstated",
        breaks="the documented observation path matches the files actually written",
        filename="agent.py",
        before="OBSERVATION_STEP_DIGITS = 5",
        after="OBSERVATION_STEP_DIGITS = 4",
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="observation-path-convention-dropped-from-the-contract",
        breaks="the model is told how to construct an observation filename",
        filename="agent.py",
        before="    PATH_CONVENTION + SCHEMA_CONVENTION + RUNTIME_CONVENTION + IMPORT_CONVENTION",
        after="    SCHEMA_CONVENTION + RUNTIME_CONVENTION + IMPORT_CONVENTION",
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="documented-path-expression-is-wrong",
        breaks="the Python expression the contract hands the model resolves to a real file",
        filename="agent.py",
        before="'f\"observations/step-{step:0' + str(OBSERVATION_STEP_DIGITS) + 'd}.json\"'",
        after="'f\"observations/step_{step:0' + str(OBSERVATION_STEP_DIGITS) + 'd}.json\"'",
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="rendered-image-path-dropped-from-the-contract",
        breaks="the model is told the per-step rendered images exist",
        filename="agent.py",
        before='f"as an image at the same name: {OBSERVATION_IMAGE_EXAMPLE}.\\n"',
        after='f".\\n"',
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="workspace-contents-misstated",
        breaks="the contract lists what the workspace already contains",
        filename="agent.py",
        before='    "public_history.jsonl",\n)\n# self._history[-8:]',
        after=')\n# self._history[-8:]',
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="superseded-frames-still-sent",
        breaks="exactly one frame image reaches the model per request",
        filename="agent.py",
        before="            superseded_frame_note=SUPERSEDED_FRAME_NOTE,",
        after="            superseded_frame_note=None,",
        prototypes=("p9_2_context_loss.py",),
    ),
    Mutant(
        name="frame-helper-rejects-a-path",
        breaks="the helper accepts a file path as well as a step number",
        filename="frame_tools.py",
        before="    if isinstance(step_or_path, str):\n        return step_or_path",
        after="    if False:\n        return step_or_path",
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="frame-helper-reads-the-stale-layer",
        breaks="the helper reads the current layer of a multi-layer frame",
        filename="frame_tools.py",
        before='    return np.array(observation["pixels"][-1])',
        after='    return np.array(observation["pixels"][0])',
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="coordinate-rulers-removed",
        breaks="the frame the model sees carries its coordinate rulers",
        filename="annotated_frame.py",
        before="MARGIN = 24",
        after="MARGIN = 0",
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="notes-block-not-echoed",
        breaks="a conclusion written before an eviction is still in context after it",
        filename="agent.py",
        before='        payload["notes"] = self._read_notes()',
        after='        payload["notes"] = {}',
        prototypes=("p9_2_context_loss.py",),
    ),
    Mutant(
        name="notes-truncation-silent",
        breaks="notes truncation is stated rather than silent",
        filename="agent.py",
        before='            "truncated": truncated,',
        after='            "truncated": False,',
        prototypes=(),
    ),
    Mutant(
        name="duplicated-payload-mirrors-restored",
        breaks="the observation payload carries each key once",
        filename="agent.py",
        before="            if key in payload and key in payload[\"environment_observation\"]:",
        after="            if False:",
        prototypes=("p9_0_shared_arm_seam.py",),
    ),
    Mutant(
        name="current-layer-index-misstated",
        breaks="the documented current layer is the one the harness renders",
        filename="agent.py",
        before="CURRENT_LAYER_INDEX = -1",
        after="CURRENT_LAYER_INDEX = 0",
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="recent-transitions-cap-misstated",
        breaks="the documented recent_transitions cap is the real one",
        filename="agent.py",
        before="RECENT_TRANSITIONS_LIMIT = 8",
        after="RECENT_TRANSITIONS_LIMIT = 12",
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="package-allowlist-misstated",
        breaks="the packages the contract names are the ones the sandbox has",
        filename="agent.py",
        before="AVAILABLE_IMPORTS = tuple(sorted(roots[0] for roots in MODEL_RUNTIME_DISTRIBUTIONS.values()))",
        after='AVAILABLE_IMPORTS = ("numpy", "pandas")',
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="observation-schema-keys-misstated",
        breaks="the documented observation keys match a real observation file",
        filename="agent.py",
        before='    return tuple(sorted({*observation_metadata(probe), "pixels", "step"}))',
        after='    return tuple(sorted({*observation_metadata(probe), "pixels"}))',
        prototypes=("p9_1_cumulative_loop.py",),
    ),
    Mutant(
        name="transport-retries-fatal-errors-too",
        breaks="only retryable transport failures are retried",
        filename="transport.py",
        before="                if kind not in RETRIED_TRANSPORT_KINDS or attempt >= self.max_retries:",
        after="                if attempt >= self.max_retries:",
        prototypes=("p9_3_failure_recovery.py",),
    ),
    Mutant(
        name="malformed-response-bodies-treated-as-fatal-again",
        breaks="a whitespace response body is retried instead of killing the run",
        filename="failures.py",
        before='    if http_status is None and response_body is not None:\n        return "transport_malformed_response"',
        after='    if False:\n        return "transport_malformed_response"',
        prototypes=(),
    ),
    Mutant(
        name="transport-retry-never-logged",
        breaks="a retry is reconstructable from the host log",
        filename="transport.py",
        before='                self.process_log.write(\n                    "model_transport_attempt_failed",',
        after='                _ = (\n                    "model_transport_attempt_failed",',
        prototypes=("p9_3_failure_recovery.py",),
    ),
    Mutant(
        name="timeout-does-not-report-restart",
        breaks="a timeout is classified as an interpreter restart with state loss",
        filename="failures.py",
        before='    restarted = bool(output.get("workspace_restarted"))',
        after="    restarted = False",
        prototypes=("p9_3_failure_recovery.py",),
    ),
    Mutant(
        name="sentinel-capability-does-nothing",
        breaks="a declared capability actually moves a declared path",
        filename="agent.py",
        before='        if "sentinel_probe_note" in self.capability_names:',
        after="        if False:",
        prototypes=("p9_0_shared_arm_seam.py",),
    ),
    Mutant(
        name="treatment-diverges-on-a-path-the-manifest-lacks",
        breaks="the observed request-payload diff, not just the declared manifest diff",
        filename="agent.py",
        before='        payload = self._extend_payload(json.loads(message["content"][0]["text"]))\n',
        after=(
            '        payload = self._extend_payload(json.loads(message["content"][0]["text"]))\n'
            "        if self.capability_names:\n"
            '            message["name"] = "treatment"\n'
        ),
        prototypes=("p9_0_shared_arm_seam.py",),
    ),
    Mutant(
        name="arm-manifest-omits-the-system-prompt",
        breaks="the verifier sees the prompt an arm actually runs",
        filename="arms.py",
        before='    system_prompt = str(agent.conversation.system_message["content"])',
        after='    system_prompt = ""',
        prototypes=("p9_0_shared_arm_seam.py",),
    ),
)


def _environment() -> dict[str, str]:
    return {
        **os.environ,
        "UV_CACHE_DIR": "/tmp/uv-cache",
        "MPLCONFIGDIR": "/tmp/mpl-cache",
        # Mutant runs must never land on the committed evidence path.
        "P9_ARTIFACTS_DIR": _SCRATCH_ARTIFACTS,
    }


def _run(command: list[str]) -> bool:
    """Return True when the command passed."""

    completed = subprocess.run(
        command, cwd=REPO, capture_output=True, text=True, env=_environment(), timeout=900
    )
    return completed.returncode == 0


def _digests() -> dict[str, str]:
    return {
        path.name: sha256(path.read_bytes()).hexdigest()
        for path in sorted(P9_ROOT.glob("*.py"))
        if path.name != "mutation_probe.py"
    }


LOCK = P9_ROOT / ".mutation-probe.lock"
_SCRATCH_ARTIFACTS = tempfile.mkdtemp(prefix="p9-probe-artifacts-")


def main() -> int:
    # Two concurrent runs restore from each other's stale snapshots and can leave
    # mutated source behind; a test run alongside one fails for no real reason.
    run_lock = P9_ROOT / ".arm-run.lock"
    if run_lock.exists():
        print(
            f"a live arm run holds {run_lock}; mutating source now would contaminate it",
            file=sys.stderr,
        )
        return 1
    if LOCK.exists():
        print(f"another mutation probe holds {LOCK}; refusing to run", file=sys.stderr)
        return 1
    LOCK.write_text(str(os.getpid()), encoding="utf-8")
    try:
        return _execute_probe()
    finally:
        LOCK.unlink(missing_ok=True)


def _execute_probe() -> int:
    baseline = _digests()
    results: list[dict[str, object]] = []
    restored = True

    for mutant in MUTANTS:
        path = P9_ROOT / mutant.filename
        original = path.read_text(encoding="utf-8")
        if mutant.before not in original:
            results.append(
                {
                    "mutant": mutant.name,
                    "breaks": mutant.breaks,
                    "status": "stale_patch",
                    "detail": f"{mutant.filename} no longer contains the mutated text",
                }
            )
            print(f"STALE   {mutant.name}", flush=True)
            continue
        path.write_text(original.replace(mutant.before, mutant.after, 1), encoding="utf-8")
        try:
            caught_by = [
                name for name in mutant.prototypes if not _run(
                    ["uv", "run", "python3", str(P9_ROOT / name)]
                )
            ]
            unit_tests_failed = not _run(
                ["uv", "run", "python3", "-m", "unittest", TEST_MODULE]
            )
        finally:
            path.write_text(original, encoding="utf-8")
        if unit_tests_failed:
            caught_by.append(TEST_MODULE)
        status = "caught" if caught_by else "survived"
        results.append(
            {
                "mutant": mutant.name,
                "breaks": mutant.breaks,
                "file": mutant.filename,
                "status": status,
                "caught_by": caught_by,
                "missed_by": [name for name in mutant.prototypes if name not in caught_by],
            }
        )
        print(f"{'caught ' if status == 'caught' else 'SURVIVED'} {mutant.name}", flush=True)

    if _digests() != baseline:
        restored = False
        print("\nSOURCE WAS NOT RESTORED — check `git diff` before trusting anything", file=sys.stderr)

    survivors = [item for item in results if item["status"] == "survived"]
    stale = [item for item in results if item["status"] == "stale_patch"]
    report = {
        "schema": "p9-mutation-probe-v1",
        "mutants": len(MUTANTS),
        "caught": len(results) - len(survivors),
        "survivors": [item["mutant"] for item in survivors],
        "stale": [item["mutant"] for item in stale],
        "sources_restored": restored,
        "results": results,
    }
    destination = P9_ROOT / "artifacts" / "p9-mutation-probe" / "results.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\n{report['caught']}/{report['mutants']} mutants caught -> {destination}")
    for item in survivors:
        print(f"  SURVIVED: {item['mutant']} — {item['breaks']}")
    for item in stale:
        print(f"  STALE: {item['mutant']} — the probe no longer tests this")
    return 0 if not survivors and not stale and restored else 1


if __name__ == "__main__":
    raise SystemExit(main())
