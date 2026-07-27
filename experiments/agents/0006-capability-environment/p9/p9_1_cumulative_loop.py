"""P9.1 — the cumulative model → Python/files → model loop.

Question: over three recorded observations, can a fixture model inspect an
observation, write a helper module and a notebook, receive the next observation,
reuse both, and then act — with the workspace, the transcript, and the
authoritative record all carrying forward?

The environment is a recorded capture and the model is a fixed script, so
nothing here measures gameplay. It measures whether the loop holds together.

Run: ``uv run python3 experiments/agents/0006-capability-environment/p9/p9_1_cumulative_loop.py``
"""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "p9"

from p9.agent import (
    AVAILABLE_IMPORTS,
    IMPORT_CONVENTION,
    OBSERVATION_FILENAME_EXAMPLE,
    OBSERVATION_IMAGE_EXAMPLE,
    OBSERVATION_PATH_EXPRESSION,
    CURRENT_LAYER_INDEX,
    LAYER_CONVENTION,
    RECENT_TRANSITIONS_LIMIT,
    RESERVED_WORKSPACE_ENTRIES,
    FRAME_TOOLS_CONVENTION,
    HISTORY_CONVENTION,
    IMAGE_CONVENTION,
    WORKSPACE_CONTENTS_CONVENTION,
    OBSERVATION_SCHEMA_KEYS,
    PATH_CONVENTION,
    RUNTIME_CONVENTION,
    SCHEMA_CONVENTION,
)
from p9.arms import build_arm, control_spec
from p9.evidence import Checks, artifacts_root
from p9.failures import classify_tool_result
from p9.fixtures import (
    RecordedEnvironment,
    ScriptedChatClient,
    action_turn,
    default_recording,
    load_recorded_frames,
    tool_turn,
)

from experiments.harness import run_episode


INSPECT = """\
import json
observation = json.load(open('current_observation.json'))
print('INSPECTED', observation['step'], len(observation['pixels']), len(observation['pixels'][0]))
"""

AUTHOR = """\
from pathlib import Path
Path('helpers.py').write_text(
    "import json\\n"
    "def load(step):\\n"
    "    return json.load(open('observations/step-%05d.json' % step))\\n"
    "def colors(observation):\\n"
    "    return sorted({value for layer in observation['pixels'] for row in layer for value in row})\\n"
)
Path('notebook.md').write_text('# notebook\\n- step 0: wrote helpers.load and helpers.colors\\n')
print('AUTHORED', sorted(path.name for path in Path('.').glob('*')))
"""

BARE_IMPORT = "import helpers\nprint('BARE_IMPORT_OK')\n"

REUSE = """\
import sys
sys.path.insert(0, '.')
import helpers
from pathlib import Path
observation = helpers.load(1)
line = '- step 1: reused helpers.colors -> %s' % helpers.colors(observation)
with Path('notebook.md').open('a') as stream:
    stream.write(line + '\\n')
print('REUSED', line)
print('NOTEBOOK_LINES', len(Path('notebook.md').read_text().splitlines()))
"""

SCRIPT = (
    tool_turn(INSPECT),
    tool_turn(AUTHOR),
    action_turn(1, reason="probe after authoring tools"),
    tool_turn(BARE_IMPORT),
    tool_turn(REUSE),
    action_turn(1, reason="probe after reusing tools"),
)


def _tool_outputs(process_log_path: Path) -> list[dict]:
    outputs = []
    for line in process_log_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["type"] == "python_tool":
            outputs.append(
                {
                    "action_step": record["action_step"],
                    "model_round": record["model_round"],
                    "output": json.loads(record["output"]),
                }
            )
    return outputs


def main() -> int:
    checks = Checks(
        prototype="P9.1-cumulative-loop",
        question=(
            "Does inspect -> author -> receive -> reuse -> act hold together over "
            "three recorded observations?"
        ),
    )
    repo = Path(__file__).resolve().parents[4]
    frames = load_recorded_frames(default_recording(repo, minimum_frames=3), count=3, crop=8)

    with TemporaryDirectory(prefix="p9-1-") as scratch:
        client = ScriptedChatClient(SCRIPT)
        spec = control_spec(max_actions=2, max_model_calls=12)
        with build_arm(spec, run_dir=Path(scratch) / "arm", inner_client=client) as arm:
            environment = RecordedEnvironment(frames=frames)
            result = run_episode(environment, arm.agent, max_actions=spec.max_actions)
            workspace_root = arm.workspace.root
            files = arm.workspace.evidence_state()
            outputs = _tool_outputs(arm.run_dir / "process.jsonl")
            payloads = client.request_payloads()
            workspace_observations = sorted(
                path.name for path in (workspace_root / "observations").glob("*.json")
            )
            notebook = files.get("notebook.md", {}).get("text", "")
            contract = (workspace_root / "WORKSPACE.md").read_text(encoding="utf-8")
            # Pin the stated conventions against the sandbox itself, not against a
            # second copy of the same assumption.
            # The helper runs in the sandbox, so verify it there - including on a
            # multi-layer frame, the trap it exists to make unhittable.
            frame_tools_probe = arm.workspace.execute(
                "import sys, json; sys.path.insert(0, '.')\n"
                "import frames, numpy as np\n"
                "g = frames.grid(0)\n"
                "print('STEPS', frames.steps())\n"
                "print('GRID_SHAPE', g.shape)\n"
                "d = frames.diff(0, 1)\n"
                "print('DIFF_KEYS', sorted(d))\n"
                "print('DIFF_COMPARABLE', d['comparable'])\n"
                # a synthetic multi-layer observation: grid() must take the last layer
                "obs = frames.load(0)\n"
                "obs['pixels'] = [[[0]*4]*4, [[9]*4]*4]\n"
                "open('observations/step-00099.json','w').write(json.dumps(obs))\n"
                "print('LAST_LAYER', int(frames.grid(99)[0][0]))\n"
            )
            named_imports = arm.workspace.execute(
                "import " + ", ".join(AVAILABLE_IMPORTS) + "\nprint('NAMED_IMPORTS_OK')"
            )
            unnamed_import = arm.workspace.execute("import sklearn")
            # Pin "the workspace is not empty" against what is actually there before
            # the model's first call, and the recent_transitions cap against a payload
            # built from a history longer than the cap.
            reserved_present = sorted(
                path.name
                for path in workspace_root.iterdir()
                if path.name in RESERVED_WORKSPACE_ENTRIES
            )
            arm.agent._history = [{"step": n} for n in range(RECENT_TRANSITIONS_LIMIT + 4)]
            probe_payload = json.loads(
                arm.agent._observation_message(
                    arm.agent._current, model_round=1, python_calls_this_action=0
                )["content"][0]["text"]
            )
            advertised_recent = len(
                probe_payload["environment_observation"]["recent_transitions"]
            )
            observation_digest = arm.agent._current.digest
            rendered_steps = sorted(
                path.stem for path in (workspace_root / "observations").glob("step-*.png")
            )
            observed_schema_keys = sorted(
                json.loads((workspace_root / "observations" / "step-00000.json").read_text())
            )

    kinds = [classify_tool_result(item["output"]) for item in outputs]
    checks.record(
        "the_script_ran_to_completion",
        client.remaining == 0 and result.actions == 2,
        {"remaining_script_entries": client.remaining, "actions": result.actions},
    )
    checks.record(
        "three_observations_reached_the_model_workspace",
        workspace_observations == ["step-00000.json", "step-00001.json", "step-00002.json"],
        {"observations": workspace_observations},
    )
    checks.record(
        "inspection_at_step_0_succeeded",
        kinds[0].kind == "none" and "INSPECTED" in outputs[0]["output"]["stdout"],
        {"kind": kinds[0].kind, "stdout": outputs[0]["output"]["stdout"].strip()},
    )
    checks.record(
        "helper_and_notebook_were_authored_at_step_0",
        kinds[1].kind == "none" and {"helpers.py", "notebook.md"} <= set(files),
        {"kind": kinds[1].kind, "model_authored_files": sorted(files)},
    )
    checks.record(
        "authoring_and_reuse_happened_on_opposite_sides_of_an_action",
        outputs[1]["action_step"] == 1 and outputs[3]["action_step"] == 2,
        {
            "authored_at_action_step": outputs[1]["action_step"],
            "reused_at_action_step": outputs[3]["action_step"],
        },
    )
    checks.record(
        "helper_reuse_after_the_next_observation_succeeded",
        kinds[3].kind == "none" and "REUSED" in outputs[3]["output"]["stdout"],
        {"kind": kinds[3].kind, "stdout": outputs[3]["output"]["stdout"].strip()},
    )
    checks.record(
        "the_notebook_accumulated_rather_than_being_rewritten",
        notebook.count("\n- ") == 2 and "step 0" in notebook and "step 1" in notebook,
        {"notebook": notebook},
    )
    checks.record(
        "the_helper_read_an_observation_that_arrived_after_it_was_written",
        "REUSED - step 1: reused helpers.colors -> [" in outputs[3]["output"]["stdout"],
        {"stdout": outputs[3]["output"]["stdout"].strip()},
    )
    checks.record(
        "later_requests_carry_the_earlier_turns",
        _turn_steps(payloads[-1]) == {0, 1},
        {"observation_steps_in_final_request": sorted(_turn_steps(payloads[-1]))},
    )
    checks.record(
        "no_transcript_omission_notice_appeared_without_eviction",
        not arm_notice_present(payloads),
        {"max_context_turns": spec.max_context_turns},
    )

    # Substrate finding: the observation filename width was never stated, and a model
    # constructing a path guesses four digits.
    written_widths = {
        len(match.group(1))
        for name in workspace_observations
        if (match := re.fullmatch(r"step-(\d+)\.json", name))
    }
    documented_width = len(
        re.search(r"step-(\d+)\.json", OBSERVATION_FILENAME_EXAMPLE).group(1)
    )
    checks.record(
        "the_documented_observation_filename_matches_the_files_actually_written",
        written_widths == {documented_width},
        {
            "written_widths": sorted(written_widths),
            "documented": OBSERVATION_FILENAME_EXAMPLE,
        },
    )
    checks.record(
        "the_contract_lists_exactly_what_the_workspace_already_contains",
        reserved_present == sorted(RESERVED_WORKSPACE_ENTRIES)
        and WORKSPACE_CONTENTS_CONVENTION.strip() in contract,
        {"present": reserved_present, "documented": sorted(RESERVED_WORKSPACE_ENTRIES)},
    )
    # A multi-layer frame must resolve to the layer the harness actually renders.
    import numpy as np
    from llm_repl_agent.observations import pixels_to_png, representation_descriptor

    layered = np.zeros((6, 4, 4), dtype=int)
    layered[CURRENT_LAYER_INDEX] = 7
    descriptor = representation_descriptor(layered, pixels_to_png(layered))
    rendered_layer = descriptor["rendered_image"]["source_grid_layer"]
    from p9.annotated_frame import CELL_PIXELS, MARGIN, annotate, annotation_descriptor
    from PIL import Image as _Image
    from io import BytesIO as _BytesIO

    probe_grid = np.zeros((1, 8, 8), dtype=int)
    with _Image.open(_BytesIO(annotate(probe_grid))) as annotated_image:
        annotated_size = annotated_image.size
    checks.record(
        "the_frame_the_model_sees_carries_its_coordinate_rulers",
        annotated_size == (8 * CELL_PIXELS + MARGIN, 8 * CELL_PIXELS + MARGIN)
        and 64 * CELL_PIXELS + MARGIN <= 512
        and annotation_descriptor()["x_axis"].startswith("column")
        and IMAGE_CONVENTION.strip() in contract,
        {
            "annotated_size": annotated_size,
            "margin": MARGIN,
            "full_frame_px": 64 * CELL_PIXELS + MARGIN,
        },
    )
    checks.record(
        "the_documented_current_layer_is_the_one_the_harness_renders",
        rendered_layer == len(layered) + CURRENT_LAYER_INDEX
        and LAYER_CONVENTION.strip() in contract,
        {
            "documented_index": CURRENT_LAYER_INDEX,
            "rendered_source_grid_layer": rendered_layer,
            "layers": len(layered),
        },
    )
    checks.record(
        "the_documented_recent_transitions_cap_is_the_real_one",
        advertised_recent == RECENT_TRANSITIONS_LIMIT
        and HISTORY_CONVENTION.strip() in contract,
        {"documented": RECENT_TRANSITIONS_LIMIT, "observed": advertised_recent},
    )
    checks.record(
        "the_digest_is_not_a_recomputable_hash_of_the_observation_file",
        observation_digest
        != sha256(
            (workspace_observations and files.get("observations") or {}).get("text", "").encode()
        ).hexdigest()[:16],
        {"digest": observation_digest, "documented_as": "opaque state identifier"},
    )
    checks.record(
        "every_observation_has_a_rendered_image_beside_it",
        rendered_steps == [Path(name).stem for name in workspace_observations],
        {"json": workspace_observations, "png": rendered_steps},
    )
    # Evaluate the expression the contract hands the model and require the result to
    # exist. A documented pattern can be mis-composed; a documented expression cannot.
    documented_path = eval(OBSERVATION_PATH_EXPRESSION, {"step": 0})  # noqa: S307
    checks.record(
        "the_documented_python_expression_resolves_to_a_real_file",
        Path(documented_path).name in workspace_observations
        and OBSERVATION_PATH_EXPRESSION in contract,
        {
            "expression": OBSERVATION_PATH_EXPRESSION,
            "resolves_to": documented_path,
            "files_written": workspace_observations,
        },
    )
    checks.record(
        "the_rendered_image_path_is_stated_in_the_contract",
        OBSERVATION_IMAGE_EXAMPLE in contract,
        {"example": OBSERVATION_IMAGE_EXAMPLE},
    )
    checks.record(
        "the_observation_path_convention_is_stated_in_the_contract",
        PATH_CONVENTION.strip() in contract,
        {"convention": PATH_CONVENTION.strip()},
    )

    checks.record(
        "the_provided_frame_helper_runs_in_the_sandbox",
        frame_tools_probe.ok
        and "DIFF_COMPARABLE True" in frame_tools_probe.stdout
        and "DIFF_KEYS" in frame_tools_probe.stdout,
        {"stdout": frame_tools_probe.stdout.strip()[:300], "stderr": frame_tools_probe.stderr[:200]},
    )
    checks.record(
        "the_helper_reads_the_current_layer_of_a_multi_layer_frame",
        "LAST_LAYER 9" in frame_tools_probe.stdout,
        {"line": _line(frame_tools_probe.stdout, "LAST_LAYER")},
    )
    checks.record(
        "the_frame_helper_is_stated_in_the_contract",
        FRAME_TOOLS_CONVENTION.strip() in contract,
        {"convention": FRAME_TOOLS_CONVENTION.strip()[:120]},
    )
    checks.record(
        "every_package_the_contract_names_actually_imports",
        named_imports.ok and "NAMED_IMPORTS_OK" in named_imports.stdout,
        {"named": list(AVAILABLE_IMPORTS), "stdout": named_imports.stdout.strip()},
    )
    checks.record(
        "a_package_the_contract_does_not_name_actually_fails",
        not unnamed_import.ok
        and "ModuleNotFoundError" in (unnamed_import.traceback or ""),
        {"probe": "sklearn"},
    )
    checks.record(
        "the_runtime_convention_is_stated_in_the_contract",
        RUNTIME_CONVENTION.strip() in contract,
        {"convention": RUNTIME_CONVENTION.strip()},
    )
    checks.record(
        "the_documented_observation_keys_match_a_real_observation_file",
        observed_schema_keys == sorted(OBSERVATION_SCHEMA_KEYS),
        {"observed": observed_schema_keys, "documented": sorted(OBSERVATION_SCHEMA_KEYS)},
    )
    checks.record(
        "the_schema_convention_is_stated_in_the_contract",
        SCHEMA_CONVENTION.strip() in contract,
        {"convention": SCHEMA_CONVENTION.strip()},
    )

    # Substrate finding: a model-written module is not importable by default.
    bare = outputs[2]["output"]
    checks.record(
        "the_bare_import_fails_as_a_model_repairable_python_error",
        classify_tool_result(bare).kind == "python_execution_failure"
        and "ModuleNotFoundError" in (bare.get("traceback") or ""),
        {"kind": classify_tool_result(bare).kind},
    )
    checks.record(
        "the_bare_import_convention_gap_is_documented_in_the_contract",
        IMPORT_CONVENTION.strip() in contract,
        {"contract_states_sys_path": IMPORT_CONVENTION.strip() in contract},
    )

    checks.finish(
        artifacts_root() / "p9-1-cumulative-loop" / "evidence.json",
        tool_outcomes=[
            {"action_step": item["action_step"], "model_round": item["model_round"], "kind": kind.kind}
            for item, kind in zip(outputs, kinds)
        ],
        notebook=notebook,
        findings=[
            {
                "id": "available-packages-were-never-stated",
                "observed": {"recorded_failures": 7, "example": "No module named 'sklearn'"},
                "why_it_matters": (
                    "The sandbox has numpy and Pillow only. The allowlist appears in the "
                    "host-only manifest, never to the model, so the model discovers it by "
                    "importing and failing."
                ),
                "disposition": (
                    "Stated, derived from MODEL_RUNTIME_DISTRIBUTIONS so it cannot drift, "
                    "and pinned by importing every named package in the live sandbox."
                ),
            },
            {
                "id": "observation-schema-keys-were-never-enumerated",
                "observed": {"recorded_failures": 4, "example": "KeyError: 'current'"},
                "why_it_matters": (
                    "The contract said 'flat objects with top-level public metadata' "
                    "without ever saying which keys that meant, so the model guessed."
                ),
                "disposition": (
                    "Enumerated, derived from observation_metadata so it cannot drift, and "
                    "pinned against a real observation file written during the run."
                ),
            },
            {
                "id": "observation-filename-width-was-never-stated",
                "observed": {
                    "model_visible_before": "observations/step-*.json (glob only)",
                    "actual": OBSERVATION_FILENAME_EXAMPLE,
                },
                "why_it_matters": (
                    "The glob hides the zero-padding width. The only concrete filenames "
                    "reach the model through recent_transitions, which is empty at step 0 "
                    "and carries just the last eight steps. A model constructing a path "
                    "guesses four digits and gets FileNotFoundError — observed in practice."
                ),
                "disposition": (
                    "Stated concretely in the workspace contract and system prompt, and "
                    "pinned against the filenames actually written so it cannot drift."
                ),
            },
            {
                "id": "bare-import-of-model-written-module-fails",
                "observed": {
                    "kind": classify_tool_result(bare).kind,
                    "traceback_tail": (bare.get("traceback") or "").strip().splitlines()[-1:],
                },
                "why_it_matters": (
                    "The worker runs python3 -I, so the workspace directory is absent from "
                    "sys.path. Reusing a model-written module requires an interface "
                    "convention the model would otherwise have to rediscover."
                ),
                "disposition": (
                    "Stated once in the shared workspace contract and system prompt. The "
                    "worker is inside participating_source_hashes() and is not modified."
                ),
            }
        ],
    )
    return 0


def _line(text: str, prefix: str) -> str:
    return next((line for line in text.splitlines() if line.startswith(prefix)), "")


def _turn_steps(payload: dict) -> set[int]:
    steps: set[int] = set()
    for message in payload["messages"]:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "text":
                continue
            try:
                steps.add(int(json.loads(part["text"])["step"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return steps


def arm_notice_present(payloads: list[dict]) -> bool:
    return any(
        isinstance(message.get("content"), str)
        and message["content"].startswith("Transcript omission.")
        for payload in payloads
        for message in payload["messages"]
    )


if __name__ == "__main__":
    raise SystemExit(main())
