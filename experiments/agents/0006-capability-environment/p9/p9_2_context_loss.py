"""P9.2 — what survives when the transcript does not.

Question: with the transcript forced to evict, does the model still reach the
authoritative episode record and its own files, is the omission stated rather
than silent, and does the host avoid substituting a summary for what it dropped?

The archive that used to back this prototype is gone, so what is tested here is
availability, not authority: after eviction the model can still read every past
observation and every file it wrote. The prototype also records the cost of that
decision — those files are model-writable, and nothing restores them if the model
destroys them. That is the accepted risk, measured rather than asserted.

Run: ``uv run python3 experiments/agents/0006-capability-environment/p9/p9_2_context_loss.py``
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "p9"

from p9.agent import NOTES_CHAR_LIMIT, NOTES_FILE, OBSERVATIONS_GLOB
from p9.arms import build_arm, control_spec
from p9.conversation import OMISSION_NOTICE_TEMPLATE, omission_notice_text
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


TRANSCRIPT_ONLY_MARKER = "TRANSCRIPT_ONLY_MARKER_9F2A"

MARK = f"""\
from pathlib import Path
Path('notes.md').write_text('# notes\\n- authored at step 1\\n')
Path('{NOTES_FILE}').write_text('CONCLUSION_FROM_STEP_1: action 1 moves up\\n')
print('{TRANSCRIPT_ONLY_MARKER}')
"""

RECOVER_EVICTED = """\
import json
from pathlib import Path
steps = sorted(int(p.stem.split('-')[1]) for p in Path('observations').glob('step-*.json'))
print('OBSERVATION_STEPS', steps)
first = json.load(open('observations/step-00000.json'))
print('EVICTED_STEP0_PIXEL_ROWS', len(first['pixels'][0]))
print('NOTES', Path('notes.md').read_text().strip().replace(chr(10), ' | '))
"""

MEASURE_THE_ACCEPTED_RISK = """\
from pathlib import Path
target = Path('observations/step-00000.json')
before = len(target.read_bytes())
target.write_text('{}')
print('OBSERVATIONS_ARE_MODEL_WRITABLE', before, '->', len(target.read_bytes()))
print('NOTHING_RESTORES_IT', target.read_text().strip())
"""

SCRIPT = (
    tool_turn(MARK),
    action_turn(1, reason="step 1"),
    action_turn(1, reason="step 2"),
    tool_turn(RECOVER_EVICTED),
    tool_turn(MEASURE_THE_ACCEPTED_RISK),
    action_turn(1, reason="step 3"),
    action_turn(1, reason="step 4"),
)


def _events(path: Path, event_type: str) -> list[dict]:
    return [
        record
        for record in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        if record["type"] == event_type
    ]


def _notices(payload: dict) -> list[str]:
    return [
        message["content"]
        for message in payload["messages"]
        if isinstance(message.get("content"), str)
        and message["content"].startswith("Transcript omission.")
    ]


def _payload_text(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True)


def main() -> int:
    checks = Checks(
        prototype="P9.2-context-loss",
        question=(
            "After forced transcript eviction, does the authoritative episode "
            "record survive model tampering, do model-authored files survive, is "
            "the omission explicit, and is no host summary substituted?"
        ),
    )
    repo = Path(__file__).resolve().parents[4]
    frames = load_recorded_frames(default_recording(repo, minimum_frames=5), count=5, crop=8)

    with TemporaryDirectory(prefix="p9-2-") as scratch:
        client = ScriptedChatClient(SCRIPT)
        spec = control_spec(max_actions=4, max_model_calls=14, max_context_turns=1)
        with build_arm(spec, run_dir=Path(scratch) / "arm", inner_client=client) as arm:
            result = run_episode(
                RecordedEnvironment(frames=frames), arm.agent, max_actions=spec.max_actions
            )
            log = arm.run_dir / "process.jsonl"
            evictions = _events(log, "context_eviction")
            tool_events = _events(log, "python_tool")
            outputs = [json.loads(event["output"]) for event in tool_events]
            payloads = client.request_payloads()
            retained_steps = list(arm.agent.conversation.retained_steps)
            evicted_steps = list(arm.agent.conversation.evicted_steps)

    checks.record(
        "the_script_ran_to_completion",
        client.remaining == 0 and result.actions == 4,
        {"remaining_script_entries": client.remaining, "actions": result.actions},
    )
    checks.record(
        "the_transcript_actually_evicted_turns",
        bool(evictions) and evicted_steps == [1, 2, 3],
        {"evicted_steps": evicted_steps, "retained_steps": retained_steps},
    )

    recovery, risk = outputs[1], outputs[2]
    checks.record(
        "every_observation_including_evicted_steps_is_still_readable",
        "OBSERVATION_STEPS [0, 1, 2]" in recovery["stdout"],
        {"steps": _line(recovery["stdout"], "OBSERVATION_STEPS")},
    )
    checks.record(
        "the_observation_of_an_evicted_step_is_intact",
        "EVICTED_STEP0_PIXEL_ROWS 8" in recovery["stdout"],
        {"line": _line(recovery["stdout"], "EVICTED_STEP0_PIXEL_ROWS")},
    )
    checks.record(
        "model_authored_files_survived_the_eviction",
        "NOTES # notes | - authored at step 1" in recovery["stdout"],
        {"line": _line(recovery["stdout"], "NOTES")},
    )
    # Not a defect: the accepted cost of removing the read-only archive. Recorded so
    # the contract's deviation from #9 is measured rather than asserted.
    checks.record(
        "the_accepted_risk_is_real_and_recorded",
        classify_tool_result(risk).kind == "none"
        and "OBSERVATIONS_ARE_MODEL_WRITABLE" in risk["stdout"]
        and "NOTHING_RESTORES_IT {}" in risk["stdout"],
        {"stdout": risk["stdout"].strip()},
    )

    notices = {index: _notices(payload) for index, payload in enumerate(payloads)}
    with_notice = {index: items for index, items in notices.items() if items}
    # The notes block exists so a conclusion reached before an eviction is still in
    # front of the model after it. Assert it in the payloads, not in the workspace.
    notes_echoed = [
        index
        for index, payload in enumerate(payloads)
        if any(
            isinstance(part, dict)
            and part.get("type") == "text"
            and "CONCLUSION_FROM_STEP_1" in part.get("text", "")
            for message in payload["messages"]
            if isinstance(message.get("content"), list)
            for part in message["content"]
        )
    ]
    checks.record(
        "a_conclusion_written_before_the_eviction_is_still_in_context_after_it",
        bool(with_notice) and set(with_notice) <= set(notes_echoed),
        {"requests_echoing_notes": notes_echoed, "requests_with_omission_notice": sorted(with_notice)},
    )
    checks.record(
        "the_omission_was_stated_to_the_model_after_eviction",
        bool(with_notice) and all(len(items) == 1 for items in with_notice.values()),
        {"requests_with_notice": sorted(with_notice), "requests": len(payloads)},
    )
    expected_first = omission_notice_text(steps=(1,), observations_glob=OBSERVATIONS_GLOB)
    expected_second = omission_notice_text(steps=(1, 2), observations_glob=OBSERVATIONS_GLOB)
    observed = sorted({items[0] for items in with_notice.values()})
    checks.record(
        "every_notice_is_the_template_with_only_steps_and_paths_substituted",
        set(observed) <= {expected_first, expected_second},
        {"observed": observed},
    )
    checks.record(
        "the_notice_grew_with_the_evicted_step_set_rather_than_being_static",
        expected_first in observed and expected_second in observed,
        {"distinct_notices": len(observed)},
    )
    checks.record(
        "the_evicted_turn_content_is_absent_from_later_requests",
        not any(
            TRANSCRIPT_ONLY_MARKER in _payload_text(payload)
            for index, payload in enumerate(payloads)
            if index in with_notice
        ),
        {"marker": TRANSCRIPT_ONLY_MARKER},
    )
    final_roles = [message["role"] for message in payloads[-1]["messages"]]
    checks.record(
        "no_host_authored_message_appears_beyond_the_pointer_only_notice",
        _host_authored_messages(payloads[-1]) == [expected_second],
        {
            "host_authored": _host_authored_messages(payloads[-1]),
            "final_roles": final_roles,
        },
    )

    checks.finish(
        artifacts_root() / "p9-2-context-loss" / "evidence.json",
        omission_notice_template=OMISSION_NOTICE_TEMPLATE,
        observed_notices=observed,
        eviction_events=evictions,
        recovery_stdout=recovery["stdout"],
        accepted_risk_stdout=risk["stdout"],
    )
    return 0


def _line(text: str, prefix: str) -> str:
    return next((line for line in text.splitlines() if line.startswith(prefix)), "")


def _host_authored_messages(payload: dict) -> list[str]:
    """String-content user messages the host wrote, excluding scripted model turns.

    Observation messages are structured content lists; assistant and tool
    messages come from the model and the sandbox. Anything left that is a plain
    user string was authored by the harness.
    """

    return [
        message["content"]
        for message in payload["messages"]
        if message["role"] == "user" and isinstance(message.get("content"), str)
    ]


if __name__ == "__main__":
    raise SystemExit(main())
