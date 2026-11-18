"""No network/GPU: prove date and restart decisions against durable artifacts."""
import datetime as dt
import json
import tempfile
from pathlib import Path
from startup import next_action


def at(value):
    return dt.datetime.fromisoformat(value)


with tempfile.TemporaryDirectory() as tmp:
    folder = Path(tmp)
    assert next_action(folder, at('2026-10-30T23:59:59+00:00')) == 'WAIT_FOR_DATE'
    assert next_action(folder, at('2026-10-31T00:00:00+00:00')) == 'WAIT_UPLINK_THEN_PUSH_ONCE'
    assert next_action(folder, at('2026-10-31T11:59:59+00:00')) == 'WAIT_UPLINK_THEN_PUSH_ONCE'
    assert next_action(folder, at('2026-10-31T12:00:00+00:00')) == 'LAUNCH_EXPIRED'
    assert next_action(folder, at('2026-11-01T00:00:00+00:00')) == 'LAUNCH_EXPIRED'
    marker = folder / 'launch-attempt.json'
    for state in ['attempting', 'acceptance unresolved; never retry', 'error']:
        marker.write_text(json.dumps({'state': state}))
        assert next_action(folder, at('2026-10-31T06:00:00+00:00')) == 'HOLD_AMBIGUOUS_NEVER_PUSH'
    marker.write_text(json.dumps({'state': 'accepted'}))
    assert next_action(folder, at('2026-10-31T06:00:00+00:00')) == 'OBSERVE'
    (folder / 'terminal.json').write_text('{}')
    assert next_action(folder, at('2026-10-31T06:00:00+00:00')) == 'COLLECT'
    assert next_action(folder, at('2026-11-01T12:00:00+00:00')) == 'OBSERVATION_CUTOFF_NEVER_PUSH'
    (folder / 'QUALIFICATION.json').write_text('{"verdict":"FAIL"}')
    assert next_action(folder, at('2026-10-31T06:00:00+00:00')) == 'QUALIFIED_OR_FAILED_DONE'
    (folder / 'QUALIFICATION.json').write_text('{"verdict":"PASS"}')
    assert next_action(folder, at('2026-10-31T06:00:00+00:00')) == 'QUALIFIED_OR_FAILED_DONE'
print('PASS: date boundaries; no next-day spill; attempted/ambiguous never push; accepted restart observes; terminal collects; PASS/FAIL cannot relaunch')
