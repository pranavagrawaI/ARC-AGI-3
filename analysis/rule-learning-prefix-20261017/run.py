"""Prefix-only CPU prediction diagnostic; never executes the donor notebook/install cell."""
import argparse
import hashlib
import importlib.util
import json
import signal
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(name, initial=None):
    spec = importlib.util.spec_from_file_location("prefix_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    module.__dict__.update(initial or {})
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


LX = load("lexicon")
R = load("rules", {"LX": LX})
V = load("rules_view")
R.LX = LX


class StepLimit(Exception):
    pass


def alarm(*_):
    raise StepLimit("per-transition wall cap")


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def encode_prefix(boards):
    # Prefix frames ONLY set the scale. The withheld successor never changes the fitted representation.
    lock = LX.scale_lock(boards)
    lex = LX.Lexicon(naming="stable")
    encs = []
    for board in boards:
        encs.append(LX.encode(board, lex, lock=lock, prev=encs[-1] if encs else None))
    return lock, lex, encs


def model_before(boards, actions):
    # The report's window=60 retains at most 61 frames. Scale uses the complete available prefix.
    lock, lex, encs = encode_prefix(boards)
    learner = R.Learner()
    start = max(0, len(actions) - 60)
    for i in range(start, len(actions)):
        action, click = V.parse_action(actions[i])
        if not action or action == "RESET":
            continue
        a, b = R.State(encs[i]), R.State(encs[i + 1])
        learner.observe(a, action, click, R.observe(a, b))
    return lock, lex, encs[-1], learner.rules()


def evaluate(prefix_boards, prefix_actions, action, successor):
    lock, lex, enc, rules = model_before(prefix_boards, prefix_actions)
    before = R.State(enc)
    name, click = V.parse_action(action)
    predictions, clicked_cell = R.predict(rules, before, name, click)
    visible_predictions, _ = R.predict(rules[:10], before, name, click)
    # Seal predictions and selected rules BEFORE encoding/scoring the withheld frame.
    frozen = {"rules": [r.line() for r in rules],
              "predictions": [(r.text(), i, eff) for r, i, eff in predictions],
              "prefix_sha256": digest([prefix_boards, prefix_actions])}
    prediction_sha256 = digest(frozen)
    after_code = LX.encode(successor, lex, lock=lock, prev=enc)
    after = R.State(after_code)
    events = R.observe(before, after)
    score = R.score(predictions, clicked_cell, events)
    visible_score = R.score(visible_predictions, clicked_cell, events)
    # Independent raw pixel check prevents "no events" from hiding changed visible boards.
    raw_changed = prefix_boards[-1] != successor
    changed_pixels = sum(a != b for x, y in zip(prefix_boards[-1], successor) for a, b in zip(x, y))
    return dict(status="OK", lock=lock, prefix_actions=len(prefix_actions),
                frozen_prediction_sha256=prediction_sha256, **frozen,
                score=dict(zip(["predictions_right", "predictions", "events_explained", "events", "exact"], score)),
                visible_top10_score=dict(zip(["predictions_right", "predictions", "events_explained", "events", "exact"], visible_score)),
                raw_changed=raw_changed, changed_pixels=changed_pixels,
                encoded_events=events, event_descriptions=[V.describe(e) for e in events],
                objects_before=enc["placed"], objects_after=after_code["placed"],
                code_before=LX.to_text(enc), code_after=LX.to_text(after_code))


def preflight():
    # Known external expectation: translating a colored square changes its position.
    a = [[5] * 16 for _ in range(16)]
    b = [[5] * 16 for _ in range(16)]
    for y in [6, 7]:
        for x in [6, 7]:
            a[y][x] = 8
            b[y][x + 2] = 8
    q = evaluate([a], [], "RIGHT", b)
    assert q["raw_changed"] and q["score"]["events"] > 0
    assert not q["score"]["exact"] and q["score"]["predictions"] == 0
    unchanged = evaluate([a], [], "RIGHT", a)
    assert unchanged["frozen_prediction_sha256"] == q["frozen_prediction_sha256"]
    assert unchanged["score"]["events"] == 0 and unchanged["score"]["exact"]
    # Mouse row/column adapter checked against an independent expected (x,y).
    assert V.parse_action("MOUSE(row=57, col=27)") == ("MOUSE", (27, 57))
    assert digest([[a], []]) == q["prefix_sha256"]
    print(json.dumps(dict(status="PREFLIGHT_PASS", event_descriptions=q["event_descriptions"])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        preflight()
        return
    card = json.loads((HERE / "CARD.json").read_text())
    for path, expected in card["sha256"].items():
        assert hashlib.sha256((HERE / path).read_bytes()).hexdigest() == expected, path
    output = HERE / "transitions.jsonl"
    assert not output.exists(), "Frozen run already attempted; use a distinct run ID, never overwrite it."
    signal.signal(signal.SIGALRM, alarm)
    started = time.monotonic()
    tracks = json.loads((HERE / "inputs.json").read_text())
    with output.open("x") as out:
        for track in tracks:
            boards, actions = [track["entry_board"]], []
            for row in track["actions"]:
                if time.monotonic() - started >= card["cap"]["total_wall_seconds"]:
                    raise SystemExit("TOTAL_RESOURCE_LIMIT: preserve incomplete run")
                record = dict(track=track["id"], game=track["game"], run=track["run"],
                              local_action=row["local_action"], global_action=row["index"], action=row["display"])
                if row["level_after"] != 2:
                    record["status"] = "LEVEL_BOUNDARY_NOT_SCORED"
                elif row["display"] == "RESET":
                    record["status"] = "RESET_NOT_SCORED"
                else:
                    t0 = time.monotonic()
                    signal.setitimer(signal.ITIMER_REAL, card["cap"]["per_transition_wall_seconds"])
                    try:
                        record.update(evaluate(boards, actions, row["display"], row["board"]))
                        assert record["raw_changed"] == row["board_changed"]
                    except StepLimit as exc:
                        record.update(status="RESOURCE_LIMIT", error=str(exc))
                    except Exception as exc:
                        record.update(status="ERROR", error=f"{type(exc).__name__}: {exc}")
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                    record["elapsed_s"] = time.monotonic() - t0
                out.write(json.dumps(record) + "\n")
                out.flush()
                boards.append(row["board"])
                actions.append(row["display"])
            print(json.dumps(dict(track=track["id"], completed_actions=len(actions), elapsed_s=time.monotonic() - started)), flush=True)
    (HERE / "execution.json").write_text(json.dumps(dict(status="COMPLETE", wall_s=time.monotonic() - started,
                                                          transitions_file_sha256=hashlib.sha256(output.read_bytes()).hexdigest()), indent=2) + "\n")


if __name__ == "__main__":
    main()
