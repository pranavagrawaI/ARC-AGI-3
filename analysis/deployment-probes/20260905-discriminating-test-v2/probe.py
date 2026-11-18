"""Two bounded model-selected experiments on a verified recorded UP transition."""

import hashlib
import json
import os
from pathlib import Path
import signal
import urllib.request

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
MODEL = "qwen/qwen3.8-27b"
PLAN = """You are investigating an unfamiliar grid environment. You have the full
frame before UP and the full frame after UP. Coordinates are zero-based row,column;
each row string encodes palette values 0-9,A-F. No other action has occurred.
Choose two plausible explanations consistent with these observations, and ONE next
action whose outcomes would distinguish those explanations. Ground each explanation
in specific observed pixels. Commit a concrete, checkable predicted outcome under
each explanation BEFORE seeing the result. Explain why these outcomes differ and
what each result would change about your next decision. Include the possibility
that neither prediction holds. Do not infer object identity from color alone.
You may choose UP, DOWN, LEFT, or RIGHT. You choose the observable relation;
there is no prescribed predicate or region. Return only a JSON object with keys:
observations, hypothesis_a, hypothesis_b, action, prediction_a, prediction_b,
measurement_procedure, decision_if_a, decision_if_b, decision_if_neither.
This is one experiment, not a plan for a series. Be precise and concise."""
REVIEW = """The selected action has now executed once. Here is its full resulting
frame. Evaluate your ORIGINAL committed predictions using these raw frames. Give
specific coordinates/values or counts supporting the comparison. Do not rewrite
the predictions. State whether the result distinguishes the two explanations,
which claims remain uncertain, and the next decision. Return only a JSON object
with keys: measurement, prediction_a_result, prediction_b_result,
distinguished, conclusion, uncertainty, next_decision."""


def save(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n")


def rows(grid):
    return ["".join("0123456789ABCDEF"[int(x)] for x in row) for row in grid]


def grid(frame):
    return [[int(x) for x in row] for row in frame.frame[-1]]


def main():
    from arc_agi import Arcade, OperationMode
    from arcengine import GameAction

    assert not (OUT / "requests.jsonl").exists(), "Never reuse a paid run"
    source = next((ROOT / "analysis/deployment-probes/20260906-experiment-loop-v2/workflow-b/workspaces").glob("*/ipc/transitions.jsonl"))
    transition = json.loads(source.read_text().splitlines()[0])
    before, after = transition["before"]["grid"], transition["after"]["grid"]
    save("INPUT.json", {"source": str(source), "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "before_up": before, "after_up": after})
    pin_paths = [Path(__file__), OUT / "EXPERIMENT-CARD.md"]
    save("SOURCE-PIN.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in pin_paths})
    for path in (ROOT / ".env", Path("/home/pranav/projects/arc-agi-3/.env")):
        if path.exists():
            for line in path.read_text().splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip("\"'"))
    credential = os.environ.get("OPENROUTER_API_KEY") or os.environ["ARC_LLM_API_KEY"]
    reserved = 0.10
    costs = []

    def call(arm, phase, messages):
        nonlocal reserved
        payload = dict(model=MODEL, messages=messages, max_tokens=4096, stream=False,
                       reasoning={"enabled": False}, provider={"max_price": {"prompt": 0.42, "completion": 3, "request": 0}, "allow_fallbacks": False})
        data = json.dumps(payload).encode()
        reservation = len(data) * 0.00000042 + 4096 * 0.000003 + 0.01
        assert len(data) <= 80000 and reserved + reservation <= 0.50
        reserved += reservation
        with (OUT / "requests.jsonl").open("a") as f:
            f.write(json.dumps(dict(arm=arm, phase=phase, reserved=reservation, request=payload)) + "\n")
        request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", data=data,
                                         headers={"Authorization": "Bearer " + credential, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.loads(response.read())
        with (OUT / "responses.jsonl").open("a") as f:
            f.write(json.dumps(dict(arm=arm, phase=phase, response=result)) + "\n")
        costs.append(result.get("usage", {}).get("cost"))
        content = result["choices"][0]["message"]["content"]
        save(f"{arm}-{phase}-content.json", content)
        return content

    actions = dict(UP=GameAction.ACTION1, DOWN=GameAction.ACTION2, LEFT=GameAction.ACTION3,
                   RIGHT=GameAction.ACTION4, ACTION5=GameAction.ACTION5)
    for arm in ("a", "b"):
        arcade = Arcade(operation_mode=OperationMode.OFFLINE, environments_dir=str(ROOT / "environment_files"), recordings_dir=str(OUT / arm / "recordings"))
        env = arcade.make("ls20")
        assert grid(env.reset()) == before, "Initial state differs from recorded input"
        assert grid(env.step(GameAction.ACTION1)) == after, "UP successor differs"
        messages = [{"role": "system", "content": PLAN}, {"role": "user", "content": json.dumps({"before_up": rows(before), "after_up": rows(after)})}]
        content = call(arm, "plan", messages)
        clean = content.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1].rsplit("```", 1)[0]
        plan = json.loads(clean)
        assert all(plan.get(k) for k in ("hypothesis_a", "hypothesis_b", "prediction_a", "prediction_b", "measurement_procedure"))
        selected = actions[plan["action"]]
        assert plan["action"] in transition["after"]["available_actions"]
        save(f"{arm}-COMMITTED.json", plan)
        result_grid = grid(env.step(selected))
        save(f"{arm}-TRANSITION.json", {"action": plan["action"], "before": after, "after": result_grid})
        messages += [{"role": "assistant", "content": content}, {"role": "user", "content": REVIEW + "\n" + json.dumps({"after_selected_action": rows(result_grid)})}]
        call(arm, "review", messages)
        save("COST.json", {"reported_costs": costs, "reserved_dollars": reserved})
        print(f"Completed {arm}: {plan['action']}; reported costs {costs}", flush=True)


if __name__ == "__main__":
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("600s total wall cap")))
    signal.alarm(600)
    try:
        main()
    except Exception as error:
        save("FAILURE.json", {"type": type(error).__name__, "message": str(error)})
        raise
