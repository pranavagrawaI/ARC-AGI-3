"""Reconstruct Duck's state at ft09_p0 step 14 (24 actions executed) from the saved transcript + events.

Outputs into ./state/: tool_runtime_state.json, history_messages.json, saved_prompt.txt,
last_step_summary.json, last_action_result.json, world_model.json, and a fidelity report comparing
the prompt ToolAgent builds from this state with the saved step-14 USER PROMPT.
"""
from __future__ import annotations
import json, os, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = ROOT / "runs/periodic-v1/control/kaggle-output"
STEP = 14
os.environ.setdefault("MULTIMODAL_CONTEXT", "current_grid"); os.environ.setdefault("MULTIMODAL_UPSCALE", "4")
os.environ.setdefault("LOCAL_ANALYZER_CONTEXT_WINDOW", "32768")
sys.path.insert(0, str(ROOT / "ARC3-Inference"))
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state  # noqa: E402
from inference.agent.tool_agent import ToolAgent, _extract_scientist_note  # noqa: E402

events = [json.loads(l) for l in open(OUT / "artifacts/ft09-0d8bbf25_p0_events.jsonl")]
initial = next(e for e in events if e.get("type") == "initial")
acts = {e["action_num"]: e for e in events if e.get("type") == "action"}

def frame_after(n: int) -> Frame:
    b = initial["board"] if n == 0 else acts[n]["board"]
    lvl = initial["level"] if n == 0 else acts[n]["level"]
    return Frame(grid=tuple(tuple(int(x) for x in row) for row in b), step=n, level=int(lvl))

txt = open(OUT / "transcripts/ft09-0d8bbf25_p0.txt", encoding="utf-8").read()
att = re.split(r"(?m)^--- analysis_step=(\d+) \| action=(\d+) \| ([0-9:]+) \| tool-agent ---$", txt)
attempts = [(int(att[i]), int(att[i + 1]), att[i + 3]) for i in range(1, len(att) - 3, 4)]

def sections(body):
    parts = re.split(r"(?m)^\[([A-Z :a-z]+)\]$", body)
    return [(parts[j], parts[j + 1].strip("\n")) for j in range(1, len(parts) - 1, 2)]

agent = ToolAgent(model="stub", base_url="http://x", provider="openrouter")
messages = []
saved_prompt = None
store = {}
for step, disp_action, body in attempts:
    secs = sections(body)
    if step >= STEP:
        if step == STEP and saved_prompt is None:
            saved_prompt = next(c for l, c in secs if l == "USER PROMPT")
        break
    count = disp_action - 1
    frame = frame_after(count)
    first_user = True
    pending_calls = []
    for label, content in secs:
        if label == "USER PROMPT":
            if first_user:
                messages.append(agent._build_user_message(content, frame)); first_user = False
            else:
                messages.append({"role": "user", "content": content})
        elif label == "MODEL RESPONSE META":
            m = re.search(r"raw_tool_calls:\n(\[.*\])\s*$", content, flags=re.S)
            pending_calls = json.loads(m.group(1)) if m else []
            messages.append({"role": "assistant"})
        elif label == "THINKING":
            messages[-1]["reasoning"] = content
        elif label == "ASSISTANT":
            messages[-1]["content"] = content
            note = _extract_scientist_note(content)
            for k, v in note.items():
                if v and k != "critic_request": store[k] = v
        elif label.startswith("TOOL CALL"):
            if pending_calls and "tool_calls" not in messages[-1]:
                messages[-1]["tool_calls"] = pending_calls
        elif label.startswith("TOOL RESULT"):
            call = (messages[-1].get("tool_calls") or [{}])[0] if messages[-1].get("role") == "assistant" else {}
            # tool results are logged as their display form; re-wrap as the sandbox's JSON payload
            messages.append({"role": "tool", "tool_call_id": call.get("id", ""), "content": json.dumps({"tool": "python", "returncode": 0, "stdout": content}, indent=2)})
    # level_transition/game_over clears (none before step 14 in this episode)
for m in messages:
    if m.get("role") == "assistant" and "content" not in m: m["content"] = None

# runtime state after 24 actions
history = [HistoryEntry(action="", frame=frame_after(0))] + [HistoryEntry(action=acts[n]["action_display"], frame=frame_after(n)) for n in range(1, 25)]
state_dir = HERE / "state"; state_dir.mkdir(exist_ok=True)
write_runtime_state(state_dir / "tool_runtime_state.json", current_frame=frame_after(24), history=history)

a24 = acts[24]
payload = {"executed": True, "action_num": 24, "level": 1, "score": 0, "reward": 0.0, "state": "NOT_FINISHED", "valid_actions": ["ACTION6"],
           "board_changed": False, "done": False, "level_completed": False, "game_over": False, "run_complete": False,
           "action_name": "ACTION6", "action_data": {"x": 32, "y": 61}, "action_display": a24["action_display"], "batch_index": 1, "batch_size": 1,
           "batched": False, "requested_count": 1, "executed_count": 1, "requested_actions": [a24["action_display"]], "executed_actions": [a24["action_display"]], "stopped_early": False}
compact = agent._compact_action_result(payload)
summary = agent._summarize_step_sequence([payload])
json.dump(messages, open(state_dir / "history_messages.json", "w"))
json.dump(summary, open(state_dir / "last_step_summary.json", "w"), indent=1)
json.dump(compact, open(state_dir / "last_action_result.json", "w"), indent=1)
json.dump(store, open(state_dir / "world_model.json", "w"), indent=1)
(state_dir / "saved_prompt.txt").write_text(saved_prompt, encoding="utf-8")

# fidelity: rebuild the prompt from this state and diff against the saved one
agent._summarized_knowledge.update(store); agent._last_step_summary = summary; agent._last_action_result = compact
from inference.agent.runtime_state import load_runtime_state
cf, he = load_runtime_state(state_dir / "tool_runtime_state.json")
built = agent._build_user_prompt(24, valid_actions=["ACTION6"], current_frame=cf, history_entries=he, previous_step_summary=summary)
import difflib
if built == saved_prompt:
    print("FIDELITY: rebuilt prompt is byte-identical to the saved step-14 USER PROMPT")
else:
    d = list(difflib.unified_diff(saved_prompt.splitlines(), built.splitlines(), "saved", "rebuilt", lineterm="", n=0))
    print("FIDELITY: DIFF", len(d), "lines"); print("\n".join(d[:40]))
print(f"history messages: {len(messages)} ({sum(m['role']=='assistant' for m in messages)} assistant turns); world model keys: {sorted(store)}")
