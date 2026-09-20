"""Replay ft09_p0 step 14 under control vs treatment against qwen/qwen3.6-27b on OpenRouter.

Treatment adds exactly one line to the step-14 user prompt (frozen wording below). Everything else —
system prompt, 23 turns of reconstructed history (with grid images), runtime state, sampling
(T 0.6 / top_p 0.95 / top_k 20 / thinking on), tools — is identical between arms. Trials are
interleaved C,T,C,T,... `action()` runs against a small ft09-L1 simulator (toggle b/R on the framed
grid; completes on the pattern p1 found). Each trial is one Duck turn (ends when action() executes).

usage: replay.py N [--out DIR]
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TREATMENT_LINE = ("Decode only from values printed by code this turn; validate a mapping on every "
                  "reference instance on the board before applying it.")
ENV = {"MULTIMODAL_CONTEXT": "current_grid", "MULTIMODAL_UPSCALE": "4", "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
       "LOCAL_ANALYZER_MAX_OUTPUT": "0", "LOCAL_ANALYZER_TOOL_STEPS": "0", "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
       "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024", "LOCAL_ANALYZER_YIELD_SECONDS": "0", "LOCAL_ANALYZER_TEMPERATURE": "0.6",
       "LOCAL_ANALYZER_TOP_P": "0.95", "LOCAL_ANALYZER_TOP_K": "20", "LOCAL_ANALYZER_ENABLE_THINKING": "true",
       "LOCAL_ANALYZER_TIMEOUT": "300"}
for k, v in ENV.items(): os.environ[k] = v
sys.path.insert(0, str(ROOT / "ARC3-Inference"))
from inference.agent.runtime_state import Frame, HistoryEntry, load_runtime_state, write_runtime_state  # noqa: E402
from inference.agent import tool_agent as ta  # noqa: E402

R, B = 8, 9
ROWS = (36, 44, 52); COLS = (36, 44, 52)
TARGET = [[R, B, B], [R, None, R], [R, B, B]]

class Sim:
    """ft09 level 1: clicks inside a framed-grid cell toggle it b<->R; nothing else responds."""
    def __init__(self, state_path: Path):
        self.state_path = state_path
        frame, history = load_runtime_state(state_path)
        self.grid = [list(r) for r in frame.grid]; self.history = list(history); self.n = frame.step; self.level = frame.level
        self.log = []
    def cell(self, row, col):
        for i, r0 in enumerate(ROWS):
            for j, c0 in enumerate(COLS):
                if r0 <= row < r0 + 6 and c0 <= col < c0 + 6 and not (i == 1 and j == 1): return i, j
        return None
    def pattern(self): return [[self.grid[r0 + 2][c0 + 2] if not (i == 1 and j == 1) else None for j, c0 in enumerate(COLS)] for i, r0 in enumerate(ROWS)]
    def __call__(self, arguments):
        actions = arguments.get("actions") or [arguments]
        results = []
        for k, a in enumerate(actions, 1):
            self.n += 1; changed = False
            name = str(a.get("action", "")).upper()
            if name == "MOUSE":
                row, col = int(a.get("row", a.get("y", -1))), int(a.get("col", a.get("x", -1)))
                c = self.cell(row, col)
                if c is not None:
                    i, j = c; r0, c0 = ROWS[i], COLS[j]; cur = self.grid[r0][c0]; new = B if cur == R else R
                    for r in range(r0, r0 + 6):
                        for cc in range(c0, c0 + 6): self.grid[r][cc] = new
                    changed = True
                disp = f"MOUSE(row={row}, col={col})"
            else:
                disp = name
            completed = self.pattern() == TARGET
            if completed: self.level = 2
            frame = Frame(grid=tuple(tuple(r) for r in self.grid), step=self.n, level=self.level)
            self.history.append(HistoryEntry(action=disp, frame=frame))
            write_runtime_state(self.state_path, current_frame=frame, history=self.history)
            p = {"executed": True, "action_num": self.n, "level": self.level, "score": int(completed), "reward": float(completed), "state": "NOT_FINISHED",
                 "valid_actions": ["ACTION6"], "board_changed": changed, "done": False, "level_completed": completed, "game_over": False, "run_complete": False,
                 "action_name": "ACTION6" if name == "MOUSE" else name, "action_data": {}, "action_display": disp, "batch_index": k, "batch_size": len(actions)}
            results.append(p); self.log.append(disp)
            if completed: break
        f = dict(results[-1]); f.update(reward=sum(x["reward"] for x in results), last_reward=results[-1]["reward"], batched=len(actions) > 1,
                                    requested_count=len(actions), executed_count=len(results), requested_actions=[], executed_actions=[x["action_display"] for x in results],
                                    board_changed=any(x["board_changed"] for x in results), stopped_early=len(results) < len(actions))
        if f["level_completed"]: f["stop_reason"] = "level_completed"
        return f

def trial(arm: str, idx: int, out: Path, model: str, base_url: str) -> dict:
    tdir = out / f"{idx:02d}-{arm}"; tdir.mkdir(parents=True, exist_ok=True)
    state_path = tdir / "tool_runtime_state.json"
    state_path.write_bytes((HERE / "state/tool_runtime_state.json").read_bytes())
    agent = ta.ToolAgent(model=model, base_url=base_url, provider="openrouter")
    agent._ensure_session(state_path)
    agent._history_messages = copy.deepcopy(json.load(open(HERE / "state/history_messages.json")))
    agent._last_step_summary = json.load(open(HERE / "state/last_step_summary.json"))
    agent._last_action_result = json.load(open(HERE / "state/last_action_result.json"))
    agent._summarized_knowledge.update(json.load(open(HERE / "state/world_model.json")))
    if arm == "treatment":
        orig = agent._build_user_prompt
        def patched(action_num, **kw):
            p = orig(action_num, **kw)
            return p + "\n" + TREATMENT_LINE if action_num == 24 else p
        agent._build_user_prompt = patched
    sim = Sim(state_path)
    t0 = time.time()
    res = agent.analyze(state_path, 24, valid_actions=["ACTION6"], step_env=sim, transcript_path=tdir / "transcript.txt", analysis_step=14)
    rec = {"arm": arm, "idx": idx, "seconds": round(time.time() - t0, 1), "step_executed": bool(res and res.step_executed),
           "retryable_failure": bool(res and res.retryable_failure), "actions": sim.log, "final_pattern": sim.pattern(), "level_completed": sim.level == 2,
           "tokens": agent.total_tokens}
    json.dump(rec, open(tdir / "result.json", "w"), indent=1)
    return rec

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("n", type=int); ap.add_argument("--out", default=str(HERE / "trials"))
    ap.add_argument("--model", default="qwen/qwen3.6-27b"); ap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    a = ap.parse_args(); out = Path(a.out)
    for i in range(a.n):
        for arm in ("control", "treatment"):
            rec = trial(arm, i, out, a.model, a.base_url)
            print(json.dumps({k: rec[k] for k in ("idx", "arm", "seconds", "step_executed", "level_completed", "actions")}), flush=True)

if __name__ == "__main__":
    main()
