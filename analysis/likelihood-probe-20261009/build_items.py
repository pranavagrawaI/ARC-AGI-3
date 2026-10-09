"""Build likelihood-probe items from the logged dfranzen runs (no GPU).

  solvers/tycho_nooa/.venv/bin/python -I build_items.py OUT.jsonl.xz

An item is one decision point: the last request of a turn, i.e. the exact context the model saw just before its code
executed actions. For each legal candidate action we record the oracle distance of its successor (if the level has a
BFS graph) and whether it matches a decisive action (the level-finishing click/press in a run that solved the level).
On Kaggle, probe.py scores log p(candidate code | context) by teacher forcing, with no sampling:
  R0: empty think block, then `action([...])` -- the model's "intuitive" choice
  R1: the model's own logged reasoning for that turn, then `action([...])` -- the post-reasoning choice
Images are stripped from the context (text-only); see RESULT.md.
"""
import collections, copy, glob, hashlib, json, lzma, re, sys
from pathlib import Path
import jinja2, numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "oracle-progress-20261009"))
sys.path.insert(0, str(HERE.parent / "metric-validation-20261008"))
from replay import game, GameAction, ActionInput, P          # noqa: E402
from oracle import H, components                              # noqa: E402
from validate import RUNS                                     # noqa: E402

TOK_DIR = Path.home() / ".cache/flashnext-tok"
NAMES = ["dfz-control", "dfz-r2", "dfz-s1"]
E2M = {"ACTION1": "UP", "ACTION2": "DOWN", "ACTION3": "LEFT", "ACTION4": "RIGHT", "ACTION5": "SPACE", "ACTION6": "MOUSE"}
MAX_CLICKS, DECISIVE_PER_RUN_LEVEL = 48, 10
MOVES = {"ACTION1", "ACTION2", "ACTION3", "ACTION4"}

env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])
env.filters["tojson"] = lambda x, **k: json.dumps(x, ensure_ascii=False, **k)
def _raise(m): raise jinja2.TemplateError(m)
env.globals["raise_exception"] = _raise
TEMPLATE = env.from_string((TOK_DIR / "chat_template.jinja").read_text())
TOK = Tokenizer.from_file(str(TOK_DIR / "tokenizer.json"))
ids = lambda s: TOK.encode(s, add_special_tokens=False).ids


def clean(messages):
    """Drop image parts; parse logged tool_calls (sometimes a repr string) into the shape the server templates."""
    out = []
    for m in messages:
        m = dict(m); c = m.get("content")
        if isinstance(c, list):
            m["content"] = "".join(p.get("text", "") for p in c if p.get("type") == "text")
        tc = m.get("tool_calls")
        if isinstance(tc, str): tc = eval(tc)  # logged as a Python repr of OpenAI objects (trusted local logs)
        if tc:
            calls = []
            for t in tc:
                f = t["function"]; a = f.get("arguments")
                try: a = json.loads(a) if isinstance(a, str) else a
                except json.JSONDecodeError: a = {"code": a}
                calls.append({"type": "function", "function": {"name": f["name"], "arguments": a if isinstance(a, dict) else {"code": str(a)}}})
            m["tool_calls"] = calls
        out.append(m)
    return out


def render(messages, tools):
    return TEMPLATE.render(messages=messages, tools=tools, add_generation_prompt=True, preserve_thinking=True)


def code_for(a):
    i, d = a
    if i == "ACTION6": return f"action([{{'action':'MOUSE','row':{d['y']},'col':{d['x']}}}])"
    return f"action(['{E2M[i]}'])"


def same(a, b):
    if a[0] != b[0]: return False
    return a[0] != "ACTION6" or (abs(a[1]["x"] - b[1]["x"]) <= 2 and abs(a[1]["y"] - b[1]["y"]) <= 2)


def load_graphs():
    G = {}
    for f in glob.glob(str(HERE.parent / "oracle-progress-20261009/graphs/*.json")):
        j = json.loads(Path(f).read_text()); s = j["summary"]
        if j["dist"]: G[(s["game"][:4], s["level"])] = (s["exhausted"], j["dist"])
    return G


def level_runs():
    """Per run: game -> (history, apl, done). Plus decisive actions per (game4, level) from any solving dfz run."""
    R, dec = {}, collections.defaultdict(list)
    for n in NAMES:
        b = json.loads((RUNS[n] / "benchmark.json").read_text()); R[n] = {}
        for r in b["game_runs"]:
            h, apl, done = P(r["history"]), P(r["actions_per_level"]), int(r["levels_completed"] or 0)
            R[n][r["game_id"]] = (h, apl, done); k = 0
            for L, m in enumerate(apl):
                if L < done and m:
                    a = h[k + m - 1]["action"]
                    if a["id"] not in MOVES: dec[(r["game_id"][:4], L)].append((a["id"], a.get("data") or {}))
                k += m
    return R, dec


def turn_requests(run, gid):
    """step -> (messages, tools, next assistant reasoning or None) for the LAST request before actions at that step."""
    f = RUNS[run] / f"{gid}_p0_requests.jsonl"
    req = [r for r in map(json.loads, open(f)) if r["event"] == "request"]
    last = {}
    for i, r in enumerate(req): last[r["action"]] = i
    out = {}
    for step, i in last.items():
        r = req[i]; reason = None
        if i + 1 < len(req):
            nxt = req[i + 1]["messages"]; n = len(r["messages"])
            if nxt[:n] == r["messages"] and len(nxt) > n and nxt[n]["role"] == "assistant":
                reason = nxt[n].get("reasoning_content")
        out[step] = (r["messages"], r["tools"], reason)
    return out


def main(out_path):
    G, (R, dec) = load_graphs(), level_runs()
    items = []
    for run in NAMES:
        for gid, (h, apl, done) in sorted(R[run].items()):
            g4 = gid[:4]
            want = {L for L in range(len(apl)) if (g4, L) in G or (g4, L) in dec}
            if not want: continue
            if not (RUNS[run] / f"{gid}_p0_requests.jsonl").exists(): continue   # only dfz-control kept request logs
            turns = turn_requests(run, gid)
            starts = {}; k = 0
            for L, m in enumerate(apl):
                starts[L] = k; k += m
            # level of the state after s actions (s = 0..len(h))
            lvl = [sum(1 for L in range(done) if starts[L] + apl[L] <= s) for s in range(len(h) + 1)]
            e = game(gid); fr = e.perform_action(ActionInput(id=GameAction.RESET))
            # decisive-item sampling: at most DECISIVE_PER_RUN_LEVEL evenly spaced turn starts per level
            chosen_dec = {}
            for L in want:
                if (g4, L) not in dec or (g4, L) in G: continue
                steps = [s + 1 for s in range(len(h) + 1) if lvl[s] == L and s + 1 in turns]
                if len(steps) > DECISIVE_PER_RUN_LEVEL:
                    steps = [steps[round(i * (len(steps) - 1) / (DECISIVE_PER_RUN_LEVEL - 1))] for i in range(DECISIVE_PER_RUN_LEVEL)]
                chosen_dec[L] = set(steps)
            for s_idx in range(len(h) + 1):          # state after s_idx actions; its turn has step s_idx + 1
                step = s_idx + 1
                L = fr.levels_completed
                if L in want and step in turns and ((g4, L) in G or step in chosen_dec.get(L, ())):
                    item = make_item(run, gid, L, s_idx, step, e, fr, h, apl, done, starts, turns[step], G.get((g4, L)), dec.get((g4, L), []))
                    if item: items.append(item)
                if s_idx < len(h):
                    a = h[s_idx]["action"]; fr = e.perform_action(ActionInput(id=GameAction[a["id"]], data=a.get("data") or {}))
            print(run, gid[:4], sum(1 for it in items if it["run"] == run and it["game"] == g4), flush=True)
    with lzma.open(out_path, "wt", preset=9 | lzma.PRESET_EXTREME) as f:
        for it in items: f.write(json.dumps(it) + "\n")
    print(f"{len(items)} items -> {out_path}")


def make_item(run, gid, L, s_idx, step, e, fr, h, apl, done, starts, turn, graph, decs):
    messages, tools, reason = turn
    avail = [f"ACTION{a}" for a in fr.available_actions if a in range(1, 7)]
    cands = [(a, {}) for a in avail if a != "ACTION6"]
    taken = (h[s_idx]["action"]["id"], h[s_idx]["action"].get("data") or {}) if s_idx < len(h) else None
    if "ACTION6" in avail:
        clicks = [("ACTION6", {"x": x, "y": y}) for x, y in components(fr)]
        must = [c for c in [taken, *decs] if c and c[0] == "ACTION6"]
        keep = [c for c in clicks if any(same(c, m) for m in must)]
        rest = [c for c in clicks if c not in keep]
        rng = np.random.default_rng(int(hashlib.md5(f"{gid}{s_idx}".encode()).hexdigest()[:8], 16))
        rest = [rest[i] for i in sorted(rng.permutation(len(rest))[:max(0, MAX_CLICKS - len(keep) - len(must))])]
        cands += [m for m in must if not any(m == c for c in keep)] + keep + rest
    seen, uniq = set(), []
    for c in cands:
        key = (c[0], tuple(sorted(c[1].items())))
        if key not in seen: seen.add(key); uniq.append(c)
    cands = uniq
    exact, dist = graph if graph else (None, None)
    d0 = dist.get(H(fr)) if dist else None
    rows = []
    for c in cands:
        nd = None
        if dist is not None:
            x = copy.deepcopy(e); f2 = x.perform_action(ActionInput(id=GameAction[c[0]], data=c[1]))
            nd = 0 if f2.levels_completed > L else (dist.get(H(f2)) if f2.state.name != "GAME_OVER" else -1)
            if nd is not None and H(f2) == H(fr) and f2.levels_completed == L: nd = d0  # no-op
        rows.append(dict(a=c[0], d=c[1], code=code_for(c), nd=nd, decisive=any(same(c, m) for m in decs),
                         taken=bool(taken and same(c, taken))))
    if len(rows) < 2: return None
    msgs = clean(messages)
    t_lvl_start = h[starts[L] - 1]["wallclock_seconds"] if starts[L] else 0.0
    t_now = h[s_idx - 1]["wallclock_seconds"] if s_idx else 0.0
    solved = L < done
    t_solve = h[starts[L] + apl[L] - 1]["wallclock_seconds"] if solved and apl[L] else None
    return dict(id=f"{run}|{gid[:4]}|{L}|{s_idx}", run=run, game=gid[:4], level=L, step=s_idx, exact=exact, d0=d0,
                min_in_level=(t_now - t_lvl_start) / 60, solved=solved, min_to_solve=(t_solve - t_now) / 60 if solved else None,
                messages=msgs, tools=tools, reasoning=reason, cands=rows)


if __name__ == "__main__":
    main(sys.argv[1])
