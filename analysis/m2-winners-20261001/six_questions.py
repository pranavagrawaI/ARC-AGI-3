"""Evidence for the six waste questions, over a TAAF run with request logs + transcripts + events."""
import json, re, sys, collections, hashlib
from pathlib import Path
out = Path(sys.argv[1])
SEC = re.compile(r"^\[(SYSTEM PROMPT|USER PROMPT|MODEL RESPONSE META|THINKING|TOOL CALL: python|TOOL RESULT: python|ANALYZER STATUS)\]\s*$")

def turns(text):
    cur, buf, items = None, [], []
    for line in text.splitlines():
        m = SEC.match(line)
        if m or line.startswith("--- analysis_step="):
            if cur: items.append((cur, "\n".join(buf)))
            cur, buf = (m.group(1) if m else "TURN"), []
        else: buf.append(line)
    if cur: items.append((cur, "\n".join(buf)))
    return items

R = collections.Counter(); ex = collections.defaultdict(list)
code_lines_total = code_lines_seen = 0
dup_calls = calls = 0; think_total = 0; think_grid = 0; think_arith = 0
for t in sorted(out.glob("transcripts/*.txt")):
    g = t.name[:4]; seen_lines = set(); seen_calls = set()
    items = turns(t.read_text(errors="replace"))
    for kind, body in items:
        if kind == "TOOL CALL: python":
            code = "\n".join(re.findall(r"<parameter=code>\n(.*?)</parameter>", body, re.S)) or body
            calls += 1
            h = hashlib.md5(code.strip().encode()).hexdigest()
            if h in seen_calls: dup_calls += 1
            seen_calls.add(h)
            lines = [l.strip() for l in code.splitlines() if len(l.strip()) > 25]
            code_lines_total += len(lines); code_lines_seen += sum(l in seen_lines for l in lines)
            seen_lines.update(lines)
            # mechanical-knowledge probes
            if re.search(r"\.ascii\.split|segmentation\[.nodes.\]|for r in range\(64\)|for r in range\(len\(", code): R["calls_reparse_board"] += 1
            if re.search(r"frame_diff|last_transition|previous_frame|before_frame", code) and "action(" not in code: R["calls_inspect_last_change"] += 1
            if "action(" in code: R["calls_with_action"] += 1
        elif kind == "THINKING":
            think_total += len(body)
            # grid transcription in thinking: lines that look like board rows
            rows = [l for l in body.splitlines() if re.fullmatch(r"\s*(i?\d+:?)?\s*[A-Za-z.#|]( ?[A-Za-z.#|]){7,}\s*", l)]
            think_grid += sum(len(l) for l in rows)
            ar = re.findall(r"\(\s*\d+\s*,\s*\d+\s*\)|\d+\s*[-+*/]\s*\d+\s*=\s*\d+", body)
            think_arith += len(ar)
            for kw, key in [(r"wait|hmm|actually", "self_correction"), (r"let me (re-?check|verify|recount|re-?read|index carefully|count)", "recheck_phrases")]:
                R[key] += len(re.findall(kw, body, re.I))
        elif kind == "TOOL RESULT: python":
            if "Traceback" in body or body.lstrip().startswith("error"):
                m = re.findall(r"^(\w+Error)[: ](.*)$", body, re.M)
                key = m[-1][0] if m else "other"; R["err_" + key] += 1
                if m: ex[key].append(m[-1][1][:90])

print(f"tool calls {calls}; exact duplicate calls {dup_calls} ({dup_calls/calls:.1%}); "
      f"code lines (>25ch) re-typed from earlier calls in same game: {code_lines_seen}/{code_lines_total} ({code_lines_seen/code_lines_total:.0%})")
print(f"calls re-parsing the board from ascii/segmentation: {R['calls_reparse_board']} ({R['calls_reparse_board']/calls:.0%}); "
      f"inspect-last-change calls without action: {R['calls_inspect_last_change']}; calls with action: {R['calls_with_action']}")
print(f"thinking chars {think_total:,}; grid transcribed into thinking {think_grid:,} ({think_grid/think_total:.1%}); "
      f"coordinate tuples/arithmetic in thinking {think_arith:,}; self-corrections (wait/hmm/actually) {R['self_correction']:,}; recheck phrases {R['recheck_phrases']}")
print("errors:", {k[4:]: v for k, v in R.most_common() if k.startswith("err_")})
for k in ("NameError", "AttributeError", "KeyError", "TypeError", "ImportError"):
    print(f"  {k}:", collections.Counter(re.sub(r"\d+", "N", e) for e in ex[k]).most_common(4))

# ---- actions: useless ones we already know
noop = 0; acts = 0; rep_noop = 0; same_fail = 0; deaths = 0; repeat_death = 0
for evf in out.glob("artifacts/*_events.jsonl"):
    ev = [json.loads(l) for l in evf.open()]
    prev_board = None; noop_keys = set(); death_keys = set()
    for e in ev:
        if e["type"] == "initial": prev_board = hashlib.md5(json.dumps(e.get("board")).encode()).hexdigest(); continue
        if e["type"] != "action": continue
        acts += 1
        bh = hashlib.md5(json.dumps(e.get("board")).encode()).hexdigest()
        key = (prev_board, e.get("action_display") or e.get("action_name"))
        if e.get("board_changed") is False:
            noop += 1
            if key in noop_keys: rep_noop += 1
            noop_keys.add(key)
        if e.get("game_over"):
            deaths += 1
            if key in death_keys: repeat_death += 1
            death_keys.add(key)
        prev_board = bh
print(f"actions {acts}; board-unchanged actions {noop} ({noop/acts:.0%}); of those, the SAME action from the SAME board already known to be a no-op: {rep_noop}; "
      f"deaths {deaths}, repeated identical fatal (board,action): {repeat_death}")

# ---- information lost: history drains
drains = 0; dropped = 0; resets_after_level = 0
for f in out.glob("*_p0_requests.jsonl"):
    prev = None
    for l in f.open():
        r = json.loads(l)
        if r["event"] != "response": continue
        p = r["usage"]["prompt_tokens"]
        if prev and p < 0.7 * prev: drains += 1; dropped += prev - p
        prev = p
print(f"history drains {drains} (~{drains/25:.1f}/game), prompt tokens dropped at drains ~{dropped:,}")
