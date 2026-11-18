"""Index B1 cn04 L2 transcripts and action receipts without model calls."""
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
GID = "cn04-2fe56bfb"
HEADER = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d:\d\d:\d\d) \| tool-agent ---$", re.M)
MARKERS = re.compile(r"^\[(THINKING|TOOL CALL[^\]]*|TOOL RESULT[^\]]*|MODEL RESPONSE[^\]]*|USER PROMPT|SYSTEM PROMPT|ANALYZER STATUS)\]\s*$", re.M)


def clean(s):
    return re.sub(r"\s+", " ", s).strip()


def clip(s, n=650):
    s = clean(s)
    return s if len(s) <= n * 2 else s[:n] + f" [...{len(s)-n*2} chars...] " + s[-n:]


def main(sources=None, game_id=None, output_dir=None):
    gid = game_id or GID
    out = Path(output_dir) if output_dir is not None else OUT
    out.mkdir(parents=True, exist_ok=True)
    sources = sources or {label: ROOT / "solvers/m2_b1/experiment/pair-1" / label / "output"
                          for label in ["b1-r1", "b1-r2"]}
    records = {}
    for label, p in sources.items():
        p = Path(p)
        tp = p / "transcripts" / (gid + "_p0.txt")
        text = tp.read_text()
        hs = list(HEADER.finditer(text))
        r = next(r for r in json.loads((p/"benchmark.json").read_text())["game_runs"] if r["game_id"] == gid)
        h, apl = r["history"], r["actions_per_level"]
        start_idx, end_idx = apl[0], sum(apl[:2])
        t0 = h[start_idx-1]["wallclock_seconds"]
        t1 = h[end_idx-1]["wallclock_seconds"] if r["levels_completed"] >= 2 else r["final_wallclock_seconds"]
        events = [json.loads(line) for line in (p / "artifacts" / (gid + "_p0_events.jsonl")).open()]
        acts = [e for e in events if e["type"] == "action"]
        assert len(acts) == len(h) == sum(apl)
        entry = acts[start_idx-1]["board"]
        assert acts[start_idx-1]["level"] == 2
        action_rows = []
        previous = entry
        for i in range(start_idx, end_idx):
            e, a = acts[i], h[i]
            same = previous == e["board"]
            row = {"index": i+1, "local_action": i-start_idx+1,
                   "minutes_since_entry": (a["wallclock_seconds"]-t0)/60,
                   "analysis_step": e["analysis_step"], "action": a["action"],
                   "display": e["action_display"], "generated_tokens": a["generated_tokens"],
                   "board_changed": not same, "reported_board_changed": e["board_changed"],
                   "score": e["score"], "level_after": e["level"],
                   "red_pixels": sum(x==8 for row in e["board"] for x in row),
                   "board": e["board"], "board_ascii": e["board_ascii"]}
            assert row["board_changed"] == bool(e["board_changed"])
            action_rows.append(row)
            previous = e["board"]
        blocks, digest, texts = [], [], []
        for i, header in enumerate(hs):
            end = hs[i+1].start() if i+1<len(hs) else len(text)
            chunk = text[header.end():end]
            state = re.search(r"Current state: step (\d+), level (\d+)\.", chunk)
            if not state or int(state[2]) != 2:
                continue
            markers = list(MARKERS.finditer(chunk))
            fragments = []
            for j, m in enumerate(markers):
                mend = markers[j+1].start() if j+1 < len(markers) else len(chunk)
                fragments.append({"type": m[1], "text": chunk[m.end():mend].strip(),
                                  "source_line": text.count("\n", 0, header.end()+m.end())+1})
            step = int(header[1])
            b = {"analysis_step": step, "action_before": int(header[2])-1, "clock": header[3],
                 "source_line": text.count("\n", 0, header.start())+1,
                 "fragments": fragments}
            blocks.append(b)
            texts.append(header.group()+chunk)
            thoughts = [f for f in fragments if f["type"] == "THINKING"]
            tools = [f for f in fragments if f["type"].startswith("TOOL CALL:")]
            outputs = [f for f in fragments if f["type"].startswith("TOOL RESULT:")]
            action_subset = [a for a in action_rows if a["analysis_step"] == step]
            digest.append(f"\nSTEP {step} | before action {b['action_before']} | {header[3]} | source line {b['source_line']} | thinking chars {sum(len(f['text']) for f in thoughts)}")
            digest.append("ACTIONS: " + str([(a["local_action"], round(a["minutes_since_entry"],2), a["display"], a["board_changed"]) for a in action_subset]))
            if thoughts:
                digest.append("FIRST THOUGHT: " + clip(thoughts[0]["text"], 500))
                if len(thoughts)>1:
                    digest.append("LAST THOUGHT: " + clip(thoughts[-1]["text"], 500))
            for code, result in zip(tools, outputs):
                digest.append("CODE: " + clip(code["text"], 180))
                digest.append("RESULT: " + clip(result["text"], 200))
        (out/(label+"-digest.txt")).write_text("\n".join(digest)+"\n")
        (out/(label+"-level2.txt")).write_text("\n".join(texts))
        counter_path = p/"artifacts"/(gid+"_p0_tool_runtime_state_b1.json")
        records[label] = {"source": str(tp.relative_to(ROOT)), "source_sha256": hashlib.sha256(tp.read_bytes()).hexdigest(),
                          "entry_seconds": t0, "end_seconds": t1, "level_minutes": (t1-t0)/60,
                          "solved": r["levels_completed"]>=2, "entry_board": entry,
                          "entry_board_sha256": hashlib.sha256(json.dumps(entry).encode()).hexdigest(),
                          "actions": action_rows, "blocks": blocks,
                          "mechanism_counters": json.loads(counter_path.read_text()) if counter_path.exists() else {}}
        print(label, "minutes",round((t1-t0)/60,3), "actions",len(action_rows), "transcript turns",len(blocks),
              "digest_lines", len(digest), "board_sha", records[label]["entry_board_sha256"][:12])
    assert len({rec["entry_board_sha256"] for rec in records.values()}) == 1
    (out/"records.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
