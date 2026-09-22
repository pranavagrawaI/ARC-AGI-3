"""qwen38-qualify readback: the Step 1 table on the 3.8 checkpoint, plus the kill-rule checks.

usage: readback.py <kaggle-output-dir> [--baseline <step1-kaggle-output-dir>]

Reproduces the columns of duck-harness/runs/step1-duck-alone/RESULT.md (levels, actions, turns,
s/turn, s->L1, actions->L1, board_changed %) so the checkpoints can be compared row by row, and
answers the three kill-rule questions from CARD.md: did the server serve, did tool calls parse,
did sb26/su15 reach L1. s->L1 is turn-rounded (it anchors on the level-completing action's
transcript step), so it reads a few seconds high against RESULT.md's per-action figure. Adds the decode decomposition (output tokens/call, tok/s per stream)
since that is the quantity the checkpoint swap was expected to move.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

STEP_RE = re.compile(r"^--- analysis_step=(\d+) \| action=(\d+) \| (\d\d):(\d\d):(\d\d)", re.M)
META_RE = re.compile(
    r"\[MODEL RESPONSE META\]\nfinish_reason: (\w+)\ntool_call_count: (\d+)\n"
    r"content_chars: (\d+)\nreasoning_chars: (\d+)\ntool_call_markup_in_text: (\w+)\n"
    r"tool_calls_recovered_from_markup: (\w+)"
)
ARGS_RE = re.compile(r'"arguments": "((?:[^"\\]|\\.)*)"')
VLLM_RE = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs"
)


def episodes(out: Path) -> list[dict]:
    rows = []
    for ev_path in sorted((out / "artifacts").glob("*_events.jsonl")):
        stem = ev_path.name.replace("_events.jsonl", "")
        ev = [json.loads(l) for l in ev_path.read_text().splitlines() if l.strip()]
        acts = [e for e in ev if e.get("type") == "action"]
        levels = max((int(e.get("level") or 1) for e in ev), default=1) - 1
        changed = [e for e in acts if e.get("board_changed")]
        tr = out / "transcripts" / f"{stem}.txt"
        text = tr.read_text() if tr.exists() else ""
        steps = STEP_RE.findall(text)
        secs = [int(h) * 3600 + int(m) * 60 + int(s) for _, _, h, m, s in steps]
        span = secs[-1] - secs[0] if len(secs) > 1 else 0
        metas = META_RE.findall(text)
        code = [len(a) for a in ARGS_RE.findall(text)]
        # first action that completed a level, and the transcript second it happened at
        s_to_l1 = a_to_l1 = None
        for i, e in enumerate(acts):
            if int(e.get("level") or 1) > 1 or e.get("level_completed"):
                a_to_l1 = i + 1
                step = e.get("analysis_step")
                idx = [int(s[0]) for s in steps]
                if step in idx:
                    s_to_l1 = secs[idx.index(step)] - secs[0]
                break
        rows.append(
            dict(
                episode=stem.split("-")[0] + stem[stem.rindex("_"):],
                levels=levels,
                actions=len(acts),
                turns=len(steps),
                span=span,
                s_per_turn=span / len(steps) if steps else 0.0,
                s_to_l1=s_to_l1,
                a_to_l1=a_to_l1,
                changed_pct=100 * len(changed) / len(acts) if acts else 0.0,
                calls=len(metas),
                bad_finish=sum(1 for f, *_ in metas if f != "tool_calls"),
                markup_recovered=sum(1 for m in metas if m[5] == "yes"),
                out_chars=sum(int(c) + int(r) for _, _, c, r, *_ in metas) + sum(code),
                reasoning_chars=sum(int(r) for _, _, _, r, *_ in metas),
                code_chars=sum(code),
            )
        )
    return rows


def vllm(out: Path) -> dict:
    log = out / "vllm-openai-server.log"
    if not log.exists():
        return {}
    rows = [tuple(float(x) for x in m) for m in VLLM_RE.findall(log.read_text())]
    active = [r for r in rows if r[2] > 0]
    full = [r for r in active if r[2] >= 10]
    return dict(
        samples=len(rows),
        gen_tok_s=st.mean(r[1] for r in full) if full else 0.0,
        per_stream=(st.mean(r[1] for r in full) / 10) if full else 0.0,
        max_running=max((r[2] for r in rows), default=0),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("output", type=Path)
    p.add_argument("--baseline", type=Path, default=None)
    args = p.parse_args()

    rows = episodes(args.output)
    if not rows:
        # The failure path: vLLM never served, or every episode died before a transcript.
        # This is when the kill rule fires, so say why rather than crashing on empty stats.
        print(f"NO EPISODES under {args.output}")
        log = args.output / "vllm-openai-server.log"
        if not log.exists():
            print("  vllm-openai-server.log absent — the server was never started (check the kernel log).")
        else:
            text = log.read_text()
            print(f"  vllm-openai-server.log present, {len(text.splitlines())} lines")
            bad = [l for l in text.splitlines() if re.search(r"Error|Traceback|ValueError|unsupported|not supported|CUDA out of memory", l)]
            for l in bad[:10]:
                print("   ", l.strip()[:200])
            if not bad:
                print("    no error lines matched; inspect the tail manually")
        raise SystemExit(1)

    hdr = f"{'episode':<12}{'levels':>7}{'actions':>8}{'turns':>7}{'s/turn':>8}{'s→L1':>7}{'act→L1':>8}{'chg%':>6}{'calls':>7}{'tok/call':>9}"
    print(hdr)
    for r in rows:
        print(
            f"{r['episode']:<12}{r['levels']:>7}{r['actions']:>8}{r['turns']:>7}"
            f"{r['s_per_turn']:>8.1f}{(r['s_to_l1'] if r['s_to_l1'] is not None else '–'):>7}"
            f"{(r['a_to_l1'] if r['a_to_l1'] is not None else '–'):>8}{r['changed_pct']:>6.0f}"
            f"{r['calls']:>7}{(r['out_chars'] / r['calls'] / 3.5 if r['calls'] else 0):>9.0f}"
        )

    tot_calls = max(1, sum(r["calls"] for r in rows))
    tot_out = max(1, sum(r["out_chars"] for r in rows))
    reasoning = sum(r["reasoning_chars"] for r in rows)
    code = sum(r["code_chars"] for r in rows)
    l1 = {r["episode"]: r["levels"] for r in rows}
    gate = [k for k in l1 if k.startswith(("sb26", "su15"))]

    print("\nKill rule (CARD.md):")
    print(f"  server served            : {len(rows)} episodes produced transcripts")
    bad = sum(r["bad_finish"] for r in rows)
    rec = sum(r["markup_recovered"] for r in rows)
    print(f"  tool calls parsed        : {tot_calls - bad}/{tot_calls} finish_reason=tool_calls; {rec} recovered from markup")
    print(f"  L1 on sb26/su15          : {sum(1 for k in gate if l1[k] >= 1)}/{len(gate)}  (3.6 baseline: 4/4)")

    print("\nDecode profile:")
    print(f"  median s/turn            : {st.median([r['s_per_turn'] for r in rows]):.1f}  (3.6: 46)")
    print(f"  calls / turn             : {tot_calls / max(1, sum(r['turns'] for r in rows)):.2f}  (3.6: 1.67)")
    # chars/3.5 is an estimate; the vLLM log's own token count gives 573 for the same 3.6 run.
    print(f"  output tokens / call ~   : {tot_out / tot_calls / 3.5:.0f}  (3.6, same estimator: 514)")
    print(f"  reasoning / code share   : {reasoning / tot_out:.0%} / {code / tot_out:.0%}  (3.6: 59% / 37%)")
    v = vllm(args.output)
    if v:
        print(f"  vLLM gen tok/s @10-way   : {v['gen_tok_s']:.0f} aggregate, {v['per_stream']:.1f} per stream  (3.6: 187 / 18.7)")

    if args.baseline:
        base = episodes(args.baseline)
        print(f"\nvs baseline {args.baseline}: levels {sum(r['levels'] for r in rows)} vs {sum(r['levels'] for r in base)}; "
              f"median s/turn {st.median([r['s_per_turn'] for r in rows]):.1f} vs {st.median([r['s_per_turn'] for r in base]):.1f}")


if __name__ == "__main__":
    main()
