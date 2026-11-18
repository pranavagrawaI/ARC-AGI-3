#!/usr/bin/env python3
"""S1: executed reaches to a named module, against the REAL nemo trace schema.

The first draft of this looked for `type == "repl_execute"`, a `code` field, and
`status == "complete"`. None exist. The events carry `event`, `success`, `error`,
`code_length` and a `functions_used` list that holds only bare resolved names — on q4,
four distinct names across 399 cells, with `grid_array` absent. A gate built on any of
that reads 0 whether or not the module was used, so it could not have detected the thing
it claims to measure.

Cell source lives in `agent_logs/**/messages/step_*_round_*_assistant.md` as
`[tool_call] name=execute_python args={"code": ...}`. Reaches are read from there;
whether the reach *executed* is read from events.jsonl.

    python3 reach_scorer.py <run_dir> [--module symmetry] [--probe grid_array]

`--probe` is the validation lever: name a token known to be present and confirm the
extractor finds it. An extractor that cannot see `grid_array` in q4's cells would not
have seen `symmetry.canon` either.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

CALL = re.compile(r'\[tool_call\]\s+name=execute_python\s+args=(\{.*?\})\s*$', re.S | re.M)
MSG = re.compile(r"step_(\d+)_round_(\d+)_assistant\.md$")


def cells(run: Path):
    """Every executed CodeAct cell as (step, round, order_in_step, source)."""
    out = []
    for d in run.glob("agent_logs/*/*/messages"):
        for f in sorted(d.iterdir()):
            m = MSG.search(f.name)
            if not m:
                continue
            step, rnd = int(m.group(1)), int(m.group(2))
            for blob in CALL.findall(f.read_text(errors="replace")):
                try:
                    src = json.loads(blob).get("code", "")
                except json.JSONDecodeError:
                    continue
                out.append([step, rnd, 0, src])
    out.sort(key=lambda r: (r[0], r[1]))
    n = defaultdict(int)
    for row in out:
        row[2] = n[row[0]]
        n[row[0]] += 1
    return out


def executions(run: Path):
    """Per step: (n_events, failures, {code_length: [ok]}).

    Resolution is deliberately step-level, because the order-join is unsound: the
    framework prepends prefill cells that never appear in an assistant message, and on q4
    the per-step surplus was 2 for 46 steps, 1 for 10 and 3 for 6 — not a constant offset.
    `code_length` joins exact-1 for only 241/279 cells.

    So: a reach in a step with zero failed cells executed, no join needed. A reach in a
    step that contains any failure must be pinned by an exact-unique `code_length`, and an
    unresolvable one is scored ERR. That can undercount and can never inflate, which is
    the property the original uptake scorer was built to protect.

    On q3c and q4 the cell failure rate was 0/177 and 0/399, so this resolves everything.
    """
    by_step = {}
    for f in run.glob("agent_logs/*/*/events.jsonl"):
        rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
        for r in rows:
            if r.get("event") != "repl_execute":
                continue
            step = int(r.get("step", -1))
            err = r.get("error")
            ok = str(r.get("success")).lower() == "true" and err in (None, "None", "")
            d = by_step.setdefault(step, {"n": 0, "failures": 0, "by_len": defaultdict(list)})
            d["n"] += 1
            d["failures"] += 0 if ok else 1
            d["by_len"][int(r.get("code_length", -1))].append(ok)
    return by_step


def resolve(step_info: dict, src: str):
    """(executed, how) for one reach."""
    if step_info is None:
        return False, "no-events-for-step"
    if step_info["failures"] == 0:
        return True, "step-clean"
    cand = step_info["by_len"].get(len(src), [])
    if len(cand) == 1:
        return cand[0], "code_length"
    return False, "unresolvable-in-failing-step"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--module", default="symmetry")
    ap.add_argument("--probe", default="")
    a = ap.parse_args()
    run = Path(a.run)

    use = re.compile(rf"\b{re.escape(a.module)}\s*\.\s*(\w+)")
    cs, ex = cells(run), executions(run)
    total_ex = sum(v["n"] for v in ex.values())
    print(f"cells parsed from messages: {len(cs)}   repl_execute events: {total_ex}")
    fails = sum(v["failures"] for v in ex.values())
    print(f"  cells not in messages (framework prefill): {total_ex - len(cs)}   failed cells: {fails}")

    reaches, executed, names = 0, 0, set()
    for step, rnd, k, src in cs:
        hits = use.findall(src)
        if not hits:
            continue
        reaches += 1
        ok, how = resolve(ex.get(step), src)
        if ok:
            executed += 1
            names.update(hits)
        print(f"  [{'ok ' if ok else 'ERR'}] step {step} round {rnd} ({how}): {sorted(set(hits))}")

    print(f"\nS1  module={a.module!r}  reached={reaches}  executed={executed}  functions={sorted(names) or '-'}")
    if a.probe:
        p = re.compile(rf"\b{re.escape(a.probe)}\b")
        hit = sum(1 for _, _, _, s in cs if p.search(s))
        print(f"VALIDATION probe={a.probe!r} found in {hit}/{len(cs)} cells "
              f"— {'extractor can see cell source' if hit else 'EXTRACTOR IS BLIND, do not launch'}")
        return 0 if hit else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
