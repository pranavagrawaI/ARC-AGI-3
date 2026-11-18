#!/usr/bin/env python3
"""How much of level 1 is spent re-deriving what level 0 already established?

Baseline for a game-local-skill card. Level-1 completion cannot be its endpoint:
across every local run, level 1 has been entered 3 times and completed 0 times.
Rediscovery can be, because it is measured per action and per authored claim and
has a real nonzero rate.

No single clean definition survives the level change -- the state abstraction that
makes within-level re-entry decidable (`playfield_key`) cannot match across levels,
because the layout changes. So this reports three complementary channels and says
what each over- and under-counts. A card picks one and states why.

    python3 rediscovery.py                # all level-1 traces
    python3 rediscovery.py --run <dir>    # one

Outcome labelling is copied from `score_q4.py` rather than imported, following the
house rule that a frozen scorer is not reused.
"""

from __future__ import annotations

import argparse
import ast
import itertools
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import replay

HUD_MIN_FRAC = 0.9
SETTLED_MIN_OCCURRENCES = 4
SETTLED_MIN_SHARE = 0.8
CLAIM_OVERLAP = 0.6

PROBES = Path(__file__).resolve().parent.parent / "local-probes"
TRACES = {
    "q2d": "20260903T-skill-following-q2/results/nemo_solver/20260903_160733_ls20_mdfiles_q2d",
    "q3a": "20260904T-host-control-loop-q3/results/nemo_solver/20260904_011341_ls20_mdfiles_q3a",
    "q3c": "20260904T-host-control-loop-q3/results/nemo_solver/20260904_140403_ls20_mdfiles_q3c",
}


# ------------------------------------------------------- labelling (score_q4 copy)


def discover_hud_rows(frames: list[np.ndarray]) -> list[int]:
    n = len(frames) - 1
    if n <= 0:
        return []
    hits = np.zeros(frames[0].shape[0], dtype=int)
    for a, b in itertools.pairwise(frames):
        hits[(a != b).any(axis=1)] += 1
    return [int(r) for r, h in enumerate(hits) if h / n >= HUD_MIN_FRAC]


def classify(a: np.ndarray, b: np.ndarray, hud: list[int]) -> str:
    m = a != b
    if not m.any():
        return "blocked"
    rows = set(np.where(m.any(axis=1))[0].tolist())
    return "hud_only" if rows <= set(hud) else "playfield"


def outcome_class(label: str, state: str, level_up: bool) -> str:
    if state in ("GAME_OVER", "WIN"):
        return "terminal"
    if level_up:
        return "level_up"
    return "no_impact" if label in ("hud_only", "blocked") else "move"


# ----------------------------------------------------------------- per-action rows


@dataclass(frozen=True)
class Action:
    step: int
    level: int
    base: str
    outcome: str
    batch_len: int


def _per_action(run_dir: Path) -> list[Action]:
    events = replay._load_jsonl(replay._events_path(run_dir))
    env = sorted(
        (e for e in events if e.get("event") == "env_step"), key=lambda e: e["step"]
    )
    grids = [np.array(e["grid_data"], dtype=np.int16) for e in env]
    if len(grids) < 2:
        return []
    hud = discover_hud_rows(grids)
    labels = [classify(grids[i], grids[i + 1], hud) for i in range(len(grids) - 1)]
    level_at = {int(e["step"]): int(e.get("levels_completed", 0)) for e in env}

    states = replay._load_jsonl(run_dir / "ipc" / "states.jsonl")
    batches = replay._load_jsonl(run_dir / "ipc" / "actions.jsonl")
    by_submit = {b.get("turn"): b for b in batches}

    rows: list[Action] = []
    for st in states:
        results = st.get("action_results") or []
        if not results:
            continue
        end, k = st.get("step"), len(results)
        batch = by_submit.get((st.get("turn") or 0) - 1)
        blen = len(batch.get("actions", [])) if batch else 1
        for j, r in enumerate(results):
            idx = end - k + j
            if idx < 0 or idx >= len(labels):
                continue
            action = (r.get("action") or "").strip()
            rows.append(
                Action(
                    step=idx + 1,
                    level=level_at.get(idx, 0),
                    base=action.split()[0] if action else "",
                    outcome=outcome_class(
                        labels[idx], r.get("state", ""), bool(r.get("level_completed"))
                    ),
                    batch_len=blen,
                )
            )
    return rows


# ----------------------------------------------------------------- authored content

_TOOL_ARGS = re.compile(r"^\[tool_call\]\s+name=\S+\s+args=(\{.*\})\s*$", re.MULTILINE)
_MSG_ASSISTANT = re.compile(r"step_(\d+)_round_(\d+)_assistant\.md$")


def _string_bindings(tree: ast.AST) -> dict[str, str]:
    """Names bound to a string constant in this cell.

    Helper source is normally built as `src = '''...'''` and then passed as
    `write_helper("game.py", src)`, so a walk that only accepts a literal second
    argument silently reports zero helper writes. That is a missed measurement,
    not an absence.
    """
    out: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                out[target.id] = node.value.value
    return out


def _authored(run_dir: Path, fns: set[str]) -> tuple[list[tuple[int, str, str]], int]:
    """(step, target, payload) per authoring call, plus a count of unresolved ones."""
    messages = replay._messages_dir(run_dir)
    if messages is None:
        return [], 0
    out: list[tuple[int, str, str]] = []
    unresolved = 0
    for path in sorted(messages.glob("*_assistant.md")):
        m = _MSG_ASSISTANT.search(path.name)
        if m is None:
            continue
        step = int(m.group(1))
        for raw in _TOOL_ARGS.findall(path.read_text()):
            try:
                code = json.loads(raw).get("code", "")
                tree = ast.parse(code)
            except (json.JSONDecodeError, SyntaxError):
                continue
            bound = _string_bindings(tree)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
                    continue
                if node.func.attr not in fns or len(node.args) < 2:
                    continue
                a, b = node.args[0], node.args[1]
                if not (isinstance(a, ast.Constant) and isinstance(a.value, str)):
                    continue
                if isinstance(b, ast.Constant) and isinstance(b.value, str):
                    out.append((step, a.value, b.value))
                elif isinstance(b, ast.Name) and b.id in bound:
                    out.append((step, a.value, bound[b.id]))
                else:
                    unresolved += 1
    return out, unresolved


_STOP = frozenset(
    ["a", "an", "the", "of", "at", "in", "on", "to", "and", "or", "for", "with", "is", "are", "was", "were", "be", "not", "no", "this", "that", "these", "those", "it", "its", "from", "by", "as", "into", "if", "then", "when"]
)


def _content(text: str) -> set[str]:
    return {
        t for t in re.findall(r"[a-z0-9_]+", text.lower()) if t not in _STOP and len(t) > 2
    }


def _overlap(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a) if a else 0.0


def _claims(payload: str) -> list[str]:
    """One claim per hypothesis entry or bullet; both note shapes the models use."""
    out = [h.claim for h in replay._parse_hypotheses(payload)]
    if out:
        return out
    return [
        line.strip(" -*\t")
        for line in payload.splitlines()
        if line.strip().startswith(("-", "*")) and len(line.strip()) > 8
    ]


# ---------------------------------------------------------------------- indicators


def measure(run_dir: Path) -> dict:
    actions = _per_action(run_dir)
    l0 = [a for a in actions if a.level == 0]
    l1 = [a for a in actions if a.level == 1]
    if not l1:
        return {"level_1_actions": 0}

    # R3 -- settled action laws. In level 0, a base is "settled" if it occurred
    # often enough and its modal outcome dominated. A length-1 turn is the
    # exploratory unit by the host's own rule: a longer batch has to be earned by
    # a correct prediction, so it is plan execution rather than probing.
    settled: dict[str, str] = {}
    for base in {a.base for a in l0 if a.base}:
        obs = [a.outcome for a in l0 if a.base == base]
        if len(obs) < SETTLED_MIN_OCCURRENCES:
            continue
        modal = max(set(obs), key=obs.count)
        if obs.count(modal) / len(obs) >= SETTLED_MIN_SHARE:
            settled[base] = modal

    probes = [a for a in l1 if a.batch_len == 1]
    r3 = [a for a in probes if settled.get(a.base) == a.outcome]
    inert = {b for b, o in settled.items() if o == "no_impact"}
    r3_strict = [a for a in l1 if a.base in inert and a.outcome == "no_impact"]

    # R1 / R2 -- content authored during level 1 that restates level-0 content.
    step_level = {a.step: a.level for a in actions}

    def level_of(step: int) -> int:
        prior = [s for s in sorted(step_level) if s <= step]
        return step_level[prior[-1]] if prior else 0

    notes, _ = _authored(run_dir, {"append_notes", "write_notes"})
    c0 = [c for s, _, p in notes if level_of(s) == 0 for c in _claims(p)]
    c1 = [(s, c) for s, _, p in notes if level_of(s) == 1 for c in _claims(p)]
    t0 = [_content(c) for c in c0]
    # Report across thresholds: at a loose cutoff, coordinate-heavy sentences match
    # on shared vocabulary rather than shared content. The card picks a threshold
    # and says why, the way the stall operating point was chosen from the sweep.
    dup_at = {}
    for thr in (0.6, 0.75, 0.9):
        dup_at[thr] = [
            (s, c) for s, c in c1 if any(_overlap(_content(c), t) >= thr for t in t0)
        ]
    dup = dup_at[CLAIM_OVERLAP]

    helpers, unresolved = _authored(run_dir, {"write_helper"})
    h0 = {s: p for s, _, p in helpers if level_of(s) == 0}
    h1 = {s: p for s, _, p in helpers if level_of(s) == 1}
    f0: set[str] = set()
    for src in h0.values():
        f0 |= set(re.findall(r"^def\s+(\w+)", src, re.MULTILINE))
    redefined: set[str] = set()
    for src in h1.values():
        redefined |= set(re.findall(r"^def\s+(\w+)", src, re.MULTILINE)) & f0

    return {
        "level_0_actions": len(l0),
        "level_1_actions": len(l1),
        "level_1_probes": len(probes),
        "settled_laws": settled,
        "R3_settled_probe_rate": len(r3) / len(l1),
        "R3_count": len(r3),
        "R3_strict_inert_retest_rate": len(r3_strict) / len(l1),
        "R3_strict_count": len(r3_strict),
        "R1_claims_level0": len(c0),
        "R1_claims_level1": len(c1),
        "R1_duplicate_rate": (len(dup) / len(c1)) if c1 else None,
        "R1_by_threshold": {
            t: (len(v) / len(c1)) if c1 else None for t, v in dup_at.items()
        },
        "R1_duplicates": [c for _, c in dup][:4],
        "R2_functions_level0": sorted(f0),
        "R2_redefined_in_level1": sorted(redefined),
        "R2_redefinition_rate": (len(redefined) / len(f0)) if f0 else None,
        "R2_unresolved_helper_writes": unresolved,
    }


def render(name: str, m: dict) -> str:
    if not m.get("level_1_actions"):
        return f"{name}: no level-1 actions"
    lines = [
        (
            f"{name}  L0={m['level_0_actions']} actions, L1={m['level_1_actions']} actions "
            f"({m['level_1_probes']} of them length-1 probes)"
        ),
        f"  settled L0 action laws: {m['settled_laws'] or 'none'}",
        (
            f"  R3  settled-law re-probing      {m['R3_settled_probe_rate']:.3f}  "
            f"({m['R3_count']}/{m['level_1_actions']})   [upper bound]"
        ),
        (
            f"  R3s inert-base re-test          {m['R3_strict_inert_retest_rate']:.3f}  "
            f"({m['R3_strict_count']}/{m['level_1_actions']})"
        ),
    ]
    by = m["R1_by_threshold"]
    shown = "  ".join(
        f"@{t}={v:.3f}" if v is not None else f"@{t}=n/a" for t, v in sorted(by.items())
    )
    lines.append(
        f"  R1  duplicate authored claims   {shown}"
        f"   ({m['R1_claims_level1']} L1 claims vs {m['R1_claims_level0']} L0)"
    )
    for c in m["R1_duplicates"]:
        lines.append(f"        restates L0: {c[:88]}")
    r2 = m["R2_redefinition_rate"]
    unresolved = m["R2_unresolved_helper_writes"]
    lines.append(
        "  R2  helper fns redefined in L1  "
        + (f"{r2:.3f}" if r2 is not None else "  n/a")
        + f"  {m['R2_redefined_in_level1'] or '[]'} of {m['R2_functions_level0'] or '[]'}"
        + (f"   ({unresolved} writes unresolved)" if unresolved else "")
    )
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    targets = (
        {args.run.name: args.run}
        if args.run
        else {k: PROBES / v for k, v in TRACES.items()}
    )
    out = {}
    for name, path in targets.items():
        if not path.exists():
            print(f"missing: {path}", file=sys.stderr)
            return 2
        out[name] = measure(path)
        print(render(name, out[name]))
        print()

    usable = [m for m in out.values() if m.get("level_1_actions")]
    if len(usable) > 1:
        tot1 = sum(m["level_1_actions"] for m in usable)
        print("POOLED across level-1 traces")
        print(f"  level-1 actions            {tot1}")
        print(
            f"  R3  settled-law re-probing {sum(m['R3_count'] for m in usable) / tot1:.3f}"
        )
        print(
            f"  R3s inert-base re-test     "
            f"{sum(m['R3_strict_count'] for m in usable) / tot1:.3f}"
        )
        c1 = sum(m["R1_claims_level1"] for m in usable)
        if c1:
            d = sum(
                round((m["R1_duplicate_rate"] or 0) * m["R1_claims_level1"])
                for m in usable
            )
            print(f"  R1  duplicate claims       {d / c1:.3f}  ({d}/{c1})")
    if args.json:
        args.json.write_text(json.dumps(out, indent=2, default=str) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
