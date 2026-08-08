"""H2: can a small model name the goal relation prospectively, from the digest alone?

`digest_probe.py` established that the machinery finds the goal relation retrospectively:
it enumerates invariant x mutable pairs and asks which one flips from mismatch to match
between spawn and the win. That is a verifier, not a policy. At play time the win frame
does not exist yet, so the flip cannot be computed and the machinery cannot choose.

This probe measures the one job the inverted bet assigns to the model. Build the digest
from a PREFIX of the level only - no win frame, no outcome. Enumerate the candidate
pairs. Ask the model to name the pair the game is about. Score against the flip pair the
full level later confirms.

The model never sees a frame, a game name, an action, or the outcome. Its whole input is
a few hundred tokens of sprite bookkeeping.

    source .env && export OPENROUTER_API_KEY
    uv run python experiments/hypothesis/0001-inverted-authority/select_probe.py --dry-run

Baselines, all computed on the same candidate lists:

    random      1/N
    biggest     largest invariant x largest mutable
    busiest     largest invariant x most-changed mutable
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from digest_probe import REPO, digest, lenses, level_slice, load  # noqa: E402

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "qwen/qwen3.5-9b"

# Traces that reach a WIN, so a level's end frame is a real end. Game identity stays out
# of every model-visible string; it is here only so the report can name the cell.
TRACES = {
    "A": (REPO / "recordings/ae20a3e4-ca6e-4f29-b463-042fd0dd6710", [1, 3, 5]),
    "B": (REPO / "human-play/cd82/20260701T095558Z/raw-recordings"
          / "8b4a4e3d-0000-0000-0000-000000000000", [2, 3, 4, 5]),
}


def resolve(path: Path) -> Path:
    """Recording directories are guid-named; take the only child if the guid is stale."""
    if path.is_dir() and any(path.glob("*.jsonl")):
        return path
    parent = path.parent
    kids = [p for p in parent.iterdir() if p.is_dir() and any(p.glob("*.jsonl"))]
    if len(kids) != 1:
        raise FileNotFoundError(f"cannot resolve a single recording under {parent}")
    return kids[0]


# --- candidates and oracle ----------------------------------------------------------


def pairs_of(d: dict) -> list[tuple]:
    """Every invariant x mutable pair the machinery can state a relation about."""
    inv, mut = d["_tracks"]["inv"], d["_tracks"]["mut"]
    out = []
    for it in inv:
        if it.sprites[min(it.sprites)].area > 60:
            continue
        for mt in mut:
            if mt.sprites[min(mt.sprites)].area > 60:
                continue
            out.append((it, mt))
    return out


def oracle_pair(full: dict) -> tuple[int, int] | None:
    """The unique retrospective flip: false at spawn, true at the level's end."""
    frames = full["_tracks"]["frames"]
    hits = []
    for it, mt in pairs_of(full):
        i0, i1 = min(it.sprites), max(it.sprites)
        m0, m1 = min(mt.sprites), max(mt.sprites)
        spawn = lenses(it.sprites[i0].paint(frames[i0]), mt.sprites[m0].paint(frames[m0]))
        end = lenses(it.sprites[i1].paint(frames[i1]), mt.sprites[m1].paint(frames[m1]))
        if any(not spawn[k] and end[k] for k in spawn):
            hits.append((it.tid, mt.tid))
    return hits[0] if len(hits) == 1 else None


# --- the model-visible description --------------------------------------------------


def describe(d: dict, cands: list[tuple]) -> str:
    lines = []
    c = d["controlled"]
    if c:
        moves = {a: v for a, v in c["action_model"].items() if v != (0, 0)}
        lines.append(f"You control object {c['tid']}: area {c['area']}, shape {c['extent']}.")
        lines.append(f"Its response to each command: {moves}.")
    for r in d.get("resources", []):
        spend = "shrinks" if r["step"] < 0 else "grows"
        lines.append(f"Budget meter #{r['tid']}: {spend} by {abs(r['step'])} per action, "
                     f"on {r['spent']} of {r['observed']} actions, {r['refills']} refills. "
                     f"It tracks the action count, not the puzzle. Running it out ends the "
                     f"run, so it bounds how long you may take. It is never the objective.")
    lines.append("")
    lines.append("Objects that never changed appearance while you watched:")
    for x in d["invariant"]:
        lines.append(f"  #{x['tid']}  area {x['area']}  shape {x['extent']}  "
                     f"colours {x['colours']}  seen in {x['seen']} frames")
    lines.append("")
    lines.append("Objects that did change appearance:")
    for x in d["mutable"]:
        lines.append(f"  #{x['tid']}  area {x['area']}  shape {x['extent']}  "
                     f"colours {x['colours']}  changed {x['changes']} times "
                     f"in {x['seen']} frames")
    if d["attribution"]:
        lines.append("")
        lines.append("Changes that followed contact by the controlled object:")
        for a in d["attribution"]:
            for r in a["rows"][:2]:
                lines.append(f"  #{a['tid']} changed on {r['changes']} of "
                             f"{r['opportunities']} contacts with cells {r['contact_class']} "
                             f"(background rate {a['baseline_rate']:.2f})")
    lines.append("")
    lines.append("Candidate pairs (unchanging object, changing object):")
    for k, (it, mt) in enumerate(cands):
        i, m = it.sprites[min(it.sprites)], mt.sprites[min(mt.sprites)]
        lines.append(f"  {k}: #{it.tid} (area {i.area}, shape {i.extent}) with "
                     f"#{mt.tid} (area {m.area}, shape {m.extent})")
    return "\n".join(lines)


SYSTEM = (
    "You analyse an unfamiliar puzzle game. You cannot see it. You get only measurements "
    "another program made from watching part of one level.\n\n"
    "In this family of games one object states a target and another object is the "
    "workspace the player edits. The level ends when the workspace matches the target "
    "under some correspondence. Your only job is to say which pair that is.\n\n"
    "Answer with JSON and nothing else: {\"pair\": <index>, \"why\": \"<one short "
    "sentence>\"}"
)


def ask(prompt: str, model: str, key: str, tokens: int, think: bool, tries: int = 3,
        timeout: int = 300) -> tuple[int | None, str, str]:
    """Returns (choice, model text, note). A transport failure is data, not a crash."""
    req_body = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": prompt}],
        "temperature": 0,
        # Small Qwen models reason before answering and charge those tokens to the same
        # budget. At 400 the reasoning consumed all of it and content came back empty;
        # at 6000 a 15-way choice still finished with reason=length and no answer.
        # Reasoning length is the binding cost on an 8 GB card, so it is a probe knob.
        "max_tokens": tokens,
    }
    if not think:
        req_body["reasoning"] = {"enabled": False}
    body = json.dumps(req_body).encode()
    last = ""
    for attempt in range(tries):
        req = urllib.request.Request(
            API_URL, data=body,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                payload = json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
            continue
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            continue
        ch = payload["choices"][0]
        text = (ch["message"].get("content") or "").strip()
        u = payload.get("usage", {})
        note = (f"finish={ch.get('finish_reason')} "
                f"reasoning={u.get('completion_tokens_details', {}).get('reasoning_tokens')} "
                f"completion={u.get('completion_tokens')}")
        m = re.search(r'"pair"\s*:\s*(\d+)', text) or re.search(r"\b(\d+)\b", text)
        if m or attempt == tries - 1:
            return (int(m.group(1)) if m else None), text, note
        last = note + " (no answer in content)"
    return None, "", last


# --- baselines ----------------------------------------------------------------------


def baselines(cands: list[tuple], rng: random.Random) -> dict[str, int]:
    def area(t):
        return t.sprites[min(t.sprites)].area

    def changes(t):
        return sum(1 for i in sorted(t.sigs) if i - 1 in t.sigs and t.sigs[i] != t.sigs[i - 1])

    big_i = max({it.tid: it for it, _ in cands}.values(), key=area)
    big_m = max({mt.tid: mt for _, mt in cands}.values(), key=area)
    busy_m = max({mt.tid: mt for _, mt in cands}.values(), key=changes)
    idx = {(it.tid, mt.tid): k for k, (it, mt) in enumerate(cands)}
    return {
        "random": rng.randrange(len(cands)),
        "biggest": idx.get((big_i.tid, big_m.tid), -1),
        "busiest": idx.get((big_i.tid, busy_m.tid), -1),
        "scale": scale_rule(cands),
    }


def scale_rule(cands: list[tuple]) -> int:
    """The one pair whose extents differ by an integer factor and areas by its square.

    Added after both a 9B and a 397B model failed to state this relation from prompts
    that contained every number it needs. It is the third apparent 'the model must
    choose' case that dissolved into enumeration, so it belongs to the machinery.
    """
    hits = []
    for k, (it, mt) in enumerate(cands):
        i = it.sprites[min(it.sprites)]
        m = mt.sprites[min(mt.sprites)]
        for f in range(2, 6):
            if m.extent == (i.extent[0] * f, i.extent[1] * f) and m.area == i.area * f * f:
                hits.append(k)
                break
    return hits[0] if len(hits) == 1 else -1


# --- the run ------------------------------------------------------------------------


def cell(steps, level: int, prefix: float, knob: str = "colour") -> dict | None:
    lvl = level_slice(steps, level)
    if len(lvl) < 8:
        return None
    full = digest(lvl, knob, horizon=len(lvl))
    truth = oracle_pair(full)
    if truth is None:
        return None
    head = max(4, int(len(lvl) * prefix))
    part = digest(lvl, knob, horizon=head)
    cands = pairs_of(part)
    if not cands:
        return None
    idx = {(it.tid, mt.tid): k for k, (it, mt) in enumerate(cands)}
    if truth not in idx:
        return {"level": level, "n": len(cands), "answer": None,
                "note": "oracle pair not yet visible in the prefix"}
    return {"level": level, "n": len(cands), "answer": idx[truth], "truth": truth,
            "prompt": describe(part, cands), "cands": cands,
            "frames_seen": head, "frames_total": len(lvl)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--prefix", type=float, default=0.6)
    ap.add_argument("--tokens", type=int, default=6000)
    ap.add_argument("--think", action="store_true", help="allow chain of thought")
    ap.add_argument("--dry-run", action="store_true", help="build prompts, call nothing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "")
    if not args.dry_run and not key:
        print("OPENROUTER_API_KEY is unavailable; source .env first", file=sys.stderr)
        return 2

    rng = random.Random(args.seed)
    score = {"model": 0, "random": 0, "biggest": 0, "busiest": 0, "scale": 0}
    cells = 0
    for tag, (path, levels) in TRACES.items():
        steps = load(resolve(path))
        for level in levels:
            c = cell(steps, level, args.prefix)
            if c is None:
                print(f"[{tag}/L{level}] skipped: no unique retrospective flip")
                continue
            if c["answer"] is None:
                print(f"[{tag}/L{level}] skipped: {c['note']} ({c['n']} candidates)")
                continue
            cells += 1
            base = baselines(c["cands"], rng)
            for name, pick in base.items():
                score[name] += pick == c["answer"]
            head = (f"[{tag}/L{level}] {c['n']} candidates, "
                    f"{c['frames_seen']}/{c['frames_total']} frames seen, "
                    f"answer {c['answer']} (#{c['truth'][0]} with #{c['truth'][1]})")
            if args.dry_run:
                print(f"\n{'=' * 78}\n{head}\n{'=' * 78}\n{c['prompt']}")
                print(f"-- baselines {base}")
                continue
            pick, text, note = ask(c["prompt"], args.model, key, args.tokens, args.think)
            ok = pick == c["answer"]
            score["model"] += ok
            print(f"{head}\n    model -> {pick} [{'HIT' if ok else 'MISS'}]  {note}\n"
                  f"    {text[:220]}\n    baselines {base}", flush=True)

    print(f"\n{cells} scored cells")
    if not args.dry_run:
        print(f"model {score['model']}/{cells}   " + "   ".join(
            f"{k} {v}/{cells}" for k, v in score.items() if k != "model"))
    else:
        print("baselines " + "   ".join(
            f"{k} {v}/{cells}" for k, v in score.items() if k != "model"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
