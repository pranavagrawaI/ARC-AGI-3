"""Read a tycho_nooa run as a thinking-aloud protocol.

A run leaves four JSONL streams. Separately each is a log; together they are a
record of what both agents thought, what they looked at, and what made them
think at all:

    llm-trace.jsonl              per call: reasoning text, content, tokens, cost
    tool-trace.jsonl             actor's tool calls WITH results
    builder-tool-trace.jsonl     builder's tool calls WITH results
    trigger-builder-trace.jsonl  why the builder fired, and what the actor did

New traces share each provider `tool_call_id` between the LLM and tool rows.
Older traces lack that key and retain the order-only fallback in
`_pair_calls_with_tools`.

Usage:
    python tools/read_protocol.py <run-dir> triggers
    python tools/read_protocol.py <run-dir> narrative --who actor
    python tools/read_protocol.py <run-dir> narrative --who builder --turn 4
    python tools/read_protocol.py <run-dir> summary
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    run = Path(args.run_dir)
    if not (run / "llm-trace.jsonl").exists():
        raise SystemExit(f"no llm-trace.jsonl under {run}")
    {"summary": show_summary, "triggers": show_triggers, "narrative": show_narrative}[
        args.mode
    ](run, args)


def show_summary(run: Path, args: argparse.Namespace) -> None:
    """One screen: who spoke, how much it cost, how far context grew."""
    calls = _load(run / "llm-trace.jsonl")
    print(f"# {run.name}\n")
    for who in ("actor", "builder"):
        mine = [c for c in calls if c["source"] == who]
        if not mine:
            continue
        prompts = [c["prompt_tokens"] for c in mine]
        reasoning = [c.get("reasoning") or "" for c in mine]
        print(
            f"{who:<8} calls {len(mine):>4}  "
            f"cost ${sum(c.get('cost_usd') or 0 for c in mine):>6.3f}  "
            f"prompt {prompts[0]:>6} -> {prompts[-1]:>6} (max {max(prompts)})  "
            f"reasoning {sum(len(r) for r in reasoning):>8,} chars"
        )
    empty = sum(1 for c in calls if not (c.get("content") or "").strip())
    print(
        f"\n{empty}/{len(calls)} calls returned empty content (reasoning-channel model)"
    )


def show_triggers(run: Path, args: argparse.Namespace) -> None:
    """What summoned the builder, what it changed, and whether the actor listened.

    Read this first. It answers "what is the builder thinking about" more
    directly than the reasoning text does, because the builder only ever thinks
    about what the trigger points at.
    """
    events = _load(run / "trigger-builder-trace.jsonl")

    print("## Trigger checks (blank cause = did not fire)\n")
    for e in [x for x in events if x["event"] == "mechanical_trigger"]:
        mark = "FIRED" if e["fired"] else "  -  "
        print(f"  turn {e['turn_in_level']:>3}  {mark}  {e.get('cause') or ''}")

    print("\n## Builder passes\n")
    for e in [x for x in events if x["event"] == "builder_result"]:
        changed = [
            k.rsplit("/", 1)[-1]
            for k, v in e["artifact_hashes"].items()
            if v.get("changed")
        ]
        print(
            f"  turn {e['turn_in_level']:>3}  wm_forced={e['world_model_checkpoint_forced']!s:<5} "
            f"report_forced={e['report_forced']!s:<5} changed={changed}"
        )

    print("\n## Actor uptake\n")
    uptake = [x for x in events if x["event"] == "actor_uptake"]
    for e in uptake:
        recommended = (e.get("recommended_action") or {}).get("action") or "none"
        actual = (e.get("actor_action") or {}).get("action") or "error"
        followed = e.get("followed_recommended_action")
        print(
            f"  turn {e['turn_in_level']:>3}  recommended={recommended:<9} "
            f"actual={actual:<9} followed={str(followed) if followed is not None else 'n/a':<5} "
            f"tested_report={e['tested_report']}"
        )
    if uptake:
        print(
            f"\n  followed {sum(e.get('followed_recommended_action') is True for e in uptake)}/{len(uptake)}, "
            f"tested {sum(e['tested_report'] for e in uptake)}/{len(uptake)}"
        )


def show_narrative(run: Path, args: argparse.Namespace) -> None:
    """Interleave each call's reasoning with the tools it called and what came back."""
    calls = [c for c in _load(run / "llm-trace.jsonl") if c["source"] == args.who]
    tools = _load(
        run
        / ("tool-trace.jsonl" if args.who == "actor" else "builder-tool-trace.jsonl")
    )

    for call, used in _pair_calls_with_tools(calls, tools):
        turn = used[0]["turn_in_level"] if used else None
        if args.turn is not None and turn != args.turn:
            continue
        print("=" * 78)
        print(
            f"call {call['call_index']}  [{args.who}]  turn {turn}  {call['prompt_tokens']} prompt tokens"
        )
        print("-" * 78)
        print(_clip(call.get("reasoning") or "(no reasoning)", args.chars))
        for entry in used:
            print(
                f"\n  -> {entry['name']}({_clip(json.dumps(entry['arguments']), 300)})  [{entry['status']}]"
            )
            print(f"     {_clip(str(entry.get('result')), args.result_chars)}")
        print()


def _pair_calls_with_tools(
    calls: list[dict[str, Any]], tools: list[dict[str, Any]]
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Join by provider tool-call id, falling back for legacy traces only.

    Executed-tool traces intentionally omit some requested calls. Consuming by
    order shifts every later result after such a gap, so it is safe only for old
    rows that have no correlation key at all.
    """
    if all("tool_call_ids" in call for call in calls) and all(
        tool.get("tool_call_id") is not None for tool in tools
    ):
        by_id = {tool["tool_call_id"]: tool for tool in tools}
        return [
            (
                call,
                [
                    by_id[call_id]
                    for call_id in call["tool_call_ids"]
                    if call_id in by_id
                ],
            )
            for call in calls
        ]

    paired, cursor = [], 0
    for call in calls:
        count = len(call.get("tool_calls") or [])
        paired.append((call, tools[cursor : cursor + count]))
        cursor += count
    return paired


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _clip(text: str, limit: int) -> str:
    return (
        text
        if len(text) <= limit
        else text[:limit] + f"... [+{len(text) - limit} chars]"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("run_dir")
    parser.add_argument(
        "mode",
        choices=["summary", "triggers", "narrative"],
        nargs="?",
        default="summary",
    )
    parser.add_argument("--who", choices=["actor", "builder"], default="actor")
    parser.add_argument("--turn", type=int, default=None)
    parser.add_argument(
        "--chars", type=int, default=2000, help="max reasoning chars per call"
    )
    parser.add_argument(
        "--result-chars", type=int, default=600, help="max tool-result chars"
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
