"""Step-2 gate: run the critic policy over real Duck packets and report whether it
returns well-formed reports, at what cost.

    python -m inference.critic.gate --packets DIR --model qwen/qwen3.6-27b \
        --provider openrouter --base-url https://openrouter.ai/api/v1 --limit 12

Packets come from `inference.critic.replay --dump-dir DIR`. Notes are carried per
source game in replay order, so scratchpad compounding is exercised too.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from inference.critic.packet import CriticPacket
from inference.critic.policy import CriticModelConfig, CriticNotes, CriticPolicy, CriticPolicyConfig


def _packet_files(packets_dir: Path, limit: int, per_source: int) -> list[Path]:
    files = sorted(packets_dir.glob("*.packet.json"))
    by_source: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        by_source[path.name.split("_turn", 1)[0]].append(path)
    chosen: list[Path] = []
    # Round-robin across games so one long trace does not eat the budget.
    rounds = 0
    while len(chosen) < limit and rounds < per_source:
        for source in sorted(by_source):
            bucket = by_source[source]
            if rounds < len(bucket) and len(chosen) < limit:
                chosen.append(bucket[rounds])
        rounds += 1
    return chosen


def run_gate(
    packet_paths: list[Path],
    *,
    policy: CriticPolicy,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    notes_by_source: dict[str, CriticNotes] = defaultdict(CriticNotes)
    rows: list[dict[str, Any]] = []
    for index, path in enumerate(packet_paths, start=1):
        source = path.name.split("_turn", 1)[0]
        packet = CriticPacket.from_json(path.read_text(encoding="utf-8"))
        firing = policy.run(packet, notes_by_source[source])
        row = {
            "index": index,
            "source": source,
            "packet": path.name,
            "level": packet.level,
            "action_num": packet.action_num,
            "cause": packet.trigger_cause,
            "beliefs_nonempty": bool(any(packet.actor_beliefs.values())),
            "seconds": firing.seconds,
            "model_calls": firing.model_calls,
            "tool_calls": firing.tool_calls,
            "tool_errors": firing.tool_errors,
            "refused_action_calls": firing.refused_action_calls,
            "prompt_tokens": firing.prompt_tokens,
            "completion_tokens": firing.completion_tokens,
            "parsed": firing.parsed,
            "fallback": firing.fallback,
            "forced_final": firing.forced_final,
            "confidence": firing.report.confidence,
            "recommended_actions": list(firing.report.recommended_actions),
            "recommended_legal": firing.recommended_legal,
            "error": firing.error,
        }
        rows.append(row)
        stem = path.name.removesuffix(".packet.json")
        (out_dir / f"{stem}.firing.json").write_text(
            json.dumps(
                {**row, "report_text": firing.report_text, "report": firing.report.to_payload(), "scratchpad": firing.scratchpad, "transcript": firing.transcript},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print(
            f"[{index}/{len(packet_paths)}] {source} L{packet.level} a{packet.action_num} {packet.trigger_cause}: "
            f"{firing.seconds:.1f}s calls={firing.model_calls} tools={firing.tool_calls} "
            f"tok={firing.prompt_tokens}+{firing.completion_tokens} parsed={firing.parsed} "
            f"legal={firing.recommended_legal} rec={list(firing.report.recommended_actions)[:2]} {firing.error}"
            + (f"\n      Q: {firing.report.open_questions[0]}" if firing.report.open_questions else ""),
            flush=True,
        )

    def med(key: str) -> float | None:
        values = [float(r[key]) for r in rows if r.get(key) is not None]
        return round(statistics.median(values), 2) if values else None

    summary = {
        "firings": len(rows),
        "parsed": sum(r["parsed"] for r in rows),
        "fallback": sum(r["fallback"] for r in rows),
        "forced_final": sum(r["forced_final"] for r in rows),
        "errors": sum(bool(r["error"]) for r in rows),
        "tool_errors": sum(r["tool_errors"] for r in rows),
        "refused_action_calls": sum(r["refused_action_calls"] for r in rows),
        "recommended_present": sum(bool(r["recommended_actions"]) for r in rows),
        "recommended_legal": sum(bool(r["recommended_legal"]) for r in rows),
        "median_seconds": med("seconds"),
        "median_model_calls": med("model_calls"),
        "median_tool_calls": med("tool_calls"),
        "median_prompt_tokens": med("prompt_tokens"),
        "median_completion_tokens": med("completion_tokens"),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "total_completion_tokens": sum(r["completion_tokens"] for r in rows),
        "by_cause": {
            cause: sum(1 for r in rows if r["cause"] == cause) for cause in sorted({r["cause"] for r in rows})
        },
        "policy": policy.config.__dict__,
        "rows": rows,
    }
    (out_dir / "gate-summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packets", type=Path, required=True, help="directory of *.packet.json from replay --dump-dir")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--per-source", type=int, default=4, help="max packets taken from one game trace")
    parser.add_argument("--model", default=os.environ.get("CRITIC_MODEL_ID", "qwen/qwen3.6-27b"))
    parser.add_argument("--provider", default=os.environ.get("CRITIC_PROVIDER", "openrouter"))
    parser.add_argument("--base-url", default=os.environ.get("CRITIC_BASE_URL", "https://openrouter.ai/api/v1"))
    parser.add_argument("--api-key", default="", help="defaults to CRITIC_API_KEY, then OPENROUTER_API_KEY")
    parser.add_argument("--mode", choices=["question", "report"], default=CriticPolicyConfig.mode)
    parser.add_argument("--max-tool-steps", type=int, default=CriticPolicyConfig.max_tool_steps)
    parser.add_argument("--max-seconds", type=float, default=CriticPolicyConfig.max_seconds)
    parser.add_argument("--max-output-tokens", type=int, default=CriticPolicyConfig.max_output_tokens)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--request-timeout", type=float, default=180.0)
    args = parser.parse_args(argv)

    api_key = args.api_key or os.environ.get("CRITIC_API_KEY", "") or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and args.provider == "openrouter":
        print("missing API key: set OPENROUTER_API_KEY or pass --api-key", file=sys.stderr)
        return 2
    policy = CriticPolicy(
        CriticModelConfig(
            provider=args.provider,
            base_url=args.base_url,
            model_id=args.model,
            api_key=api_key,
            thinking=not args.no_thinking,
            request_timeout_seconds=args.request_timeout,
        ),
        CriticPolicyConfig(
            mode=args.mode,
            max_tool_steps=args.max_tool_steps,
            max_seconds=args.max_seconds,
            max_output_tokens=args.max_output_tokens,
        ),
    )
    packet_paths = _packet_files(args.packets, args.limit, args.per_source)
    if not packet_paths:
        print(f"no *.packet.json under {args.packets}", file=sys.stderr)
        return 2
    started = time.monotonic()
    summary = run_gate(packet_paths, policy=policy, out_dir=args.out)
    summary_line = {k: v for k, v in summary.items() if k not in {"rows", "policy"}}
    print(json.dumps(summary_line, indent=2))
    print(f"wall: {time.monotonic() - started:.0f}s; per-firing files in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
