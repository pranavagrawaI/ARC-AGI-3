"""Plan or execute one throwaway 0002 vs crude-0006 ls20 micro-pair."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
AGENTS_ROOT = HERE.parents[1]
BASELINE_RUNNER = AGENTS_ROOT / "0002-llm-repl-agent" / "run.py"
TREATMENT_RUNNER = HERE / "run_treatment.py"
CONFIRMATION = "execute-ls20-crude-0006-pair"


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--model", required=True)
    value.add_argument("--model-profile", required=True)
    value.add_argument("--endpoint")
    value.add_argument("--env-seed", type=int, default=0)
    value.add_argument("--model-seed", type=int, default=0)
    value.add_argument("--pair-order-seed", type=int, default=0)
    value.add_argument("--temperature", type=float, default=1.0)
    value.add_argument("--max-completion-tokens", type=int, default=4096)
    value.add_argument("--max-actions", type=int, default=30)
    value.add_argument("--max-model-calls", type=int, default=60)
    value.add_argument("--max-reported-tokens", type=int, default=120_000)
    value.add_argument("--max-context-turns", type=int, default=6)
    value.add_argument("--max-model-rounds-per-action", type=int, default=4)
    value.add_argument("--max-decision-repairs", type=int, default=3)
    value.add_argument("--output-root", type=Path)
    value.add_argument("--execute-gameplay", action="store_true")
    value.add_argument("--confirm")
    return value


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command(args: argparse.Namespace, *, condition: str, run_dir: Path) -> list[str]:
    runner = BASELINE_RUNNER if condition == "baseline" else TREATMENT_RUNNER
    command = [
        sys.executable,
        str(runner),
        "--game", "ls20",
        "--model", args.model,
        "--model-profile", args.model_profile,
        "--operation-mode", "offline",
        "--env-seed", str(args.env_seed),
        "--model-seed", str(args.model_seed),
        "--temperature", str(args.temperature),
        "--max-completion-tokens", str(args.max_completion_tokens),
        "--max-actions", str(args.max_actions),
        "--max-model-calls", str(args.max_model_calls),
        "--max-reported-tokens", str(args.max_reported_tokens),
        "--max-context-turns", str(args.max_context_turns),
        "--max-model-rounds-per-action", str(args.max_model_rounds_per_action),
        "--max-decision-repairs", str(args.max_decision_repairs),
        "--drop-stale-python-context",
        "--no-retrospective",
        "--run-dir", str(run_dir),
    ]
    if args.endpoint:
        command.extend(("--endpoint", args.endpoint))
    return command


def main() -> None:
    args = parser().parse_args()
    if args.execute_gameplay and args.confirm != CONFIRMATION:
        raise SystemExit(f"gameplay requires --confirm {CONFIRMATION}")
    if not args.execute_gameplay and args.confirm is not None:
        raise SystemExit("--confirm is accepted only with --execute-gameplay")
    for name in (
        "max_completion_tokens", "max_actions", "max_model_calls",
        "max_reported_tokens", "max_context_turns", "max_model_rounds_per_action",
        "max_decision_repairs",
    ):
        if getattr(args, name) < 1:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = args.output_root or HERE / "artifacts" / f"pair-{stamp}"
    if root.exists():
        raise SystemExit(f"output root already exists: {root}")
    root.mkdir(parents=True)

    order = ["baseline", "treatment"]
    random.Random(args.pair_order_seed).shuffle(order)
    commands = {
        condition: _command(args, condition=condition, run_dir=root / condition)
        for condition in order
    }
    plan = {
        "schema": "crude-0006-micro-pair-v0",
        "question": (
            "Does the crude capability package improve preserved hypotheses, contradiction "
            "reconciliation, informative probe choice, or action efficiency versus 0002?"
        ),
        "status": "authorized_for_execution" if args.execute_gameplay else "preflight_only",
        "game": "ls20",
        "order": order,
        "pairing": {
            "model": args.model,
            "model_profile": args.model_profile,
            "endpoint": args.endpoint,
            "environment_seed": args.env_seed,
            "model_seed": args.model_seed,
            "temperature": args.temperature,
            "max_completion_tokens": args.max_completion_tokens,
            "max_actions": args.max_actions,
            "max_model_calls": args.max_model_calls,
            "max_reported_tokens": args.max_reported_tokens,
            "max_context_turns": args.max_context_turns,
            "max_model_rounds_per_action": args.max_model_rounds_per_action,
            "max_decision_repairs": args.max_decision_repairs,
            "provider_only": os.environ.get("OPENROUTER_PROVIDER_ONLY"),
            "reasoning": os.environ.get("OPENROUTER_REASONING"),
        },
        "source_hashes": {
            "baseline_runner": _sha256(BASELINE_RUNNER),
            "treatment_runner": _sha256(TREATMENT_RUNNER),
            "prototype_agent": _sha256(HERE / "prototype_agent.py"),
        },
        "commands": {key: value for key, value in commands.items()},
        "execution": [],
    }
    _write(root / "pair-plan.json", plan)

    if not args.execute_gameplay:
        print(json.dumps({
            "status": "preflight_only",
            "order": order,
            "plan": str(root / "pair-plan.json"),
            "execution_guard": f"--execute-gameplay --confirm {CONFIRMATION}",
        }, indent=2, sort_keys=True))
        return

    for condition in order:
        completed = subprocess.run(commands[condition], check=False)
        manifest_path = root / condition / "manifest.json"
        member = {
            "condition": condition,
            "returncode": completed.returncode,
            "manifest": str(manifest_path) if manifest_path.is_file() else None,
        }
        plan["execution"].append(member)
        _write(root / "pair-plan.json", plan)
        if completed.returncode != 0:
            plan["status"] = "member_failed_pair_stopped"
            _write(root / "pair-plan.json", plan)
            raise SystemExit(completed.returncode)

    analyzer = subprocess.run(
        [sys.executable, str(HERE / "analyze_micro_pair.py"), "--pair-root", str(root)],
        check=False,
    )
    plan["status"] = "pair_complete" if analyzer.returncode == 0 else "analysis_failed"
    _write(root / "pair-plan.json", plan)
    raise SystemExit(analyzer.returncode)


if __name__ == "__main__":
    main()
