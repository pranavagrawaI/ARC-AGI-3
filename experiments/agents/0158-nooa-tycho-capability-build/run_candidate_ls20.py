"""Run one exact, matched low-cost model candidate through the 0158 LS20 host."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import run_live_ls20 as base


@dataclass(frozen=True)
class Candidate:
    key: str
    model: str
    model_route: str
    provider_route_id: str
    provider_route: dict[str, object]
    runtime_name: str
    card_name: str
    approval_name: str
    qualification_name: str
    run_slug: str


CANDIDATES = {
    "deepseek-v4-flash": Candidate(
        key="deepseek-v4-flash",
        model="deepseek/deepseek-v4-flash-0731",
        model_route="openai/deepseek/deepseek-v4-flash-0731",
        provider_route_id="coreweave/fp8",
        provider_route={
            "order": ["coreweave/fp8"],
            "only": ["coreweave/fp8"],
            "allow_fallbacks": False,
            "require_parameters": False,
            "quantizations": ["fp8"],
        },
        runtime_name="deepseek-v4-flash",
        card_name="0158-deepseek-v4-flash-coreweave-matched-44-action-v32.md",
        approval_name="0158-ls20-deepseek-v4-flash-coreweave-v32.json",
        qualification_name="qualification-deepseek-v4-flash-coreweave-v32.json",
        run_slug="deepseek-v4-flash-coreweave-matched-44-v32",
    ),
    "qwen37-flash": Candidate(
        key="qwen37-flash",
        model="qwen/qwen3.7-flash",
        model_route="openai/qwen/qwen3.7-flash",
        provider_route_id="alibaba",
        provider_route={
            "order": ["alibaba"],
            "only": ["alibaba"],
            "allow_fallbacks": False,
            "require_parameters": False,
        },
        runtime_name="qwen37-flash",
        card_name="0158-qwen37-flash-matched-44-action-v32.md",
        approval_name="0158-ls20-qwen37-flash-v32.json",
        qualification_name="qualification-qwen37-flash-v32.json",
        run_slug="qwen37-flash-matched-44-v32",
    ),
    "muse-glimmer-30b": Candidate(
        key="muse-glimmer-30b",
        model="meta/muse-glimmer-30b",
        model_route="openai/meta/muse-glimmer-30b",
        provider_route_id="deepinfra/bf16",
        provider_route={
            "order": ["deepinfra/bf16"],
            "only": ["deepinfra/bf16"],
            "allow_fallbacks": False,
            "require_parameters": False,
            "quantizations": ["bf16"],
        },
        runtime_name="muse-glimmer-30b",
        card_name="0158-muse-glimmer-30b-matched-44-action-v32.md",
        approval_name="0158-ls20-muse-glimmer-30b-v32.json",
        qualification_name="qualification-muse-glimmer-30b-v32.json",
        run_slug="muse-glimmer-30b-matched-44-v32",
    ),
}


def configure(candidate: Candidate) -> None:
    runtime = base.EXPERIMENT / "candidate-runtimes" / candidate.runtime_name
    base.RUNTIME = runtime
    base.CARD = base.REPO / "experiments/cards" / candidate.card_name
    base.APPROVAL = base.REPO / "experiments/approvals" / candidate.approval_name
    base.CONSUMPTION = base.APPROVAL.with_suffix(".consumed.json")
    base.QUALIFICATION = base.EXPERIMENT / candidate.qualification_name
    base.MODEL = candidate.model
    base.MODEL_ROUTE = candidate.model_route
    base.PROVIDER_ROUTE_ID = candidate.provider_route_id
    base.PROVIDER_ROUTE = candidate.provider_route
    base.REASONING_EFFORT = None
    # Frozen provider validation requires this compatibility switch. No
    # candidate-specific reasoning parameter is supplied, so nothing
    # treatment-bearing is silently dropped.
    base.DROP_PARAMS = True
    base.RUN_SLUG = candidate.run_slug
    base.CANDIDATE_IDENTITY_FILES = (Path(__file__).resolve(), runtime / "launcher.py")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate", choices=sorted(CANDIDATES), required=True)
    parser.add_argument("--run-root", type=Path)
    args = parser.parse_args()
    candidate = CANDIDATES[args.candidate]
    configure(candidate)
    run_root = (args.run_root or base._run_root()).resolve()
    print(base.json.dumps(base.run(run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
