"""Regenerate the upstream prompt snapshots. Run ONLY against unmodified upstream code."""
from __future__ import annotations

from pathlib import Path

from inference.agent.tool_agent import ToolAgent
from prompt_cases import prompt_cases, render_case


def main() -> None:
    out_dir = Path(__file__).parent / "fixtures"
    out_dir.mkdir(exist_ok=True)
    for name, case in prompt_cases().items():
        agent = ToolAgent(model="snapshot-model")
        (out_dir / f"prompt_{name}.txt").write_text(render_case(agent, case), encoding="utf-8")
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
