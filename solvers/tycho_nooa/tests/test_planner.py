from __future__ import annotations

import pytest

from tycho_nooa.planner import planner_command


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("python3 plan.py bfs", "python plan.py bfs"),
        ('subprocess.run(["python", "plan.py", "astar"])', "python plan.py astar"),
        ('runpy.run_path("plan.py")', "python plan.py"),
        ('subprocess.run(["python", "other.py"])', None),
        ("subprocess.run(", None),
        ("", None),
    ],
)
def test_planner_command_recognizes_only_seeded_planner_execution(
    source: str, expected: str | None
) -> None:
    assert planner_command(source) == expected
