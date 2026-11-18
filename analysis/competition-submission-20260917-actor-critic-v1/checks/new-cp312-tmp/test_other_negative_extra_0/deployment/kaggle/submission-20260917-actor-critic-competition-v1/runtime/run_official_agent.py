"""Run the pinned official ARC agent launcher with fail-fast thread errors."""

from __future__ import annotations

import os
import runpy
import sys
import threading
from pathlib import Path

THREAD_FAILURE_EXIT_CODE = 70


def _fail_fast_thread_exception(args: threading.ExceptHookArgs) -> None:
    """Keep the traceback, then make a game-thread failure fail the notebook."""
    threading.__excepthook__(args)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(THREAD_FAILURE_EXIT_CODE)


def run(main_path: Path, arguments: list[str]) -> None:
    """Execute the unchanged official launcher in this guarded subprocess."""
    main_path = main_path.resolve()
    if not main_path.is_file():
        raise FileNotFoundError(f"official launcher does not exist: {main_path}")
    os.chdir(main_path.parent)
    sys.path.insert(0, str(main_path.parent))
    sys.argv = [str(main_path), *arguments]
    threading.excepthook = _fail_fast_thread_exception
    runpy.run_path(str(main_path), run_name="__main__")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(
            "usage: run_official_agent.py MAIN_PATH [official launcher arguments...]"
        )
    run(Path(sys.argv[1]), sys.argv[2:])


if __name__ == "__main__":
    main()
