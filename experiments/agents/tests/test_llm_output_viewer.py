from __future__ import annotations

import importlib.util
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[3]
VIEWER = (
    REPO
    / "experiments/agents/0002-llm-repl-agent"
    / "output_viewer.py"
)


def _load_viewer():
    spec = importlib.util.spec_from_file_location("llm_output_viewer", VIEWER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {VIEWER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(viewer, runs_root: Path, path: str) -> tuple[int, bytes]:
    handler = viewer.Handler.__new__(viewer.Handler)
    handler.server = SimpleNamespace(runs_root=runs_root)
    handler.path = path
    handler.command = "GET"
    handler.requestline = f"GET {path} HTTP/1.1"
    handler.request_version = "HTTP/1.1"
    handler.wfile = BytesIO()
    handler.log_message = lambda *_args: None
    handler.do_GET()
    response = handler.wfile.getvalue()
    status = int(response.split(b" ", 2)[1])
    return status, response.partition(b"\r\n\r\n")[2]


class LlmOutputViewerPageTests(unittest.TestCase):
    def test_renders_one_accessible_replayer_instead_of_prototype_variants(self) -> None:
        viewer = _load_viewer()
        data = {
            "run_name": "run-1",
            "manifest": {},
            "summary": {
                "status": "budget_exhausted",
                "model": "model-1",
                "game": "anonymous-game",
                "actions": 2,
                "levels_completed": 0,
                "model_calls": 3,
                "python_calls": 1,
                "usage": {"total_tokens": 42},
            },
            "events": [],
        }

        page = viewer.render_page(data=data, run_names=["run-1"]).decode()

        self.assertIn("<title>LLM Run Replayer</title>", page)
        self.assertIn('aria-label="Replay position"', page)
        self.assertIn('aria-label="Play replay"', page)
        self.assertIn('aria-label="Jump to start"', page)
        self.assertIn('aria-label="Previous action"', page)
        self.assertIn('aria-label="Next action"', page)
        self.assertIn('aria-label="Jump to end"', page)
        self.assertIn('id="reasoning-log"', page)
        self.assertNotIn("Prototype view switcher", page)
        self.assertNotIn("__INITIAL_VARIANT__", page)
        self.assertNotIn("?variant=", page)

    def test_terminal_events_are_attached_to_the_last_attempt_not_setup(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            run_dir = runs_root / "run-test"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps({"actions": 1, "status": "budget_exhausted"}),
                encoding="utf-8",
            )
            records = [
                {"type": "observation", "event_index": 1, "action_step": 0},
                {"type": "model_response", "event_index": 2, "action_step": 2},
                {
                    "type": "model_budget_exhausted",
                    "event_index": 3,
                    "reason": "token limit",
                },
                {
                    "type": "run_stopped",
                    "event_index": 4,
                    "stop_reason": "model_budget_exhausted",
                },
            ]
            (run_dir / "process.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            data = viewer.load_run(runs_root, "run-test")

        by_type = {event["type"]: event for event in data["events"]}
        self.assertEqual(by_type["observation"]["action_step"], 0)
        self.assertEqual(by_type["model_budget_exhausted"]["action_step"], 2)
        self.assertEqual(by_type["run_stopped"]["action_step"], 2)

    def test_python_event_preserves_every_nonempty_output_channel(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            run_dir = runs_root / "run-test"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
            tool_output = json.dumps(
                {
                    "ok": False,
                    "stdout": "printed before failure",
                    "stderr": "warning",
                    "traceback": "Traceback: boom",
                }
            )
            (run_dir / "process.jsonl").write_text(
                json.dumps(
                    {
                        "type": "python_tool",
                        "event_index": 1,
                        "action_step": 1,
                        "output": tool_output,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            data = viewer.load_run(runs_root, "run-test")

        self.assertEqual(
            data["events"][0]["result"]["channels"],
            [
                {"label": "Stdout", "text": "printed before failure"},
                {"label": "Stderr", "text": "warning"},
                {"label": "Traceback", "text": "Traceback: boom"},
            ],
        )

    def test_selecting_a_reasoning_step_does_not_scroll_the_whole_page(self) -> None:
        viewer = _load_viewer()

        page = viewer.render_page(
            data={
                "run_name": "run-1",
                "manifest": {},
                "summary": {"actions": 0, "usage": {}},
                "events": [],
            },
            run_names=["run-1"],
        ).decode()

        self.assertNotIn("scrollIntoView", page)
        self.assertIn("list.scrollTop +=", page)

    def test_global_shortcuts_ignore_focused_interactive_controls(self) -> None:
        viewer = _load_viewer()

        page = viewer.render_page(
            data={
                "run_name": "run-1",
                "manifest": {},
                "summary": {"actions": 0, "usage": {}},
                "events": [],
            },
            run_names=["run-1"],
        ).decode()

        self.assertIn(
            "input, select, textarea, button, a, summary, [contenteditable=true]",
            page,
        )

    def test_asset_resolution_rejects_run_symlinks_outside_the_runs_root(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            runs_root = root / "runs"
            runs_root.mkdir()
            valid_run = runs_root / "run-valid"
            valid_run.mkdir()
            valid_image = valid_run / "frame.png"
            valid_image.write_bytes(b"png")
            outside_run = root / "run-outside"
            outside_run.mkdir()
            (outside_run / "frame.png").write_bytes(b"outside")
            (outside_run / "process.jsonl").write_text("{}\n", encoding="utf-8")
            (runs_root / "run-link").symlink_to(outside_run, target_is_directory=True)

            resolved = viewer.resolve_asset_path(
                runs_root, "run-valid", "frame.png"
            )
            with self.assertRaisesRegex(ValueError, "unknown run"):
                viewer.resolve_asset_path(runs_root, "run-link", "frame.png")
            run_names = viewer._run_names(runs_root)

        self.assertEqual(resolved, valid_image.resolve())
        self.assertEqual(run_names, [])

    def test_http_handler_serves_run_and_png_but_rejects_traversal(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            run_dir = runs_root / "run-valid"
            observations = run_dir / "workspace" / "observations"
            observations.mkdir(parents=True)
            (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
            (run_dir / "process.jsonl").write_text(
                json.dumps({"type": "observation", "action_step": 0}) + "\n",
                encoding="utf-8",
            )
            image = observations / "step-00000.png"
            image.write_bytes(b"png bytes")

            page_status, page = _request(
                viewer, runs_root, "/?run=run-valid"
            )
            image_status, image_body = _request(
                viewer,
                runs_root,
                "/asset?run=run-valid&file=workspace/observations/step-00000.png",
            )
            traversal_status, _ = _request(
                viewer,
                runs_root,
                "/asset?run=run-valid&file=../../README.md",
            )

        self.assertEqual(page_status, 200)
        self.assertIn(b"<title>LLM Run Replayer</title>", page)
        self.assertEqual(image_status, 200)
        self.assertEqual(image_body, b"png bytes")
        self.assertEqual(traversal_status, 403)


if __name__ == "__main__":
    unittest.main()
