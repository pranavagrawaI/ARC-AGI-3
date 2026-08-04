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

    def test_desktop_layout_uses_equal_game_and_reasoning_columns(self) -> None:
        viewer = _load_viewer()
        page = viewer.render_page(
            data={
                "run_name": "run-1",
                "manifest": {},
                "summary": {"status": "complete", "actions": 1, "usage": {}},
                "events": [],
            },
            run_names=["run-1"],
        ).decode("utf-8")

        self.assertIn(
            "grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);",
            page,
        )

    def test_page_polls_live_run_data_and_renders_human_feedback(self) -> None:
        viewer = _load_viewer()
        page = viewer.render_page(
            data={
                "run_name": "run-live",
                "manifest": {},
                "summary": {"status": "running", "actions": 1, "usage": {}},
                "events": [],
            },
            run_names=["run-live"],
        ).decode("utf-8")

        self.assertIn("/api/run?run=", page)
        self.assertIn("setInterval(refreshLiveRun", page)
        self.assertIn("human_feedback", page)
        self.assertIn("Human feedback after stage", page)

    def test_page_offers_feedback_centered_steering_audit_without_causal_claims(self) -> None:
        viewer = _load_viewer()
        page = viewer.render_page(
            data={
                "run_name": "run-steered",
                "manifest": {},
                "summary": {"status": "complete", "actions": 2, "usage": {}},
                "events": [],
                "steering_audit": [],
            },
            run_names=["run-steered"],
        ).decode("utf-8")

        self.assertIn('id="steering-mode"', page)
        self.assertIn("Steering audit", page)
        self.assertIn("Before feedback", page)
        self.assertIn("Human intervention", page)
        self.assertIn("Observed after feedback", page)
        self.assertIn("buildSteeringAudit", page)
        self.assertIn("request_payload", page)
        self.assertIn("response_body", page)
        self.assertIn("process_event", page)
        self.assertNotIn("caused by feedback", page.lower())
        self.assertNotIn("shows observed uptake", page.lower())
        self.assertIn("temporal sequence is not proof", page.lower())
        self.assertIn("card.dataset.cardKey", page)
        self.assertIn("for (const audit of audits)", page)
        self.assertNotIn("[...audits].reverse()", page)
        self.assertIn(
            "for (const event of events) body.append(eventBlock(event));",
            page,
        )

    def test_embedded_run_data_cannot_terminate_the_script_element(self) -> None:
        viewer = _load_viewer()
        hostile_text = "</script><script>globalThis.pwned = true</script>"

        page = viewer.render_page(
            data={
                "run_name": "run-live",
                "manifest": {},
                "summary": {"status": "running", "actions": 1, "usage": {}},
                "events": [{"type": "human_feedback", "text": hostile_text}],
            },
            run_names=["run-live"],
        ).decode("utf-8")

        self.assertNotIn(hostile_text, page)
        self.assertIn(r"<\/script><script>globalThis.pwned = true<\/script>", page)

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

    def test_learning_loop_commitments_and_results_are_first_class_events(self) -> None:
        viewer = _load_viewer()
        commitment = {
            "working_state": {"belief": "blue moves right"},
            "test": "click blue",
            "prediction": [
                {"observable": "changed_pixels", "op": ">", "value": 0}
            ],
            "history_steps": [1],
            "reconciliation": None,
        }
        checks = [
            {
                "observable": "changed_pixels",
                "op": ">",
                "value": 0,
                "observed": 0,
                "matched": False,
            }
        ]
        objective_result = {
            "claim": "clicking arbitrary cells is the objective",
            "outcome": "falsified",
            "remaining_steps": 0,
            "check": {
                "observable": "levels_completed",
                "op": ">",
                "value": 0,
                "observed": 0,
                "matched": False,
            },
        }
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            run_dir = runs_root / "run-test"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "environment": {
                            "model": "google/gemma-4-31b-it",
                            "game": "cd82",
                        }
                    }
                ),
                encoding="utf-8",
            )
            records = [
                {
                    "type": "decision",
                    "event_index": 1,
                    "action_step": 1,
                    "action": 6,
                    "data": {"x": 2, "y": 3},
                    "reasoning": {
                        "reason": "click blue",
                        "commitment": commitment,
                    },
                },
                {
                    "type": "prediction_result",
                    "event_index": 2,
                    "action_step": 1,
                    "outcome": "mismatch",
                    "checks": checks,
                    "objective_result": objective_result,
                },
            ]
            (run_dir / "process.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            data = viewer.load_run(runs_root, "run-test")
            page = viewer.render_page(data=data, run_names=["run-test"]).decode()

        self.assertEqual(data["events"][0]["commitment"], commitment)
        self.assertEqual(data["events"][1]["outcome"], "mismatch")
        self.assertEqual(data["events"][1]["checks"], checks)
        self.assertEqual(data["events"][1]["objective_result"], objective_result)
        self.assertEqual(data["summary"]["model"], "google/gemma-4-31b-it")
        self.assertEqual(data["summary"]["game"], "cd82")
        self.assertIn("Learning-loop commitment", page)
        self.assertIn("Prediction result", page)
        self.assertIn("Objective result:", page)

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

    def test_run_discovery_accepts_custom_directory_names(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            custom_run = runs_root / "smoke-payload-test"
            custom_run.mkdir()
            (custom_run / "process.jsonl").write_text("{}\n", encoding="utf-8")

            run_names = viewer._run_names(runs_root)

        self.assertEqual(run_names, ["smoke-payload-test"])

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

    def test_http_handler_serves_allowlisted_raw_evidence_only(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            run_dir = runs_root / "run-valid"
            responses = run_dir / "response-bodies"
            requests = run_dir / "request-payloads"
            responses.mkdir(parents=True)
            requests.mkdir()
            raw_process_event = '{ "type": "decision", "event_index": 6 }\n'
            (run_dir / "process.jsonl").write_text(
                raw_process_event,
                encoding="utf-8",
            )
            (run_dir / "manifest.json").write_text("{}", encoding="utf-8")
            (responses / "000006.raw").write_bytes(b"raw provider body")
            (requests / "000006.json").write_text('{"model":"m"}', encoding="utf-8")
            (run_dir / "secret.txt").write_text("not public", encoding="utf-8")

            raw_status, raw_body = _request(
                viewer,
                runs_root,
                "/evidence?run=run-valid&file=response-bodies/000006.raw",
            )
            request_status, request_body = _request(
                viewer,
                runs_root,
                "/evidence?run=run-valid&file=request-payloads/000006.json",
            )
            secret_status, _ = _request(
                viewer,
                runs_root,
                "/evidence?run=run-valid&file=secret.txt",
            )
            traversal_status, _ = _request(
                viewer,
                runs_root,
                "/evidence?run=run-valid&file=response-bodies/../../secret.txt",
            )
            event_status, event_body = _request(
                viewer,
                runs_root,
                "/process-event?run=run-valid&event=6",
            )
            invalid_event_status, _ = _request(
                viewer,
                runs_root,
                "/process-event?run=run-valid&event=../../6",
            )

        self.assertEqual((raw_status, raw_body), (200, b"raw provider body"))
        self.assertEqual((request_status, request_body), (200, b'{"model":"m"}'))
        self.assertEqual(secret_status, 403)
        self.assertEqual(traversal_status, 403)
        self.assertEqual((event_status, event_body), (200, raw_process_event.encode()))
        self.assertEqual(invalid_event_status, 403)

    def test_api_exposes_live_frame_and_stage_linked_human_feedback(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            run_dir = runs_root / "run-live"
            live_frames = run_dir / "live" / "observations"
            live_frames.mkdir(parents=True)
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "running"}),
                encoding="utf-8",
            )
            (run_dir / "process.jsonl").write_text(
                "".join(
                    json.dumps(record) + "\n"
                    for record in (
                        {"type": "observation", "event_index": 1, "action_step": 0},
                        {"type": "transition", "event_index": 2, "action_step": 1},
                        {
                            "type": "human_feedback",
                            "event_index": 3,
                            "after_step": 1,
                            "text": "the lower tile moved",
                            "channel": "human_coach",
                        },
                    )
                ),
                encoding="utf-8",
            )
            (live_frames / "step-00000.png").write_bytes(b"frame zero")
            (live_frames / "step-00001.png").write_bytes(b"frame one")

            status, body = _request(viewer, runs_root, "/api/run?run=run-live")
            data = json.loads(body)

        self.assertEqual(status, 200)
        transition = next(event for event in data["events"] if event["type"] == "transition")
        feedback = next(event for event in data["events"] if event["type"] == "human_feedback")
        self.assertIn("live/observations/step-00001.png", transition["after_image"])
        self.assertEqual(feedback["action_step"], 1)
        self.assertEqual(feedback["text"], "the lower tile moved")
        self.assertEqual(feedback["next_action"], 2)
        self.assertEqual(
            feedback["process_event"],
            "/process-event?run=run-live&event=3",
        )

    def test_api_bundles_feedback_with_before_and_after_trajectory_evidence(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            run_dir = runs_root / "run-steered"
            requests = run_dir / "request-payloads"
            responses = run_dir / "response-bodies"
            requests.mkdir(parents=True)
            responses.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "complete", "actions": 2}),
                encoding="utf-8",
            )
            records = [
                {
                    "type": "model_response",
                    "event_index": 2,
                    "action_step": 1,
                    "model_round": 1,
                    "response": {"reasoning_content": "continue the old theory"},
                },
                {
                    "type": "decision",
                    "event_index": 3,
                    "action_step": 1,
                    "action": 6,
                    "data": {"x": 4, "y": 4},
                    "reasoning": {"reason": "test old target"},
                },
                {
                    "type": "transition",
                    "event_index": 4,
                    "action_step": 1,
                    "changed_pixels": 1,
                    "levels_before": 0,
                    "levels_after": 0,
                },
                {
                    "type": "human_feedback",
                    "event_index": 5,
                    "after_step": 1,
                    "text": "break the loop; test blue",
                    "channel": "human_coach",
                },
                {
                    "type": "model_response",
                    "event_index": 6,
                    "action_step": 2,
                    "model_round": 1,
                    "response": {"reasoning_content": "inspect blue components"},
                },
                {
                    "type": "python_tool",
                    "event_index": 7,
                    "action_step": 2,
                    "code": "print('blue bounds')",
                    "output": json.dumps({"ok": True, "stdout": "x=60..63"}),
                },
                {
                    "type": "model_response",
                    "event_index": 8,
                    "action_step": 2,
                    "model_round": 2,
                    "response": {"content": "choose the discriminating probe"},
                },
                {
                    "type": "decision",
                    "event_index": 9,
                    "action_step": 2,
                    "action": 6,
                    "data": {"x": 60, "y": 32},
                    "reasoning": {"reason": "test blue control"},
                },
                {
                    "type": "transition",
                    "event_index": 10,
                    "action_step": 2,
                    "changed_pixels": 265,
                    "levels_before": 0,
                    "levels_after": 0,
                },
            ]
            (run_dir / "process.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            for event_index in (6, 8):
                (requests / f"{event_index:06d}.json").write_text("{}", encoding="utf-8")
                (responses / f"{event_index:06d}.raw").write_text("raw", encoding="utf-8")

            status, body = _request(viewer, runs_root, "/api/run?run=run-steered")
            data = json.loads(body)

        self.assertEqual(status, 200)
        self.assertEqual(len(data["steering_audit"]), 1)
        audit = data["steering_audit"][0]
        self.assertEqual(audit["feedback_event_index"], 5)
        self.assertEqual(audit["source_step"], 1)
        self.assertEqual(audit["next_action"], 2)
        self.assertEqual(
            audit["before_event_indices"],
            [2, 3, 4],
        )
        self.assertEqual(
            audit["after_event_indices"],
            [6, 7, 8, 9, 10],
        )
        self.assertEqual(audit["model_rounds"], 2)
        self.assertEqual(audit["python_probes"], 1)
        self.assertEqual(audit["decision_event_index"], 9)
        self.assertEqual(audit["transition_event_index"], 10)
        self.assertIsNone(audit["terminal_event_index"])
        self.assertNotIn("before_events", audit)
        self.assertLess(len(json.dumps(audit)), 500)
        events = {event["event_index"]: event for event in data["events"]}
        self.assertEqual(events[5]["text"], "break the loop; test blue")
        self.assertEqual(events[9]["data"], {"x": 60, "y": 32})
        self.assertEqual(events[10]["changed_pixels"], 265)
        self.assertEqual(
            events[6]["request_payload"],
            "/evidence?run=run-steered&file=request-payloads/000006.json",
        )
        self.assertEqual(
            events[6]["response_body"],
            "/evidence?run=run-steered&file=response-bodies/000006.raw",
        )
        self.assertEqual(
            events[5]["process_event"],
            "/process-event?run=run-steered&event=5",
        )

    def test_run_failed_overrides_stale_running_manifest_and_closes_final_audit(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            runs_root = Path(directory)
            run_dir = runs_root / "run-failed"
            run_dir.mkdir()
            (run_dir / "manifest.json").write_text(
                json.dumps({"status": "running", "actions": 1}),
                encoding="utf-8",
            )
            records = [
                {
                    "type": "human_feedback",
                    "event_index": 1,
                    "after_step": 1,
                    "text": "inspect all four blocks",
                },
                {
                    "type": "model_response",
                    "event_index": 2,
                    "action_step": 2,
                    "response": {"reasoning_content": "test another block"},
                },
                {
                    "type": "decision",
                    "event_index": 3,
                    "action_step": 2,
                    "action": 6,
                    "data": {"x": 1, "y": 35},
                },
                {
                    "type": "run_failed",
                    "event_index": 4,
                    "actions": 1,
                    "error": "environment step returned no frame",
                },
            ]
            (run_dir / "process.jsonl").write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )

            data = viewer.load_run(runs_root, "run-failed")

        self.assertEqual(data["summary"]["status"], "failed")
        self.assertEqual(data["summary"]["stop_reason"], "environment step returned no frame")
        failed = data["events"][-1]
        self.assertEqual((failed["type"], failed["action_step"]), ("run_failed", 2))
        audit = data["steering_audit"][0]
        self.assertEqual(audit["after_event_indices"], [2, 3, 4])
        self.assertIsNone(audit["transition_event_index"])
        self.assertEqual(audit["terminal_event_index"], 4)

    def test_live_jsonl_reader_ignores_only_an_incomplete_final_record(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "process.jsonl"
            path.write_text('{"type":"run_started"}\n{"type":', encoding="utf-8")

            records = viewer._read_jsonl(path, allow_incomplete_tail=True)

        self.assertEqual(records, [{"type": "run_started"}])

    def test_completed_jsonl_reader_rejects_an_incomplete_final_record(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "process.jsonl"
            path.write_text('{"type":"run_started"}\n{"type":', encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid JSONL"):
                viewer._read_jsonl(path)


if __name__ == "__main__":
    unittest.main()
