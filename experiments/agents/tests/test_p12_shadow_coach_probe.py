from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import tempfile
import threading
from typing import Any
import unittest


ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / (
    "experiments/agents/0006-capability-environment/iterations/"
    "0003-human-teacher-shadow-coach/probe.py"
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


class ShadowCoachProbeCliTests(unittest.TestCase):
    def test_prepare_only_freezes_no_note_and_human_note_without_network(self) -> None:
        observation = {
            "step": 26,
            "environment_observation": {
                "current_metadata": {
                    "available_actions": [1, 2, 3, 4, 5, 6],
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                }
            },
        }
        source = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "actor contract"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": json.dumps(observation)},
                        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                    ],
                },
            ],
            "tools": [{"type": "function", "function": {"name": "python"}}],
            "tool_choice": "auto",
            "temperature": 1.0,
            "seed": 0,
        }
        note = {
            "schema": "0006-human-teacher-note-v1",
            "source_after_step": 26,
            "text": (
                "have we tried everything to come to a final conclusion, or do we have "
                "more of the world we don't understand?"
            ),
            "word_count": 21,
            "author": "human",
            "game_specific_content": False,
            "role": "diagnostic_teacher",
            "admissible_unsupervised_contestant": False,
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            note_path = root / "human-note.json"
            output = root / "output"
            source_path.write_bytes(_canonical(source))
            note_path.write_text(json.dumps(note, indent=2) + "\n")
            source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
            note_sha = hashlib.sha256(note_path.read_bytes()).hexdigest()

            completed = subprocess.run(
                [
                    "uv",
                    "run",
                    "python3",
                    str(PROBE),
                    "--prepare-only",
                    "--source-request",
                    str(source_path),
                    "--source-sha256",
                    source_sha,
                    "--human-note",
                    str(note_path),
                    "--human-note-sha256",
                    note_sha,
                    "--output-dir",
                    str(output),
                    "--provider",
                    "deepinfra",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            plan = json.loads((output / "plan.json").read_text())
            self.assertEqual(plan["status"], "prepared")
            self.assertEqual(plan["source_request_sha256"], source_sha)
            self.assertEqual(plan["human_note_sha256"], note_sha)
            self.assertEqual(plan["call_ceiling"], 4)
            self.assertEqual(plan["provider"], {
                "order": ["deepinfra"],
                "only": ["deepinfra"],
                "allow_fallbacks": False,
            })
            self.assertEqual([stage["name"] for stage in plan["stages"]], [
                "no_note_actor", "human_note_actor", "shadow_coach", "shadow_note_actor"
            ])
            self.assertEqual(plan["stages"][-1]["status"], "blocked_on_shadow_note")

            no_note = json.loads((output / "requests" / "no_note_actor.json").read_text())
            human = json.loads((output / "requests" / "human_note_actor.json").read_text())
            self.assertNotIn(
                "human_coach",
                json.loads(no_note["messages"][-1]["content"][0]["text"]),
            )
            human_observation = json.loads(human["messages"][-1]["content"][0]["text"])
            self.assertEqual(human_observation["human_coach"]["notes"], [{
                "after_step": 26,
                "text": note["text"],
                "channel": "human_coach",
            }])
            self.assertEqual(no_note["provider"], plan["provider"])
            self.assertEqual(human["provider"], plan["provider"])
            self.assertFalse((output / "responses").exists())

    def test_prepare_only_rejects_a_changed_source_before_writing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.json"
            note_path = root / "note.json"
            output = root / "output"
            source_path.write_text('{"model":"changed","messages":[]}')
            note_path.write_text("{}")
            completed = subprocess.run(
                [
                    "uv", "run", "python3", str(PROBE), "--prepare-only",
                    "--source-request", str(source_path),
                    "--source-sha256", "0" * 64,
                    "--human-note", str(note_path),
                    "--human-note-sha256", hashlib.sha256(note_path.read_bytes()).hexdigest(),
                    "--output-dir", str(output),
                    "--provider", "deepinfra",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("source request SHA-256 mismatch", completed.stderr)
            self.assertFalse(output.exists())

    def test_execute_persists_four_hash_linked_calls_and_labels_shadow_advice(self) -> None:
        observation = {
            "step": 26,
            "environment_observation": {"current_metadata": {
                "available_actions": [1, 2, 3, 4, 5, 6],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
            }},
        }
        source = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "actor contract"},
                {"role": "user", "content": [
                    {"type": "text", "text": json.dumps(observation)},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                ]},
            ],
            "tools": [{"type": "function", "function": {"name": "python"}}],
            "tool_choice": "auto",
            "temperature": 1.0,
            "seed": 0,
        }
        human_text = (
            "have we tried everything to come to a final conclusion, or do we have "
            "more of the world we don't understand?"
        )
        note = {
            "schema": "0006-human-teacher-note-v1",
            "source_after_step": 26,
            "text": human_text,
            "word_count": 21,
            "author": "human",
            "game_specific_content": False,
            "role": "diagnostic_teacher",
            "admissible_unsupervised_contestant": False,
        }
        received: list[dict[str, Any]] = []
        responses = [
            {"content": '{"action":6,"data":{"x":1,"y":1},"reason":"N"}'},
            {"content": '{"action":1,"data":null,"reason":"H"}'},
            {"content": '{"question":"Which assumptions remain untested or uncertain?"}'},
            {"content": '{"action":2,"data":null,"reason":"S"}'},
        ]

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib hook
                body = self.rfile.read(int(self.headers["Content-Length"]))
                received.append(json.loads(body))
                message = responses[len(received) - 1]
                response = {
                    "id": f"fake-{len(received)}",
                    "model": "test-model",
                    "choices": [{"index": 0, "finish_reason": "stop", "message": message}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": 0.001},
                }
                encoded = json.dumps(response, separators=(",", ":")).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_path = root / "source.json"
                note_path = root / "human-note.json"
                output = root / "output"
                source_path.write_bytes(_canonical(source))
                note_path.write_text(json.dumps(note, indent=2) + "\n")
                completed = subprocess.run(
                    [
                        "uv", "run", "python3", str(PROBE), "--execute",
                        "--source-request", str(source_path),
                        "--source-sha256", hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        "--human-note", str(note_path),
                        "--human-note-sha256", hashlib.sha256(note_path.read_bytes()).hexdigest(),
                        "--output-dir", str(output),
                        "--provider", "deepinfra",
                        "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                        "--api-key", "fake-key",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)
                self.assertEqual(len(received), 4)
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["status"], "completed")
                self.assertEqual(result["calls_attempted"], 4)
                self.assertEqual(result["shadow_question"],
                                 "Which assumptions remain untested or uncertain?")
                self.assertEqual([stage["status"] for stage in result["stages"]],
                                 ["completed"] * 4)
                for stage in result["stages"]:
                    request = output / stage["request_archive"]
                    response = output / stage["response_archive"]
                    self.assertEqual(stage["request_sha256"],
                                     hashlib.sha256(request.read_bytes()).hexdigest())
                    self.assertEqual(stage["response_sha256"],
                                     hashlib.sha256(response.read_bytes()).hexdigest())
                shadow_observation = json.loads(
                    received[3]["messages"][-1]["content"][0]["text"]
                )
                self.assertNotIn("human_coach", shadow_observation)
                self.assertEqual(shadow_observation["shadow_coach"]["notes"], [{
                    "after_step": 26,
                    "text": "Which assumptions remain untested or uncertain?",
                    "channel": "autonomous_shadow_coach",
                }])
                self.assertNotIn("tools", received[2])
                self.assertEqual(received[2]["provider"], received[0]["provider"])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_execute_closes_a_failed_transport_attempt_in_result(self) -> None:
        source = {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "actor contract"},
                {"role": "user", "content": [{
                    "type": "text", "text": json.dumps({"step": 26})
                }]},
            ],
            "temperature": 1.0,
        }
        note = {
            "schema": "0006-human-teacher-note-v1",
            "source_after_step": 26,
            "text": "What remains unknown?",
            "word_count": 3,
            "author": "human",
        }
        received = 0

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802 - stdlib hook
                nonlocal received
                self.rfile.read(int(self.headers["Content-Length"]))
                received += 1
                if received == 1:
                    body = json.dumps({
                        "id": "ok-1",
                        "choices": [{"message": {"content": "{}"}}],
                        "usage": {},
                    }).encode()
                    self.send_response(200)
                else:
                    body = b'{"error":"forced failure"}'
                    self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format: str, *args: object) -> None:
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_path = root / "source.json"
                note_path = root / "note.json"
                output = root / "output"
                source_path.write_bytes(_canonical(source))
                note_path.write_text(json.dumps(note) + "\n")
                completed = subprocess.run(
                    [
                        "uv", "run", "python3", str(PROBE), "--execute",
                        "--source-request", str(source_path),
                        "--source-sha256", hashlib.sha256(source_path.read_bytes()).hexdigest(),
                        "--human-note", str(note_path),
                        "--human-note-sha256", hashlib.sha256(note_path.read_bytes()).hexdigest(),
                        "--output-dir", str(output), "--provider", "deepinfra",
                        "--base-url", f"http://127.0.0.1:{server.server_port}/v1",
                        "--api-key", "fake-key",
                    ],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(completed.returncode, 2)
                result = json.loads((output / "result.json").read_text())
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["calls_attempted"], 2)
                self.assertEqual([stage["status"] for stage in result["stages"]],
                                 ["completed", "failed"])
                failed = result["stages"][1]
                self.assertEqual(failed["name"], "human_note_actor")
                self.assertEqual(failed["http_status"], 500)
                self.assertEqual(
                    failed["request_sha256"],
                    hashlib.sha256((output / failed["request_archive"]).read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    failed["response_sha256"],
                    hashlib.sha256((output / failed["response_archive"]).read_bytes()).hexdigest(),
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
