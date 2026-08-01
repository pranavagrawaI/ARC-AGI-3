from __future__ import annotations

import importlib.util
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest


REPO = Path(__file__).resolve().parents[3]
VIEWER = REPO / "experiments/agents/0002-nooa/output_viewer.py"


def _load_viewer():
    spec = importlib.util.spec_from_file_location("nooa_output_viewer", VIEWER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {VIEWER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _request(viewer, pairs_root: Path, path: str) -> tuple[int, bytes]:
    handler = viewer.Handler.__new__(viewer.Handler)
    handler.server = SimpleNamespace(pairs_root=pairs_root)
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


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def _candidate_pair(root: Path, name: str = "pair-a") -> Path:
    pair = root / name
    _write_json(pair / "summary.json", {"verdict": "interface_only"})
    _write_json(pair / "paired-run-spec.json", {"game": "ls20"})
    candidate = pair / "candidate"
    _write_json(
        candidate / "summary.json",
        {
            "arm": "0002-NOOA",
            "actions": 1,
            "levels_completed": 0,
            "state": "NOT_FINISHED",
            "stop_reason": "action_budget",
            "trace_complete": True,
        },
    )
    _write_jsonl(
        candidate / "trace.jsonl",
        [
            {
                "type": "observation",
                "step": 0,
                "pixels": [[[1]]],
                "available_actions": [1],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "full_reset": True,
            },
            {
                "type": "transition",
                "step": 1,
                "action": 1,
                "action_data": None,
                "observation": {
                    "pixels": [[[2]]],
                    "available_actions": [1],
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                    "full_reset": False,
                },
            },
        ],
    )
    _write_jsonl(
        candidate / "worker/process.jsonl",
        [
            {
                "kind": "action_request",
                "action_point_id": 0,
                "host_observation": {"path": "host-observations/step-00000.json"},
                "request": {"legal_actions": [1]},
            },
            {
                "kind": "model_crossing",
                "action_point_id": 0,
                "worker_decision": {"action": 1, "data": None, "reason": "probe"},
                "audit": {
                    "model_calls": [
                        {
                            "messages": [{"role": "user", "content": "public input"}],
                            "tools": [{"name": "execute_python"}],
                            "outcome": {
                                "kind": "success",
                                "response": {
                                    "content": "provider-returned reasoning",
                                    "tool_calls": [
                                        {
                                            "name": "execute_python",
                                            "arguments": {"code": "print('ok')"},
                                        }
                                    ],
                                },
                            },
                        }
                    ]
                },
            },
            {
                "kind": "decision_validated",
                "action_point_id": 0,
                "decision": {
                    "action": 1,
                    "data": None,
                    "reasoning": {"reason": "probe"},
                },
                "workspace_commitment": "abc123",
            },
            {
                "kind": "transition_recorded",
                "action_point_id": 0,
                "transition": {
                    "changed_pixels": 1,
                    "levels_before": 0,
                    "levels_after": 0,
                    "state_before": "NOT_FINISHED",
                    "state_after": "NOT_FINISHED",
                },
            },
        ],
    )
    image = candidate / "worker/workspace/observations/step-00000.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"not-a-real-png")
    _write_json(candidate / "worker/host-observations/step-00000.json", {"step": 0})
    control = pair / "control"
    _write_json(
        control / "summary.json",
        {
            "arm": "0002-R",
            "actions": 1,
            "levels_completed": 0,
            "state": "NOT_FINISHED",
            "stop_reason": "action_budget",
            "trace_complete": True,
        },
    )
    _write_jsonl(
        control / "trace.jsonl",
        [
            {
                "type": "observation",
                "step": 0,
                "available_actions": [1],
                "levels_completed": 0,
                "state": "NOT_FINISHED",
                "full_reset": True,
            },
            {
                "type": "transition",
                "step": 1,
                "action": 2,
                "action_data": None,
                "observation": {
                    "available_actions": [1],
                    "levels_completed": 0,
                    "state": "NOT_FINISHED",
                    "full_reset": False,
                },
            },
        ],
    )
    _write_jsonl(
        control / "process.jsonl",
        [
            {
                "type": "observation",
                "action_step": 0,
                "metadata": {"available_actions": [1]},
            },
            {
                "type": "model_response",
                "action_step": 1,
                "model_round": 1,
                "request_messages": [{"role": "user", "content": "public input"}],
                "request_payload_sha256": "request-sha",
                "request_payload_archive": "request-payloads/000003.json",
                "response_body_archive": "response-bodies/000003.raw",
                "response": {"content": "control provider text", "tool_calls": []},
                "response_body": {"output": []},
            },
            {
                "type": "python_tool",
                "action_step": 1,
                "code": "print('probe')",
                "output": "ok",
            },
            {
                "type": "workspace_change",
                "action_step": 1,
                "changes": [{"path": "G9_CAPABILITY_PROBE.json"}],
            },
            {
                "type": "decision",
                "action_step": 1,
                "action": 2,
                "data": None,
                "reasoning": {"reason": "control decision"},
            },
            {
                "type": "transition",
                "step": 1,
                "action": 2,
                "changed_pixels": 1,
            },
        ],
    )
    image = control / "workspace/observations/step-00000.png"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"not-a-real-control-png")
    _write_json(control / "request-payloads/000003.json", {"request": True})
    (control / "response-bodies").mkdir(parents=True, exist_ok=True)
    (control / "response-bodies/000003.raw").write_bytes(b"response")
    return pair


class NooaOutputViewerTests(unittest.TestCase):
    def test_discovers_pairs_and_projects_one_nooa_action_point(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            pairs_root = Path(directory)
            _candidate_pair(pairs_root)

            pair_names = viewer.pair_names(pairs_root)
            data = viewer.load_arm(pairs_root, "pair-a", "candidate")

        self.assertEqual(pair_names, ["pair-a"])
        self.assertEqual(data["arm"], "0002-NOOA")
        self.assertEqual(data["summary"]["actions"], 1)
        self.assertEqual(len(data["steps"]), 2)
        action_point = data["steps"][0]
        self.assertEqual(action_point["action_point_id"], 0)
        self.assertEqual(action_point["model_calls"][0]["content"], "provider-returned reasoning")
        self.assertEqual(action_point["decision"]["action"], 1)
        self.assertEqual(action_point["transition"]["changed_pixels"], 1)
        self.assertEqual(action_point["workspace_commitment"], "abc123")

    def test_rendered_page_labels_provider_text_and_probe_scope(self) -> None:
        viewer = _load_viewer()
        page = viewer.render_page(
            data={"pair_name": "pair-a", "arm": "0002-NOOA", "summary": {}, "steps": []},
            pair_names=["pair-a"],
        ).decode("utf-8")

        self.assertIn("NOOA Evidence Viewer", page)
        self.assertIn("Provider-returned reasoning", page)
        self.assertIn("not gameplay competence evidence", page)
        self.assertIn('aria-label="Replay position"', page)

    def test_control_arm_normalizes_one_based_actions_to_their_input_frame(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            pairs_root = Path(directory)
            _candidate_pair(pairs_root)
            data = viewer.load_arm(pairs_root, "pair-a", "control")

        self.assertEqual(data["arm"], "0002-R")
        self.assertEqual(len(data["steps"]), 2)
        stage = data["steps"][0]
        self.assertEqual(stage["frame"]["step"], 0)
        self.assertEqual(stage["decision"]["action"], 2)
        self.assertEqual(stage["transition"]["changed_pixels"], 1)
        self.assertEqual(stage["model_calls"][0]["content"], "control provider text")
        self.assertEqual(stage["python_calls"][0]["code"], "print('probe')")
        self.assertEqual(stage["workspace_changes"][0]["changes"][0]["path"], "G9_CAPABILITY_PROBE.json")
        labels = [link["label"] for link in stage["evidence_links"]]
        self.assertIn("Exact request payload", labels)
        self.assertIn("Exact response body", labels)

    def test_server_serves_only_allowlisted_artifacts_inside_the_selected_arm(self) -> None:
        viewer = _load_viewer()
        with TemporaryDirectory() as directory:
            pairs_root = Path(directory)
            _candidate_pair(pairs_root)

            status, body = _request(
                viewer,
                pairs_root,
                "/asset?pair=pair-a&arm=candidate&file=worker/workspace/observations/step-00000.png",
            )
            blocked_status, _ = _request(
                viewer,
                pairs_root,
                "/asset?pair=pair-a&arm=candidate&file=../paired-run-spec.json",
            )
            pair_status, pair_body = _request(
                viewer,
                pairs_root,
                "/pair-artifact?pair=pair-a&file=paired-run-spec.json",
            )

        self.assertEqual(status, 200)
        self.assertEqual(body, b"not-a-real-png")
        self.assertEqual(blocked_status, 403)
        self.assertEqual(pair_status, 200)
        self.assertEqual(json.loads(pair_body), {"game": "ls20"})

    def test_viewer_refuses_non_loopback_binding(self) -> None:
        viewer = _load_viewer()

        self.assertEqual(viewer._loopback_host("127.0.0.1"), "127.0.0.1")
        with self.assertRaises(viewer.argparse.ArgumentTypeError):
            viewer._loopback_host("0.0.0.0")


if __name__ == "__main__":
    unittest.main()
