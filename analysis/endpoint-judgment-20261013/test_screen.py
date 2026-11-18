"""Offline behavioral checks. No external inference or model key is used."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import screen

SNAP = {"level": 1, "state": "NOT_FINISHED", "frame_hash": "fixture", "actions": 0, "action_hash": "empty"}
PAYLOAD = {"model": "alias", "messages": [{"role": "user", "content": "state"}], "max_tokens": 1000,
           "tools": [{"type": "function", "function": {"name": "python"}}], "temperature": .7}


class FakeClock:
    def __init__(self):
        self.now, self.waits = 1791849600., []
    def time(self):
        return self.now
    def sleep(self, delay):
        self.waits.append(delay)
        self.now += delay


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=screen.HERE / "evidence")
        self.root = Path(self.tmp.name) / "start"
        self.root.mkdir()
        self.cfg = screen.config(screen.parser().parse_args(["--dry", "--token-budget", "150", "--max-rpm", "2", "--daily-cap", "3"]))
        self.clock = FakeClock()
        self.snapshot = dict(SNAP)
        self.sent = []
    def tearDown(self):
        self.tmp.cleanup()
    def transport(self, cfg=None, sender=None):
        return screen.Transport(self.root, cfg or self.cfg, lambda: dict(self.snapshot), script=[{"action": "UP"}],
                                sender=sender or self.sender, clock=self.clock.time, sleep=self.clock.sleep)
    def sender(self, url, payload, key, timeout):
        self.sent.append(payload)
        return 200, {"x-provider": "fixture"}, screen.mock_reply(len(self.sent), [{"action": "UP"}], 64)
    def test_budget_and_resume_do_not_double_charge(self):
        t = self.transport()
        t.receive(PAYLOAD)
        self.snapshot["actions"] = 1
        t.receive(PAYLOAD)
        t.receive(PAYLOAD)
        self.assertEqual(t.totals()["generated_tokens"], 150)
        with self.assertRaises(screen.Halt) as stop:
            t.receive(PAYLOAD)
        self.assertEqual(stop.exception.status, "token_budget")
        self.snapshot = dict(SNAP)
        resumed = self.transport()
        resumed.receive(PAYLOAD)
        self.snapshot["actions"] = 1
        resumed.receive(PAYLOAD)
        resumed.receive(PAYLOAD)
        self.assertEqual(resumed.totals()["requests"], 3)
        self.assertEqual(resumed.totals()["generated_tokens"], 150)
    def test_replay_state_mismatch_blocks_new_request(self):
        self.transport().receive(PAYLOAD)
        self.snapshot["frame_hash"] = "changed"
        with self.assertRaises(screen.Halt) as stop:
            self.transport().receive(PAYLOAD)
        self.assertEqual(stop.exception.status, "replay_mismatch")
    def test_replay_tool_output_mismatch_blocks_even_with_same_board(self):
        self.transport().receive(PAYLOAD)
        changed = copy.deepcopy(PAYLOAD)
        changed["messages"][0]["content"] = "different tool observation"
        with self.assertRaises(screen.Halt) as stop:
            self.transport().receive(changed)
        self.assertEqual(stop.exception.status, "replay_mismatch")
    def test_pending_request_never_retried(self):
        cfg = {**self.cfg, "dry": False}
        def failed(*args):
            self.sent.append(1)
            raise TimeoutError()
        with patch.dict(os.environ, {"ENDPOINT_API_KEY": "offline-fixture"}):
            for _ in range(2):
                with self.assertRaises(screen.Halt) as stop:
                    self.transport(cfg, failed).receive(PAYLOAD)
                self.assertEqual(stop.exception.status, "pending_unknown")
        self.assertEqual(len(self.sent), 1)
        self.assertTrue((self.root / "pending.json").exists())
    def test_429_backoff_counts_attempts_and_daily_cap(self):
        cfg = {**self.cfg, "dry": False, "base_url": "http://offline.test"}
        def throttled(url, payload, key, timeout):
            self.sent.append(payload)
            if len(self.sent) == 1:
                return 429, {"Retry-After": "7"}, {"error": "rate limited"}
            return 200, {}, screen.mock_reply(0, [{"action": "UP"}], 64)
        with patch.dict(os.environ, {"ENDPOINT_API_KEY": "offline-fixture"}):
            t = self.transport(cfg, throttled)
            t.receive(PAYLOAD)
            t.receive(PAYLOAD)
            with self.assertRaises(screen.Halt) as stop:
                t.receive(PAYLOAD)
        self.assertEqual(stop.exception.status, "daily_cap")
        self.assertEqual(len(self.sent), 3)
        self.assertEqual(self.clock.waits[0], 7)
        self.assertGreater(sum(self.clock.waits), 60)
    def test_utc_daily_rollover(self):
        q = screen.Quota(self.root / "quota.json", 0, 1, self.clock.time, self.clock.sleep)
        q.reserve()
        with self.assertRaises(screen.Halt):
            q.reserve()
        self.clock.now += 86400
        q.reserve()
        self.assertEqual(len(screen.load(q.path)["sent"]), 2)
    def test_unknown_usage_preserves_response_and_stops(self):
        cfg = {**self.cfg, "dry": False}
        def missing(*args):
            r = screen.mock_reply(0, None, 50)
            r.pop("usage")
            return 200, {}, r
        with patch.dict(os.environ, {"ENDPOINT_API_KEY": "offline-fixture"}):
            with self.assertRaises(screen.Halt) as stop:
                self.transport(cfg, missing).receive(PAYLOAD)
        self.assertEqual(stop.exception.status, "usage_unknown")
        self.assertFalse(self.transport().totals()["usage_known"])
        self.assertTrue((self.root / "receipts/000000.json.gz").exists())
        with self.assertRaises(screen.Halt) as repeated:
            self.transport().receive(PAYLOAD)
        self.assertEqual(repeated.exception.status, "usage_unknown")
    def test_provider_cannot_overrun_cap_silently(self):
        cfg = {**self.cfg, "dry": False, "token_budget": 20}
        with patch.dict(os.environ, {"ENDPOINT_API_KEY": "offline-fixture"}):
            with self.assertRaises(screen.Halt) as stop:
                self.transport(cfg).receive(PAYLOAD)
        self.assertEqual(stop.exception.status, "budget_violation")
        self.assertEqual(self.sent[0]["max_tokens"], 20)
        self.assertEqual(self.transport().totals()["generated_tokens"], 64)
        with self.assertRaises(screen.Halt) as repeated:
            self.transport(cfg).receive(PAYLOAD)
        self.assertEqual(repeated.exception.status, "budget_violation")
    def test_fatal_http_is_not_retried(self):
        cfg = {**self.cfg, "dry": False}
        def fatal(*args):
            self.sent.append(1)
            return 400, {}, {"error": "tools unsupported"}
        with patch.dict(os.environ, {"ENDPOINT_API_KEY": "offline-fixture"}):
            with self.assertRaises(screen.Halt) as stop:
                self.transport(cfg, fatal).receive(PAYLOAD)
        self.assertEqual(stop.exception.status, "endpoint_error")
        self.assertEqual(len(self.sent), 1)
        self.assertFalse((self.root / "pending.json").exists())
    def test_wire_adapter_preserves_tool_schema_and_bounds_extras(self):
        cfg = {**self.cfg, "modality": "text", "max_tokens_field": "max_completion_tokens", "extra_body": {"max_tokens": 99999}}
        p = copy.deepcopy(PAYLOAD)
        p["messages"] = [{"role": "user", "content": [{"type": "text", "text": "ascii board"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,fixture"}}], "reasoning_content": "old"}]
        p["top_k"] = 20
        w = screen.wire_payload(p, cfg, 17)
        self.assertEqual(w["max_completion_tokens"], 17)
        self.assertNotIn("max_tokens", w)
        self.assertEqual(w["tools"], PAYLOAD["tools"])
        self.assertEqual(w["messages"], [{"role": "user", "content": "ascii board"}])
        self.assertNotIn("top_k", w)
    def test_unknown_quantisation_is_not_inferred(self):
        i = screen.identity({"model": "returned-id"}, {"x-provider": "reported"}, "configured")
        self.assertIsNone(i["quantisation"])
        self.assertEqual(i["returned_model"], "returned-id")
    def test_live_requires_zero_cost_confirmation_and_free_model(self):
        with self.assertRaises(ValueError):
            screen.config(screen.parser().parse_args([]))
        with self.assertRaises(ValueError):
            screen.config(screen.parser().parse_args(["--zero-cost-confirmed", "--base-url", "https://openrouter.ai/api/v1", "--model", "paid/model"]))
    def test_output_cannot_escape_task(self):
        with self.assertRaises(ValueError):
            screen.inside(Path("/tmp/outside-task"))
    def test_notebook_environment_is_mechanically_preserved(self):
        e = screen.notebook_env()
        self.assertEqual(e["LOCAL_ANALYZER_TEMPERATURE"], "0.7")
        self.assertEqual(e["LOCAL_ANALYZER_CONTEXT_WINDOW"], "131072")
        self.assertEqual(e["ARC3_STALE_STATE_BLOCK"], "1")
        self.assertEqual(e["ARC3_PERSISTENT_FUNCTIONS_SCOPE"], "game")
    def test_standard_api_key_environment_survives_harness_wiring(self):
        cfg = {**self.cfg, "api_key_env": "OPENAI_API_KEY"}
        with patch.dict(os.environ, {"OPENAI_API_KEY": "offline-fixture"}):
            env = screen.child_env(self.root, cfg)
        self.assertEqual(env["OPENAI_API_KEY"], "offline-fixture")
    def test_http_adapter_sends_exact_endpoint_model_body_and_key(self):
        class Response:
            status = 200
            headers = {"x-request-id": "offline-id"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return b'{"model":"exact-model","choices":[{"message":{"role":"assistant","content":"ok"}}],"usage":{"completion_tokens":2}}'
        seen = []
        def open_(req, timeout):
            seen.append((req, timeout))
            return Response()
        with patch("urllib.request.urlopen", open_):
            status, headers, response = screen.send_http("https://offline.test/v1/chat/completions", {"model": "exact-model"}, "fixture-key", 3)
        req, timeout = seen[0]
        self.assertEqual(req.full_url, "https://offline.test/v1/chat/completions")
        self.assertEqual(json.loads(req.data), {"model": "exact-model"})
        self.assertEqual(req.get_header("Authorization"), "Bearer fixture-key")
        self.assertEqual((status, timeout, response["model"]), (200, 3, "exact-model"))


class EngineTests(unittest.TestCase):
    def test_real_harness_final_level_reset_and_idempotence(self):
        with tempfile.TemporaryDirectory(dir=screen.HERE / "evidence") as tmp:
            cfg = screen.config(screen.parser().parse_args(["--dry", "--output", tmp, "--keys", "tn36-L1", "lp85-L7", "bp35-L1", "--token-budget", "4096"]))
            root = Path(tmp)
            summary = screen.run(copy.deepcopy(cfg))
            self.assertEqual(summary["completed"], 3)
            self.assertEqual(summary["solves"], 2)
            self.assertEqual(summary["per_start"][0]["actions"], 15)
            self.assertEqual(summary["per_start"][1]["status"], "solved")
            self.assertEqual(summary["per_start"][2]["status"], "token_budget")
            receipts = {str(p): screen.sha(p.read_bytes()) for p in root.rglob("receipts/*.json.gz")}
            again = screen.run(copy.deepcopy(cfg))
            self.assertEqual(again["generated_tokens"], summary["generated_tokens"])
            self.assertEqual(receipts, {str(p): screen.sha(p.read_bytes()) for p in root.rglob("receipts/*.json.gz")})
            # A killed worker after three received responses reconstructs the
            # native actor, executes cached tools locally, then continues.
            attempt = root / "tn36-L1"
            target_receipt = screen.load(attempt / "receipts/000003.json.gz")
            boundary = target_receipt["snapshot"]
            for p in sorted((attempt / "receipts").glob("*.json.gz"))[3:]:
                p.unlink()
            screen.atomic(attempt / "checkpoint.json", {"next_index": 3, "snapshot": boundary, "context_hash": target_receipt["context_hash"]})
            (attempt / "result.json").unlink()
            cached = {str(p): screen.sha(p.read_bytes()) for p in (attempt / "receipts").glob("*.json.gz")}
            resumed = screen.run(copy.deepcopy(cfg))
            self.assertEqual(resumed["solves"], 2)
            self.assertEqual(resumed["generated_tokens"], summary["generated_tokens"])
            self.assertEqual(cached, {p: screen.sha(Path(p).read_bytes()) for p in cached})


if __name__ == "__main__":
    (screen.HERE / "evidence").mkdir(exist_ok=True)
    unittest.main(verbosity=2)
