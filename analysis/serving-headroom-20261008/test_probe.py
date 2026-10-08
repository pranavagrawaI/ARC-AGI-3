import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "mtp-depth-20261006"))
sys.path.insert(0, str(HERE))
import probe
import build


class ProbeTests(unittest.TestCase):
    def samples(self, running=10):
        return [{"t": float(i), "running": running, "queue": 0,
                 "prompt_tokens": 100, "gen_tokens": (i // 3) * 300} for i in range(30)]

    def test_bursty_counters_keep_elapsed_denominator(self):
        got = probe.plateau(self.samples(), 10)
        self.assertTrue(got["valid"])
        self.assertEqual(got["seconds"], 24)
        self.assertEqual(got["U"], 100)

    def test_prefill_queue_and_counter_reset_break_plateau(self):
        samples = self.samples()
        for row in samples: row["prompt_tokens"] = row["t"] * 10
        self.assertFalse(probe.plateau(samples, 10)["valid"])
        samples = self.samples()
        for row in samples: row["queue"] = 1
        self.assertFalse(probe.plateau(samples, 10)["valid"])
        samples = self.samples()
        for row in samples: row["gen_tokens"] = 10000 - row["t"]
        self.assertFalse(probe.plateau(samples, 10)["valid"])

    def test_wrong_occupancy_cannot_count(self):
        self.assertFalse(probe.plateau(self.samples(9), 10)["valid"])
        samples = self.samples()
        for row in samples: row["pending"] = 9
        self.assertFalse(probe.plateau(samples, 10)["valid"])

    def test_wire_execution_and_failure_preserve_raw_receipts(self):
        class Handler(BaseHTTPRequestHandler):
            fail = False
            def log_message(self, *args): pass
            def do_GET(self):
                raw = b'sglang:num_running_reqs 0\nsglang:num_queue_reqs 0\nsglang:generation_tokens_total 32\nsglang:prompt_tokens_total 100\n'
                self.send_response(200); self.end_headers(); self.wfile.write(raw)
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert body['ignore_eos'] is True and body['seed'] == 20261008
                assert self.path == '/v1/chat/completions'
                self.send_response(500 if self.fail else 200); self.end_headers()
                self.wfile.write(json.dumps({'usage': {'prompt_tokens': 100, 'completion_tokens': body['max_tokens'], 'prompt_tokens_details': {'cached_tokens': 95}}}).encode())
        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
        job = {'game': 'fixture', 'body': {'ignore_eos': True, 'seed': 20261008}}
        try:
            with tempfile.TemporaryDirectory() as tmp, patch.object(probe, 'PORT', server.server_port):
                out = Path(tmp)
                result = probe.batch('GOOD', [job], out, max_tokens=32)
                self.assertEqual(result['status'], 'complete')
                self.assertTrue(result['cache_ok'])
                self.assertFalse(result['plateau']['valid'])  # no fictitious throughput
                self.assertTrue((out/'GOOD/fixture-response.json').is_file())
                Handler.fail = True
                with self.assertRaisesRegex(RuntimeError, 'HTTP 500'):
                    probe.batch('BAD', [job], out, max_tokens=32)
                failed = json.loads((out/'BAD/summary.json').read_text())
                self.assertEqual(failed['status'], 'failed')
                events = (out/'BAD/events.jsonl').read_text()
                self.assertIn('request_error', events)
                self.assertTrue((out/'BAD/fixture-request.json').is_file())
        finally:
            server.shutdown(); server.server_close()

    def test_profile_native_contract(self):
        calls = []
        class Events:
            def emit(self, *args, **kwargs): calls.append((args, kwargs))
        with patch.object(probe, "http", return_value=b'{}') as http:
            probe.profile_block("PROFILE10", Path("/kaggle/working/headroom"), Events())
            path, body = http.call_args.args
            self.assertEqual(path, "/start_profile")
            self.assertEqual(body["num_steps"], 64)
            self.assertEqual(body["activities"], ["CPU", "GPU"])
            self.assertFalse(body["with_stack"])

    def test_actual_notebook_contract_and_parent_diff(self):
        build.build()
        nb = json.loads((HERE / "kernel" / (build.SLUG + ".ipynb")).read_text())
        parent = json.loads((build.PARENT / "kernel/mtp-depth-sweep-retry1-20261007.ipynb").read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code": compile("".join(cell["source"]), str(i), "exec")
        self.assertEqual(nb["cells"][3]["source"], parent["cells"][3]["source"])
        launcher = "".join(nb["cells"][4]["source"])
        self.assertEqual(launcher.replace("MAXREQ=16,", "MAXREQ=10,").replace("CUDAGRAPH_MAXBS=16,", "CUDAGRAPH_MAXBS=10,"), "".join(parent["cells"][4]["source"]))
        self.assertIn("SESSION_DEADLINE = time.monotonic() + 6600", "".join(nb["cells"][2]["source"]))
        self.assertIn((HERE / "probe.py").read_text(), "".join(nb["cells"][-1]["source"]))
        m = json.loads((HERE / "kernel/kernel-metadata.json").read_text())
        self.assertEqual(m["dataset_sources"], json.loads((build.PARENT / "kernel/kernel-metadata.json").read_text())["dataset_sources"])
        self.assertEqual(m["competition_sources"], ["arc-prize-2026-arc-agi-3"])


if __name__ == "__main__": unittest.main()
