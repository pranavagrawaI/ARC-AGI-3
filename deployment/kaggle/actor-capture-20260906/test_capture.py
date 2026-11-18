"""Local HTTP path tests; zero GPU, no external model calls."""

import ast
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from build import build, HERE, BASE


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        notebook = json.loads(build()["capture.ipynb"])
        assets = "".join(notebook["cells"][2]["source"])
        constants = [
            n.value
            for n in ast.walk(ast.parse(assets))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        ]
        data = (HERE / "capture_proxy.py").read_bytes()
        self.assertIn(base64.b64encode(data).decode(), constants)
        path = Path(self.temp.name) / "packaged_proxy.py"
        path.write_bytes(data)
        spec = importlib.util.spec_from_file_location("packaged_proxy", path)
        proxy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proxy)
        self.received = []
        received = self.received

        class Upstream(BaseHTTPRequestHandler):
            def do_GET(self):
                answer = b'{"sampling_defaults":"model"}'
                self.send_response(200)
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

            def log_message(self, *args):
                pass

            def do_POST(self):
                received.append(self.rfile.read(int(self.headers["Content-Length"])))
                answer = (
                    b'{"id":"completion-1","choices":[],"usage":{"prompt_tokens":100}}'
                )
                self.send_response(200)
                self.send_header("Content-Length", str(len(answer)))
                self.end_headers()
                self.wfile.write(answer)

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.thread = threading.Thread(target=self.upstream.serve_forever, daemon=True)
        self.thread.start()
        self.output = Path(self.temp.name) / "capture"
        self.proxy = proxy.CaptureServer(
            ("127.0.0.1", 0), self.upstream.server_address, self.output, max_requests=2
        )
        self.pt = threading.Thread(target=self.proxy.serve_forever, daemon=True)
        self.pt.start()

    def tearDown(self):
        self.proxy.shutdown()
        self.proxy.server_close()
        self.pt.join()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.thread.join()
        self.temp.cleanup()

    def post(self, body):
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.proxy.server_port}/v1/chat/completions",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer never-store-this",
            },
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            return response.read()

    def test_exact_bytes_and_retry_identity(self):
        body = b'{ "messages": [{"role":"user", "content":"test"}], "model":"local", "temperature":0, "tools":[] }'
        answer = self.post(body)
        self.assertEqual(self.post(body), answer)
        self.assertEqual(self.received, [body, body])
        for i in [0, 1]:
            self.assertEqual((self.output / f"{i:04d}-request.json").read_bytes(), body)
            self.assertEqual(
                (self.output / f"{i:04d}-response.json").read_bytes(), answer
            )
        self.assertNotIn(
            b"never-store-this", b"".join(p.read_bytes() for p in self.output.iterdir())
        )
        with self.assertRaises(urllib.error.HTTPError):
            self.post(body)
        self.assertEqual(len(self.received), 2)

    def test_unsupported_stream_never_forwarded(self):
        with self.assertRaises(urllib.error.HTTPError):
            self.post(b'{"messages":[{"role":"user","content":"x"}],"stream":true}')
        self.assertEqual(self.received, [])

    def test_recording_failure_never_forwarded(self):
        (self.output / "0000-request.json").write_text("existing record")
        with self.assertRaises(urllib.error.HTTPError):
            self.post(b'{"messages":[{"role":"user","content":"x"}]}')
        self.assertEqual(self.received, [])

    def test_frozen_gameplay_and_engine_cells(self):
        old = json.loads((BASE / "probe.ipynb").read_text())
        new = json.loads(build()["capture.ipynb"])
        self.assertEqual(old["cells"][2], new["cells"][3])
        code = "".join(new["cells"][4]["source"])
        self.assertIn("TYCHO_KAGGLE_MAX_ACTIONS'] = '10'", code)
        self.assertIn("TYCHO_PUBLIC_GAME_IDS'] = 'ls20'", code)
        self.assertNotIn("submission.to_parquet", code)
        # The complete original embedded solver archive must still be present.
        old_strings = [
            n.value
            for n in ast.walk(ast.parse("".join(old["cells"][1]["source"])))
            if isinstance(n, ast.Constant)
            and isinstance(n.value, str)
            and n.value.startswith("UEsDB")
        ]
        self.assertTrue(old_strings)
        for archive in old_strings:
            self.assertIn(archive, "".join(new["cells"][2]["source"]))

    def test_asset_cell_executes_locally(self):
        notebook = json.loads(build()["capture.ipynb"])
        # Redirect only extraction destinations; embedded source bytes remain exact.
        source = "".join(notebook["cells"][2]["source"])
        root = Path(self.temp.name) / "assets"
        (root / "tmp").mkdir(parents=True)
        (root / "working").mkdir()
        source = source.replace("/tmp/", str(root / "tmp") + "/").replace(
            "/kaggle/working/", str(root / "working") + "/"
        )
        exec(source, {})
        self.assertEqual(
            (root / "tmp/capture_proxy.py").read_bytes(),
            (HERE / "capture_proxy.py").read_bytes(),
        )
        runtime = (root / "tmp/official_launcher_runtime.py").read_text()
        self.assertIn('"http://127.0.0.1:30001/v1"', runtime)


if __name__ == "__main__":
    unittest.main()
