"""Loopback-only, bounded, exact-body recorder for the non-streaming Actor API."""

import hashlib
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time

BODY_CAP = 2 * 1024 * 1024


class CaptureServer(ThreadingHTTPServer):
    daemon_threads = False

    def __init__(self, address, upstream, output, max_requests=96):
        if address[0] != "127.0.0.1" or upstream[0] != "127.0.0.1":
            raise ValueError("Only loopback endpoints are permitted")
        self.upstream = upstream
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.iterdir()):
            raise ValueError("Capture directory must be fresh; never overwrite a run")
        self.lock = threading.Lock()
        self.count = 0
        self.max_requests = max_requests
        super().__init__(address, Handler)

    def event(self, **value):
        with self.lock:
            with (self.output / "events.jsonl").open("a") as handle:
                handle.write(
                    json.dumps(dict(monotonic_ns=time.monotonic_ns(), **value)) + "\n"
                )
                handle.flush()

    def admit(self, body):
        with self.lock:
            if self.count >= self.max_requests:
                raise ValueError("Physical request cap reached")
            index = self.count
            path = self.output / f"{index:04d}-request.json"
            with path.open("xb") as handle:
                handle.write(body)
                handle.flush()
            self.count += 1
        self.event(
            event="recorded_before_forward",
            attempt=index,
            sha256=hashlib.sha256(body).hexdigest(),
            bytes=len(body),
        )
        return index


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass  # Never put request headers or credentials in notebook output.

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        connection = http.client.HTTPConnection(*self.server.upstream, timeout=200)
        try:
            connection.request("GET", self.path)
            response = connection.getresponse()
            body = response.read(BODY_CAP + 1)
            if len(body) > BODY_CAP:
                raise ValueError("Response body cap")
            self.reply(response.status, body)
        finally:
            connection.close()

    def reply(self, status, body):
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        index = None
        connection = None
        try:
            if self.path != "/v1/chat/completions":
                raise ValueError("Unexpected API route")
            if self.headers.get("Transfer-Encoding"):
                raise ValueError("Chunked request unsupported")
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= BODY_CAP:
                raise ValueError("Request body cap")
            self.connection.settimeout(200)
            body = self.rfile.read(size)
            if len(body) != size:
                raise ValueError("Incomplete request body")
            request = json.loads(body)
            if request.get("stream"):
                raise ValueError(
                    "This collection qualifies non-streaming Actor calls only"
                )
            if not request.get("messages"):
                raise ValueError("Missing messages")
            if any(
                k in request for k in ["api_key", "api_base", "base_url", "headers"]
            ):
                raise ValueError("Credentials or transport fields in body")
            index = self.server.admit(body)
            connection = http.client.HTTPConnection(*self.server.upstream, timeout=200)
            if index == 0:
                connection.request("GET", "/server_info")
                info = connection.getresponse()
                config = info.read(BODY_CAP + 1)
                if info.status != 200 or len(config) > BODY_CAP:
                    raise ValueError("Unable to retain resolved server configuration")
                json.loads(config)
                (self.server.output / "server-info.json").write_bytes(config)
            # Preserve payload bytes. Authentication headers are forwarded, never stored.
            headers = {"Content-Type": "application/json"}
            if self.headers.get("Authorization"):
                headers["Authorization"] = self.headers["Authorization"]
            connection.request("POST", self.path, body=body, headers=headers)
            self.server.event(event="forwarded", attempt=index)
            response = connection.getresponse()
            answer = response.read(BODY_CAP + 1)
            if len(answer) > BODY_CAP:
                raise ValueError("Response body cap; collection invalid")
            with (self.server.output / f"{index:04d}-response.json").open(
                "xb"
            ) as handle:
                handle.write(answer)
            self.server.event(
                event="upstream_response_complete",
                attempt=index,
                status=response.status,
                sha256=hashlib.sha256(answer).hexdigest(),
                bytes=len(answer),
            )
            self.reply(response.status, answer)
            self.server.event(event="downstream_write_completed", attempt=index)
        except Exception as error:
            self.server.event(
                event="failure",
                attempt=index,
                error_type=type(error).__name__,
                error=str(error),
            )
            try:
                self.reply(
                    502,
                    json.dumps(
                        {
                            "error": {
                                "message": "request capture failed; consult retained events",
                                "type": "capture_failure",
                            }
                        }
                    ).encode(),
                )
            except (ConnectionError, OSError):
                pass
        finally:
            if connection is not None:
                connection.close()


def start(output, port=30001, upstream_port=30000):
    server = CaptureServer(("127.0.0.1", port), ("127.0.0.1", upstream_port), output)
    thread = threading.Thread(
        target=server.serve_forever, name="actor-request-capture", daemon=True
    )
    thread.start()
    return server, thread
