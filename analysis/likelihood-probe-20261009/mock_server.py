"""Mock of the two SGLang endpoints probe.py uses, for a local dry run. Fake logprob = -(token_id % 7)/10."""
import json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/generate":
            ids, s = b["input_ids"], b["logprob_start_len"]
            lp = [[None if i == 0 else -(t % 7) / 10, t, None] for i, t in enumerate(ids)][s:]
            r = {"text": "x", "meta_info": {"input_token_logprobs": lp}}
        else:
            r = {"usage": {"prompt_tokens": 1234}}
        d = json.dumps(r).encode(); self.send_response(200); self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
