"""Stand-in for the SGLang server, for CPU fault injection. Not a model: every turn calls the python tool with one
random arrow action, so the harness runs its real tool loop and executes real engine actions.

usage: fake_llm.py --port P [--ready-delay S] [--die-after S] [--mode healthy|http500] [--latency S]
  --ready-delay  /health answers 503 (and chat refuses) for S seconds, like a loading server
  --die-after    the process exits with code 3 after S seconds, like a crashed server
"""
import argparse, json, os, random, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

p = argparse.ArgumentParser()
p.add_argument('--port', type=int, required=True)
p.add_argument('--ready-delay', type=float, default=0)
p.add_argument('--die-after', type=float, default=0)
p.add_argument('--mode', default='healthy')
p.add_argument('--latency', type=float, default=0.2)
a = p.parse_args()
T0 = time.time()
CODE = ("import random\n"
        "try:\n    action(random.choice(['ACTION1','ACTION2','ACTION3','ACTION4']))\n"
        "except Exception as e:\n    print('action failed', e)\n")


def message():
    return {'role': 'assistant', 'content': '', 'reasoning_content': 'fake turn',
            'tool_calls': [{'id': f'call_{random.randrange(1 << 30)}', 'type': 'function',
                            'function': {'name': 'python', 'arguments': json.dumps({'code': CODE})}}]}


USAGE = {'prompt_tokens': 1000, 'completion_tokens': 50, 'total_tokens': 1050}


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    def log_message(self, *x): pass
    def _json(self, code, body):
        b = json.dumps(body).encode()
        self.send_response(code); self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        ready = time.time() - T0 >= a.ready_delay
        self._json(200 if ready else 503, {'ready': ready})
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get('Content-Length') or 0)) or b'{}')
        if time.time() - T0 < a.ready_delay or a.mode == 'http500':
            return self._json(503 if a.mode != 'http500' else 500, {'error': 'injected'})
        time.sleep(a.latency)
        msg = message()
        if not body.get('stream'):
            return self._json(200, {'choices': [{'message': msg, 'finish_reason': 'tool_calls'}], 'usage': USAGE})
        delta = {'role': 'assistant', 'reasoning_content': msg['reasoning_content'],
                 'tool_calls': [{'index': 0, **msg['tool_calls'][0]}]}
        chunks = [{'choices': [{'index': 0, 'delta': delta, 'finish_reason': None}]},
                  {'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'tool_calls'}]},
                  {'choices': [], 'usage': USAGE}]
        data = b''.join(f'data: {json.dumps(c)}\n\n'.encode() for c in chunks) + b'data: [DONE]\n\n'
        self.send_response(200); self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)


if a.die_after:
    threading.Timer(a.die_after, lambda: os._exit(3)).start()
ThreadingHTTPServer(('127.0.0.1', a.port), H).serve_forever()
