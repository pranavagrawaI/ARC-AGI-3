"""Provider-shaped local fixtures ONLY; never included in paid model context."""

import copy
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from experiment import catalog, evidence_rows
from transport import MODEL


def theory_one():
    evidence = catalog(evidence_rows()[:1])

    def item(identity, text):
        return dict(
            id=identity, text=text, evidence_refs=["transition:1"], claim_refs=["c1"]
        )

    return dict(
        schema_version=1,
        version=1,
        parent_sha256=None,
        evidence_refs=[dict(id=k, sha256=v) for k, v in evidence.items()],
        claims=[
            dict(
                id="c1",
                text="Fixture hypothesis: RIGHT does not change any pixels",
                status="tentative",
                evidence_refs=["transition:1"],
                counterevidence_refs=[],
                tested_scope="fixture claim only",
                open_conditions=["all generalization untested"],
            )
        ],
        observations=[item("o1", "Recorded transition available")],
        action_semantics=[item("a1", "RIGHT has a recorded successor")],
        objectives=[item("g1", "Objective unknown")],
        uncertainties=[
            item("u1", "Which observation could distinguish action effects?")
        ],
        experiments=[],
        procedures=[],
        current_policy=None,
        revisions=[],
    )


def decision_code():
    return """
print(self.operational_cycle('select', {'kind':'uncertainty','id':'u1'}))
print(self.operational_cycle('declare', 'explore'))
r = self.operational_cycle('read')
d = {'theory_sha256':r['sha256'],'mode':'explore','selection':{'kind':'uncertainty','id':'u1'},'claim_refs':['c1'],'evidence_refs':['transition:1'],'actions':['UP'],'rationale':'fixture discrimination','expected_observation':'measure changed pixels','revision_if_wrong':'narrow the claim'}
print(self.operational_cycle('decide',d))
print(self.operational_cycle('prepare',['UP']))
print(self.submit_actions(['UP'], rationale='fixture canonical decision'))
"""


def treatment_codes():
    first = "print(self.workflow_skill()); print(self.submit_actions(['UP'], rationale='fixture skip attempt'))"
    second = "\n".join(
        [
            "print(self.operational_cycle('observe',['transition:1']))",
            "print(self.operational_cycle('reconcile',{'evidence_refs':['transition:1'],'claim_refs':[],'disposition':'initial evidence'}))",
            f"print(self.operational_cycle('revise',{theory_one()!r}))",
            decision_code(),
        ]
    )
    third = """
r = self.operational_cycle('read')
t = r['theory']
old = self.operational_hash(t['claims'][0])
t['version'] += 1
t['parent_sha256'] = r['sha256']
t['evidence_refs'].append({'id':'transition:2','sha256':r['evidence_catalog']['transition:2']})
t['claims'][0]['status'] = 'contradicted'
t['claims'][0]['counterevidence_refs'] = ['transition:2']
t['revisions'] = [{'claim_id':'c1','previous_sha256':old,'reason':'counterevidence','evidence_refs':['transition:2']}]
print(self.operational_cycle('result',{'theory':t,'observation_id':'transition:2','relation':'recorded_continuation_not_action_outcome'}))
print(self.operational_cycle('observe',['transition:2']))
print(self.operational_cycle('reconcile',{'evidence_refs':['transition:2'],'claim_refs':['c1'],'disposition':'fixture revision retained'}))
r = self.operational_cycle('read')
t = r['theory']
t['version'] += 1
t['parent_sha256'] = r['sha256']
t['current_policy'] = None
t['revisions'] = []
print(self.operational_cycle('revise',t))
""" + decision_code()
    return [first, second, third]


def response(code, ordinal=1):
    return dict(
        id=f"fixture-{ordinal}",
        provider="Reka",
        model=MODEL,
        choices=[
            dict(
                index=0,
                finish_reason="tool_calls",
                message=dict(
                    role="assistant",
                    content=None,
                    refusal=None,
                    tool_calls=[
                        dict(
                            index=0,
                            id=f"call-{ordinal}",
                            type="function",
                            function=dict(
                                name="execute_python",
                                arguments=json.dumps({"code": code}),
                            ),
                        )
                    ],
                ),
            )
        ],
        usage=dict(
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            completion_tokens_details=dict(reasoning_tokens=0),
            cost=0.0001489,
        ),
        openrouter_metadata=dict(
            requested=MODEL,
            strategy="direct",
            attempt=1,
            pipeline=[],
            endpoints=dict(
                available=[dict(provider="Reka", model=MODEL, selected=True)]
            ),
        ),
    )


class Server:
    def __init__(self, replies):
        self.replies = copy.deepcopy(replies)
        self.requests = []

    def __enter__(self):
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                outer.requests.append(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                reply = (
                    outer.replies.pop(0)
                    if outer.replies
                    else (429, {"error": "fixture exhausted"})
                )
                status, reply = reply if isinstance(reply, tuple) else (200, reply)
                raw = reply if isinstance(reply, bytes) else json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}/chat"
        return self

    def __exit__(self, *args):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
