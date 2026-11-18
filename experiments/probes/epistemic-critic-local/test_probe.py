import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import build_cases
import probe


class LocalCriticTests(unittest.TestCase):
    def test_preserved_prefix_has_only_first_three_observations(self):
        cases, _ = build_cases.build()
        evidence = cases[0]['evidence']
        self.assertEqual([e['step'] for e in evidence], [1, 2, 3])
        self.assertEqual([e['action'] for e in evidence], ['CLICK 52 10', 'CLICK 10 31', 'CLICK 0 0'])
        self.assertEqual({c[0] for e in evidence for c in e['changes_y_x_before_after']}, {63})
        self.assertNotIn('Turn 7', cases[0]['actor_claim'])

    def test_controls_share_evidence_but_change_commitment(self):
        cases, rubrics = build_cases.build()
        self.assertEqual(len(cases), 8)
        for a, b, x, y in zip(cases[::2], cases[1::2], rubrics[::2], rubrics[1::2]):
            self.assertEqual(a['evidence'], b['evidence'])
            self.assertNotEqual(a['actor_proposal'], b['actor_proposal'])
            self.assertEqual((x['expected_verdict'], y['expected_verdict']), ('challenge', 'allow'))

    def test_fresh_requests_exclude_answers_and_source_labels(self):
        for _, body in probe.requests('test-model'):
            self.assertEqual([m['role'] for m in body['messages']], ['system', 'user'])
            packet = json.loads(body['messages'][1]['content'])
            self.assertEqual(set(packet), {'evidence', 'actor_claim', 'actor_proposal'})
            self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
            self.assertNotIn('tools', body)

    def test_prepare_has_no_network_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d, patch.object(probe.urllib.request, 'build_opener', side_effect=AssertionError('network')):
            out = Path(d) / 'attempt'
            result = probe.run(out, 'test-model')
            self.assertEqual(result['status'], 'prepared')
            self.assertEqual(len(list(out.glob('*.request.json'))), 8)
            with self.assertRaises(FileExistsError):
                probe.run(out, 'test-model')

    def test_local_endpoint_rejects_remote_and_credentials(self):
        for url in ['https://example.com/v1', 'http://example.com/v1', 'http://user:pass@127.0.0.1/v1', 'http://127.0.0.1/v1?key=x']:
            with self.assertRaises(ValueError): probe.local_endpoint(url)
        self.assertEqual(probe.local_endpoint('http://127.0.0.1:8000/v1/'), 'http://127.0.0.1:8000/v1')

    def test_fabricated_references_rejected_but_semantics_not_auto_graded(self):
        reply = {'verdict': 'challenge', 'evidence_refs': ['future:4'], 'issue': 'x', 'next_step': 'y'}
        self.assertTrue(probe.validate_reply(json.dumps(reply), {'transition:1'}))
        reply['evidence_refs'] = ['transition:1']
        self.assertEqual(probe.validate_reply(json.dumps(reply), {'transition:1'}), [])
        self.assertTrue(probe.validate_reply('not json', set()))

class HttpBoundaryTests(unittest.TestCase):
    def test_local_server_receives_eight_independent_requests(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        received = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args): pass
            def send_json(self, value):
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(value).encode())
            def do_GET(self):
                self.send_json({'data': [{'id': 'fixture-model'}]})
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                received.append(body)
                reply = {'verdict': 'allow', 'evidence_refs': [], 'issue': '', 'next_step': 'Proceed provisionally.'}
                self.send_json({'model': 'fixture-model', 'choices': [{'finish_reason': 'stop',
                    'message': {'content': json.dumps(reply)}}]})

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as d:
                result = probe.run(Path(d) / 'run', 'fixture-model', f'http://127.0.0.1:{server.server_port}/v1')
                self.assertEqual(result['status'], 'completed')
                self.assertEqual(len(received), 8)
                self.assertTrue(all(len(r['messages']) == 2 for r in received))
                self.assertEqual(len(list((Path(d) / 'run').glob('*.response.json'))), 8)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
