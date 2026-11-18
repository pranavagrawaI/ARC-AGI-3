"""Actual HTTP parser failure; delayed delivery crosses the fixed cutoff."""
import http.client
import json
import socket
from pathlib import Path
import time
import unittest
from unittest.mock import patch
import test_replay as fixtures
import reduce_api

def malformed_status(handler):
    body=handler.rfile.read(int(handler.headers['Content-Length']))
    handler.server.seen.append((handler.headers['X-Request-Id'],body,time.monotonic()))
    handler.wfile.write(b'HTTP/1.1 200 '+b'X'*65536+b'\r\n\r\n');handler.wfile.flush()

class ProtocolCutoffTests(unittest.TestCase):
    run_fixture=fixtures.ReplayTests.run_fixture
    def test_real_line_too_long_delivered_after_cutoff_stays_fatal(self):
        actual=http.client.HTTPResponse._read_status;detected=[]
        def delayed(response):
            try:return actual(response)
            except http.client.LineTooLong:
                detected.append(time.monotonic());time.sleep(.3);raise
        with patch.object(fixtures.Handler,'do_POST',malformed_status),patch.object(http.client.HTTPResponse,'_read_status',delayed):
            root,server,obj,events,ok,elapsed=self.run_fixture('ok',(.05,.05))
        self.assertEqual(len(detected),2)
        self.assertTrue(all(t<obj.end for t in detected),'actual parser must detect before cutoff')
        self.assertFalse(ok,'definite parser failure must not become successful cutoff')
        self.assertTrue(obj.failed)
        for p in (root/'out').glob('*/http.json'):
            receipt=json.loads(p.read_text())
            self.assertEqual(receipt['termination'],'ERROR')
            self.assertEqual(receipt['error_kind'],'TRANSPORT_OR_PROTOCOL')
            self.assertTrue(receipt['error'].startswith('LineTooLong('))
    def test_pending_parser_rejects_definite_protocol_causes(self):
        witness=Path(__file__).resolve().parents[2]/'coordination-20261023/trial8-r2-protocol-witness.json'
        base=json.loads(witness.read_text())['receipts'][0]
        for kind in ('LineTooLong','HTTPException','BadStatusLine'):
            receipt=dict(base,cause=f"{kind}('definite parser failure')",cause_type=kind)
            with self.subTest(kind=kind),self.assertRaisesRegex(ValueError,'protocol'):
                reduce_api.pending_http(receipt,receipt['cutoff_shutdown_at']-.001,lambda ref:b'',receipt['rid'])
    def test_owned_cutoff_remote_eof_is_allowed_despite_badstatusline_ancestry(self):
        def no_response(handler):
            body=handler.rfile.read(int(handler.headers['Content-Length']))
            handler.server.seen.append((handler.headers['X-Request-Id'],body,time.monotonic()))
            time.sleep(.3)
        self.assertTrue(issubclass(http.client.RemoteDisconnected,http.client.BadStatusLine))
        self.assertTrue(issubclass(http.client.RemoteDisconnected,ConnectionResetError))
        # Isolate EOF from the competing socket-inactivity timeout. The parent
        # still owns the fixed .1s shutdown; no production clock is changed.
        actual_timeout=socket.socket.settimeout
        def timeout(sock,value):
            return actual_timeout(sock,1. if value is not None and 0<value<.2 else value)
        with patch.object(fixtures.Handler,'do_POST',no_response),patch.object(socket.socket,'settimeout',timeout):
            root,server,obj,events,ok,elapsed=self.run_fixture('ok',(.05,.05))
        self.assertTrue(ok);self.assertFalse(obj.failed)
        receipts=[json.loads(p.read_text()) for p in (root/'out').glob('*/http.json')]
        self.assertEqual(len(receipts),2)
        for receipt in receipts:
            self.assertEqual(receipt['cause_type'],'RemoteDisconnected')
            self.assertEqual(receipt['termination'],'INTENTIONAL_CUTOFF')
            reduce_api.pending_http(receipt,obj.end,lambda ref:(root/'out'/ref).read_bytes(),receipt['rid'])
