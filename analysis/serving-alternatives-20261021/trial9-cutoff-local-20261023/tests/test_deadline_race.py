"""CPU loopback scheduling witnesses; no remote service or GPU is used.

Delay one real parent shutdown so a second real socket's timer can expire
before that socket gets an owned-shutdown marker. This is fault injection,
not an incidence estimate or proof of trial9's unrecorded timer provenance.
"""
import hashlib
import copy
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import threading
import time
import unittest
from unittest.mock import patch
import test_replay as fixtures
import replay
import reduce_api


def silent(handler):
    body=handler.rfile.read(int(handler.headers['Content-Length']))
    handler.server.seen.append((handler.headers['X-Request-Id'],body,time.monotonic()))
    time.sleep(.65)


class ObservedSocket:
    def __init__(self,sock,connection,log,delay_first_shutdown):
        self.sock=sock;self.connection=connection;self.log=log
        self.delay_first_shutdown=delay_first_shutdown
    def __getattr__(self,key):return getattr(self.sock,key)
    def settimeout(self,value):
        before=time.monotonic();self.sock.settimeout(value)
        self.log.append(dict(kind='actual_settimeout',t=before,ended=time.monotonic(),
                             seconds=self.sock.gettimeout(),connection=id(self.connection)))
    def shutdown(self,how):
        self.log.append(dict(kind='actual_shutdown',t=time.monotonic(),connection=id(self.connection)))
        try:return self.sock.shutdown(how)
        finally:
            if self.delay_first_shutdown and threading.current_thread() is threading.main_thread():
                self.delay_first_shutdown=False
                # Only the first of all parent shutdown calls is delayed.
                if not any(x['kind']=='shutdown_delay' for x in self.log):
                    self.log.append(dict(kind='shutdown_delay',t=time.monotonic(),seconds=.08))
                    time.sleep(.08)


class DeadlineRaceTests(unittest.TestCase):
    run_fixture=fixtures.ReplayTests.run_fixture
    def preserve(self,label,root,obj,ok,events,external,**extra):
        target=os.environ.get('LOCAL_EVIDENCE')
        if not target:return
        destination=Path(target)/label
        destination.mkdir(parents=True,exist_ok=False)
        shutil.copytree(root/'out',destination/'out')
        source=Path(replay.__file__).read_bytes()
        (destination/'witness.json').write_text(json.dumps(dict(
            scope='LOCAL_LOOPBACK_SCHEDULING_FAULT_INJECTION',source_sha256=hashlib.sha256(source).hexdigest(),
            run_ok=ok,failed=obj.failed,cutoff=obj.end,external=external,**extra),indent=2)+'\n')

    def test_deadline_timer_races_sequential_parent_shutdown(self):
        actual=replay.CaptureConnection.connect;external=[]
        def connect(connection):
            actual(connection)
            proxy=ObservedSocket(connection.sock,connection,external,True)
            connection.sock=connection.raw_socket=proxy
        with patch.object(fixtures.Handler,'do_POST',silent),patch.object(replay.CaptureConnection,'connect',connect):
            root,server,obj,events,ok,elapsed=self.run_fixture('ok',(.1,.15))
        receipts=[json.loads(p.read_text()) for p in (root/'out').glob('*/http.json')]
        self.preserve('deadline-race',root,obj,ok,events,external,elapsed=elapsed)
        self.assertEqual(len(server.seen),2)
        self.assertEqual(len(receipts),2)
        # The corrected experiment must succeed by owned EOF, not by accepting
        # arbitrary post-cutoff timeouts. This assertion is RED on frozen code.
        self.assertTrue(ok,receipts)
        self.assertFalse(obj.failed)
        for receipt in receipts:
            self.assertEqual(receipt['cause_type'],'RemoteDisconnected')
            self.assertEqual(receipt['termination'],'INTENTIONAL_CUTOFF')
            reduce_api.pending_http(receipt,obj.end,lambda ref:(root/'out'/ref).read_bytes(),receipt['rid'])

    def timeout_case(self,label,delayed):
        detected=[];actual=http.client.HTTPConnection.getresponse;external=[]
        def getresponse(connection):
            try:return actual(connection)
            except TimeoutError:
                detected.append(time.monotonic())
                if delayed:time.sleep(.4)
                raise
        # Explicitly compress the real read timer on both frozen and proposed
        # implementations, not the fixed cutoff. Connect keeps its 2s bound.
        actual_timeout=socket.socket.settimeout
        def timeout(sock,value):
            applied=.05 if value is not None and (value==900. or 0<value<1.) else value
            result=actual_timeout(sock,applied)
            if applied==.05:external.append(dict(kind='test_compressed_settimeout',t=time.monotonic(),requested=value,actual=sock.gettimeout()))
            return result
        with patch.object(fixtures.Handler,'do_POST',silent),patch.object(socket.socket,'settimeout',timeout),patch.object(http.client.HTTPConnection,'getresponse',getresponse):
            root,server,obj,events,ok,elapsed=self.run_fixture('ok',(.1,.2))
        receipts=[json.loads(p.read_text()) for p in (root/'out').glob('*/http.json')]
        self.preserve(label,root,obj,ok,events,external,detected_at=detected,elapsed=elapsed)
        self.assertEqual(len(detected),2)
        self.assertTrue(all(t<obj.end for t in detected))
        self.assertFalse(ok);self.assertTrue(obj.failed)
        for receipt in receipts:
            self.assertEqual(receipt['termination'],'ERROR')
            self.assertTrue(receipt['error'].startswith('TimeoutError('))
            self.assertEqual(receipt['phase_at_error'],'READ_HEADERS')
            arm=receipt['read_timeout']
            self.assertEqual(arm['mode'],'SOCKET_INACTIVITY')
            self.assertEqual(arm['actual_seconds'],.05)
            self.assertEqual(arm['cutoff'],obj.end)
            if delayed:
                self.assertGreater(receipt['error_observed_at'],obj.end)
                self.assertLess(receipt['cutoff_shutdown_at'],receipt['error_observed_at'])

    def test_real_inactivity_timeout_is_fatal(self):self.timeout_case('inactivity',False)
    def test_predeadline_timeout_delayed_past_owned_shutdown_is_fatal(self):self.timeout_case('delayed-timeout',True)

    def test_production_default_is_900_and_recorded(self):
        self.assertEqual(replay.READ_INACTIVITY_SECONDS,900.)
        with patch.object(fixtures.Handler,'do_POST',silent):
            root,server,obj,events,ok,elapsed=self.run_fixture('ok',(.05,.1))
        self.assertTrue(ok)
        for path in (root/'out').glob('*/http.json'):
            row=json.loads(path.read_text());arm=row['read_timeout']
            self.assertEqual(arm['actual_seconds'],900.)
            self.assertEqual(arm['mode'],'SOCKET_INACTIVITY')
            self.assertLessEqual(arm['started_at'],arm['armed_at'])
            self.assertLess(arm['armed_at'],obj.end)
            self.assertEqual(arm['cutoff'],obj.end)

    def test_reducer_does_not_trust_a_timeout_relabel(self):
        with patch.object(fixtures.Handler,'do_POST',silent):
            root,server,obj,events,ok,elapsed=self.run_fixture('ok',(.05,.1))
        for path in (root/'out').glob('*/http.json'):
            row=json.loads(path.read_text())
            row.update(cause="TimeoutError('timed out')",cause_type='TimeoutError')
            with self.assertRaisesRegex(ValueError,'timeout'):
                reduce_api.pending_http(row,obj.end,lambda ref:(root/'out'/ref).read_bytes(),row['rid'])

    def test_missing_wrong_deadline_phase_or_inactivity_evidence_rejected(self):
        with patch.object(fixtures.Handler,'do_POST',silent):
            root,server,obj,events,ok,elapsed=self.run_fixture('ok',(.05,.1))
        self.assertTrue(ok)
        original=json.loads(next((root/'out').glob('*/http.json')).read_text())
        changes=[('read_timeout',None),('phase_at_error',None),('phase_at_error','PROTOCOL')]
        for key,value in changes:
            row=copy.deepcopy(original);row[key]=value
            with self.subTest(key=key,value=value),self.assertRaises((ValueError,KeyError,TypeError)):
                reduce_api.pending_http(row,obj.end,lambda ref:(root/'out'/ref).read_bytes(),row['rid'])
        for key,value in [('mode','ABSOLUTE'),('actual_seconds',.15),('cutoff',obj.end+1),('armed_at',original['ended']+1)]:
            row=copy.deepcopy(original);row['read_timeout'][key]=value
            with self.subTest(key=key,value=value),self.assertRaises(ValueError):
                reduce_api.pending_http(row,obj.end,lambda ref:(root/'out'/ref).read_bytes(),row['rid'])

    def test_time_compressed_inactivity_is_not_absolute_read_expiry(self):
        with patch.object(replay,'READ_INACTIVITY_SECONDS',.05):
            root,server,obj,events,ok,elapsed=self.run_fixture('slow',(.1,.15))
        self.assertTrue(ok);self.assertFalse(obj.failed)
        for path in (root/'out').glob('*/http.json'):
            row=json.loads(path.read_text())
            self.assertEqual(row['termination'],'INTENTIONAL_CUTOFF')
            self.assertGreater(row['received_bytes'],1)
            self.assertGreater(obj.end-row['read_timeout']['armed_at'],.05)


if __name__=='__main__':unittest.main()
