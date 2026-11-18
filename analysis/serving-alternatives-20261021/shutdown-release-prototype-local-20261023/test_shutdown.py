import json
import copy
import hashlib
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import shutdown

HERE = Path(__file__).resolve().parent


def records(path):
    return [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []


class ShutdownTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)/'native.jsonl'
        self.children = []
        self.cleanup_bindings = []
        self.results = []

    def tearDown(self):
        for binding in self.cleanup_bindings:
            shutdown.signal_owned(binding,list(binding['members']),signal.SIGKILL,[],time.monotonic()+1)
        for p in self.children:
            p.wait(timeout=2)
        def all_gone():
            try:
                return all(shutdown.read_process(pid) is None for b in self.cleanup_bindings for pid in b['members'])
            except shutdown.UnsafeIdentity:
                return False # exit in progress is unknown, never evidence of absence
        cleanup_end=time.monotonic()+1
        while not all_gone() and time.monotonic()<cleanup_end:time.sleep(.01)
        self.assertTrue(all_gone(),'fixture cleanup did not reach explicit absence/zombie within1s')
        output=os.environ.get('SHUTDOWN_PROOF_DIR')
        if output:
            Path(output).mkdir(exist_ok=True)
            proof=dict(native_events=records(self.path),release_results=self.results,
                       all_fixture_processes_gone=all_gone())
            with (Path(output)/(self._testMethodName+'.json')).open('x') as stream:
                json.dump(proof,stream,indent=2)
        self.tmp.cleanup()

    def release(self,*args,**kwargs):
        result=shutdown.release(*args,**kwargs)
        self.results.append(result)
        return result

    def start(self, stubborn=False):
        p = subprocess.Popen([sys.executable, str(HERE/'native_fixture.py'), 'wrapper', str(self.path), *(['--stubborn'] if stubborn else [])], start_new_session=True)
        self.children.append(p)
        end = time.monotonic()+2
        while time.monotonic()<end:
            ready = [r for r in records(self.path) if r['event']=='frontend_ready']
            if ready:
                frontend = ready[0]['pid']
                argv = [sys.executable, str(HERE/'native_fixture.py'), 'frontend', str(self.path), *(['--stubborn'] if stubborn else [])]
                sources=[HERE/'native_fixture.py',HERE.parent/'trial10-release-local-20261023/sources/pinned-sglang-srt-utils-watchdog.py',HERE.parent/'trial10-release-local-20261023/sources/pinned-sglang-srt-managers-tokenizer_manager.py']
                binding=shutdown.bind(p,frontend,argv,{str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
                self.cleanup_bindings.append(copy.deepcopy(binding))
                return p,binding
            time.sleep(.01)
        self.fail('fixture startup did not complete')

    def probes(self, host=True, gpu=True):
        return {'gpu': lambda: {'eligible':gpu, 'fixture_only':True},
                'apps': lambda: {'eligible':True, 'fixture_only':True},
                'cgroup': lambda: {'eligible':host, 'fixture_only':True},
                'port': lambda: {'eligible':True, 'fixture_only':True},
                'disk': lambda: {'eligible':True, 'fixture_only':True},
                'native': lambda: {'eligible':any(r['event']=='scheduler_shutdown_dispatched' for r in records(self.path)), 'events':records(self.path)}}

    def test_frontend_first_keeps_child_alive_until_native_watchdog_stopped(self):
        p, binding = self.start()
        result = self.release(binding, self.probes(), end=time.monotonic()+1, graceful_seconds=.6, fallback_reserve=.2)
        events = [r['event'] for r in records(self.path)]
        self.assertTrue(result['pass'], result)
        self.assertLess(events.index('watchdog_stopped'), events.index('scheduler_shutdown_dispatched'))
        self.assertNotIn('crash_path', events)
        self.assertFalse(result['fallback_used'])
        self.assertEqual(result['remaining'], [])

    def test_old_group_term_reproduces_native_crash_ordering(self):
        p, binding = self.start()
        os.killpg(p.pid, signal.SIGTERM)
        end=time.monotonic()+1
        while time.monotonic()<end and not any(r['event']=='crash_path' for r in records(self.path)):
            time.sleep(.01)
        events=[r['event'] for r in records(self.path)]
        self.assertIn('crash_path',events)
        self.assertNotIn('scheduler_shutdown_dispatched',events)
        self.assertTrue(any('detokenizer' in r.get('message','') and 'exit code -15' in r['message'] for r in records(self.path)))

    def test_persistent_resources_keep_both_failures_and_block_release(self):
        import resources
        p,binding=self.start()
        primary=HERE.parent/'trial-10-exact-wire-local/remote-readback/priority-output/serving-wire-capacity10/output/control-release.json'
        raw=json.loads(primary.read_text())['raw']
        probes=resources.providers(raw,raw['gpu'].split(',')[2].strip(),lambda:dict(eligible=True,fixture_only=True),self.probes()['native'])
        result=self.release(binding,probes,end=time.monotonic()+.3,graceful_seconds=.12,fallback_reserve=.1)
        self.assertFalse(result['pass'])
        self.assertEqual(result['remaining'],[])
        for sample in result['samples']:
            self.assertEqual(set(sample['probes']),set(shutdown.PROBES))
            self.assertFalse(sample['probes']['gpu']['eligible'])
            self.assertFalse(sample['probes']['cgroup']['eligible'])
        complete=[s for s in result['samples'] if 'computed_bytes' in s['probes']['cgroup']]
        self.assertTrue(complete)
        self.assertEqual(complete[-1]['probes']['cgroup']['computed_bytes'],138308407296)

    def test_identity_drift_refuses_all_signals(self):
        p,binding=self.start(stubborn=True)
        child=next(pid for pid in binding['members'] if pid not in (p.pid,binding['frontend']))
        binding['members'][child]['starttime']+=1
        result=self.release(binding,self.probes(),end=time.monotonic()+.3,graceful_seconds=.1,fallback_reserve=.1)
        self.assertFalse(result['pass'])
        self.assertIn('identity drift',result['error'])
        self.assertEqual(result['events'],[])
        self.assertIsNone(p.poll())

    def test_unknown_group_member_refuses_all_signals(self):
        p,binding=self.start(stubborn=True)
        child=next(pid for pid in binding['members'] if pid not in (p.pid,binding['frontend']))
        del binding['members'][child]
        result=self.release(binding,self.probes(),end=time.monotonic()+.3,graceful_seconds=.1,fallback_reserve=.1)
        self.assertFalse(result['pass'])
        self.assertIn('unknown process',result['error'])
        self.assertEqual(result['events'],[])
        self.assertIsNone(p.poll())

    def test_fallback_has_one_deadline_and_removes_stubborn_owned_processes(self):
        p,binding=self.start(stubborn=True)
        start=time.monotonic();end=start+.3
        result=self.release(binding,self.probes(),end=end,graceful_seconds=.1,fallback_reserve=.1)
        self.assertFalse(result['pass']) # no native intended-order completion
        self.assertTrue(result['fallback_used'])
        self.assertEqual(result['remaining'],[])
        self.assertLess(time.monotonic()-start,.5)
        self.assertTrue(all(e['end']<=end for e in result['events']))
        self.assertEqual({e['signal'] for e in result['events']},{signal.SIGTERM,signal.SIGKILL})

    def test_blocked_probe_is_bounded_and_missing_evidence_cannot_pass(self):
        p,binding=self.start()
        probes=self.probes()
        def blocked():
            time.sleep(3)
            return dict(eligible=True)
        probes['gpu']=blocked
        start=time.monotonic()
        result=self.release(binding,probes,end=start+.25,graceful_seconds=.1,fallback_reserve=.1,probe_seconds=.04)
        self.assertFalse(result['pass'])
        self.assertLess(time.monotonic()-start,.5)
        self.assertEqual(result['remaining'],[])
        self.assertTrue(all(not s['probes']['gpu']['eligible'] for s in result['samples']))

    def test_clocked_series_rechecks_transient_resources_without_resetting_end(self):
        p,binding=self.start()
        probes=self.probes()
        become_ready=time.monotonic()+.15
        probes['gpu']=lambda:dict(eligible=time.monotonic()>=become_ready,fixture_only=True)
        result=self.release(binding,probes,end=time.monotonic()+.5,graceful_seconds=.35,fallback_reserve=.1)
        self.assertTrue(result['pass'],result)
        self.assertTrue(any(not s['probes']['gpu']['eligible'] for s in result['samples']))
        self.assertTrue(result['samples'][-1]['probes']['gpu']['eligible'])
        for sample in result['samples']:
            self.assertLessEqual(sample['process']['start'],sample['process']['end'])
            for probe in sample['probes'].values():
                self.assertLessEqual(probe['start'],probe['end'])

    def test_invalid_caps_and_wrong_frontend_identity_are_rejected(self):
        p,binding=self.start()
        with self.assertRaises(shutdown.UnsafeIdentity):shutdown.bind(p,binding['frontend'],['wrong'],binding['source_files'])
        with self.assertRaises(ValueError):shutdown.release(binding,self.probes(),end=time.monotonic()+601,graceful_seconds=1,fallback_reserve=.1)

    def test_cleanup_failure_stays_visible(self):
        from unittest.mock import patch
        p,binding=self.start(stubborn=True)
        # Fault injection at the OS signal boundary: acknowledged delivery without
        # process death models a stuck cleanup; does not reproduce a real D-state.
        real=signal.pidfd_send_signal
        def drop_kill(fd,sig,*args):
            if sig!=signal.SIGKILL: return real(fd,sig,*args)
        with patch.object(signal,'pidfd_send_signal',drop_kill):
            result=self.release(binding,self.probes(),end=time.monotonic()+.2,graceful_seconds=.07,fallback_reserve=.07)
        self.assertFalse(result['pass'])
        self.assertTrue(result['remaining'])

    def test_identity_verification_crossing_deadline_cannot_signal_late(self):
        from unittest.mock import patch
        p,binding=self.start(stubborn=True)
        original=os.pidfd_open
        def slow_open(pid,*args):
            time.sleep(.06)
            return original(pid,*args)
        with patch.object(os,'pidfd_open',slow_open):
            result=self.release(binding,self.probes(),end=time.monotonic()+.04,graceful_seconds=.01,fallback_reserve=.01)
        self.assertFalse(result['pass'])
        self.assertEqual(result['events'],[])
        self.assertIsNone(p.poll())

    def test_wrong_bound_source_is_rejected_before_any_signal(self):
        p,binding=self.start(stubborn=True)
        first=next(iter(binding['source_files']))
        binding['source_files'][first]='0'*64
        result=self.release(binding,self.probes(),end=time.monotonic()+.2,graceful_seconds=.07,fallback_reserve=.07)
        self.assertFalse(result['pass'])
        self.assertEqual(result['events'],[])
        self.assertIn('source drift',result['error'])

    def test_live_incomplete_identity_is_not_classified_as_gone(self):
        from unittest.mock import patch
        p,binding=self.start(stubborn=True)
        original=Path.stat
        def unreadable(path,*args,**kwargs):
            if str(path)==f'/proc/{binding["frontend"]}/exe':raise FileNotFoundError('injected incomplete executable visibility')
            return original(path,*args,**kwargs)
        with patch.object(Path,'stat',unreadable):
            result=self.release(binding,self.probes(),end=time.monotonic()+.2,graceful_seconds=.07,fallback_reserve=.07)
        self.assertFalse(result['pass'])
        self.assertEqual(result['events'],[])
        self.assertIn('identity incomplete',result['error'])

    def test_malformed_gpu_does_not_hide_later_host_and_port_receipts(self):
        p,binding=self.start()
        probes=self.probes(host=False)
        probes['gpu']=lambda:dict(eligible='untyped')
        result=self.release(binding,probes,end=time.monotonic()+.25,graceful_seconds=.1,fallback_reserve=.1)
        self.assertFalse(result['pass'])
        complete=[s for s in result['samples'] if s['probes']['port']['start'] is not None]
        self.assertTrue(complete)
        for sample in complete:
            self.assertIn('typed eligibility',sample['probes']['gpu']['error'])
            self.assertFalse(sample['probes']['cgroup']['eligible'])
            self.assertTrue(sample['probes']['port']['eligible'])

    def test_missing_source_retains_primary_and_secondary_failure(self):
        p,binding=self.start(stubborn=True)
        binding['source_files']={str(Path(self.tmp.name)/'missing-source.py'):'0'*64}
        result=self.release(binding,self.probes(),end=time.monotonic()+.2,graceful_seconds=.07,fallback_reserve=.07)
        self.assertFalse(result['pass'])
        self.assertIn('source unavailable',result['error'])
        self.assertIn('source unavailable',result['cleanup_error'])
        self.assertEqual(result['events'],[])


if __name__=='__main__': unittest.main()
