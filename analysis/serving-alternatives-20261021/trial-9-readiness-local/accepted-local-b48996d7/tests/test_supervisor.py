import copy
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('screen_supervisor', ROOT / 'supervisor.py')
S = importlib.util.module_from_spec(spec)
spec.loader.exec_module(S)

SERVER = '''import http.server, os, time, sys
class Handler(http.server.BaseHTTPRequestHandler):
 def do_GET(self):
  mode=sys.argv[2]
  if mode=='die': os._exit(7)
  self.send_response(503 if mode=='error' else 200)
  self.send_header('Content-Length','20' if mode=='trickle' else '2')
  self.end_headers()
  if mode=='trickle':
   for i in range(20):
    self.wfile.write(b'x'); self.wfile.flush(); time.sleep(.15)
  else: self.wfile.write(b'ok')
 def log_message(self,*args): pass
http.server.HTTPServer(('127.0.0.1',int(sys.argv[1])), Handler).serve_forever()
'''


def command(code, seconds=2):
    return dict(argv=[sys.executable, '-c', code], seconds=seconds)


def fixture(path):
    port_socket = socket.socket()
    port_socket.bind(('127.0.0.1', 0))
    port = port_socket.getsockname()[1]
    port_socket.close()
    (path / 'server.py').write_text(SERVER)
    arm = dict(prepare=command('print("prepared")'), identity=command('print("runtime identity")'),
               server=dict(argv=[sys.executable, str(path / 'server.py'), str(port), 'ok'], seconds=4),
               replay=command('import time; time.sleep(.04); print("replayed")'), qualify=command('print("qualified")'),
               ready_url=f'http://127.0.0.1:{port}/health', ready_seconds=2, request_seconds=.6, drain_seconds=.2, ready_poll_seconds=.05)
    return dict(mode='CPU_FIXTURE_ONLY', purpose='PAIRED_CAPACITY', run_id='fixture-only', cwd=str(path), total_seconds=8,
                cleanup_seconds=1, hardware=dict(argv=[sys.executable, '-c', 'print("NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 580.178, GPU-CPU-FIXTURE")'],
                seconds=1, minimum_memory_mib=90000), control=copy.deepcopy(arm), candidate=copy.deepcopy(arm), corpus_prepare=command('print("verified corpus")'), release_check=command('print("released")'), paired_reduce=command('print("reduced")'))


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.config = fixture(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def run_fixture(self):
        s = S.Supervisor(self.config, self.path / 'out')
        result = s.run()
        events = [json.loads(x) for x in (self.path / 'out/supervisor.jsonl').read_text().splitlines()]
        self.assertEqual(result['surviving_groups'], [])
        self.assertLess(result['elapsed_seconds'], self.config['total_seconds'] + .3)
        return result, events

    def test_two_fresh_processes_order_and_cleanup(self):
        result, events = self.run_fixture()
        self.assertTrue(result['pass_'], result)
        spawns = [e for e in events if e['event'] == 'spawn']
        servers = [e for e in spawns if e['phase'].endswith('.server')]
        self.assertEqual(len(servers), 2)
        self.assertNotEqual(servers[0]['pid'], servers[1]['pid'])
        cleanup = next(e for e in events if e['event'] == 'cleanup' and e['pgid'] == servers[0]['pid'])
        self.assertLess(cleanup['seq'], servers[1]['seq'])
        self.assertEqual(spawns[0]['phase'], 'hardware')

    def test_wrong_hardware_prevents_install_and_both_servers(self):
        self.config['hardware'] = command('print("Tesla T4, 16000, 580, GPU-wrong")') | dict(minimum_memory_mib=90000)
        result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        self.assertEqual([e['phase'] for e in events if e['event'] == 'spawn'], ['hardware'])

    def test_control_failure_blocks_candidate(self):
        self.config['control']['qualify'] = command('raise SystemExit(5)')
        result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        self.assertFalse(any(e.get('phase', '').startswith('candidate') for e in events))

    def test_trickle_response_has_absolute_deadline(self):
        self.config['control']['server']['argv'][-1] = 'trickle'
        self.config['control']['request_seconds'] = .3
        result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        self.assertIn('health absolute deadline', result['error'])
        self.assertLess(result['elapsed_seconds'], 2)

    def test_http_error_is_preserved_and_not_retried(self):
        self.config['control']['server']['argv'][-1] = 'error'
        result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        health_exits = [e for e in events if e['event'] == 'exit' and '.health-' in e.get('phase', '') and e['returncode'] != 75]
        self.assertEqual(len(health_exits), 1)
        self.assertEqual(health_exits[0]['returncode'], 1)
        raw = (self.path / 'out' / (health_exits[0]['phase'] + '.stdout')).read_bytes()
        header, body = raw.split(b'\n', 1)
        self.assertEqual(json.loads(header)['status'], 503)
        self.assertEqual(body, b'ok')

    def test_server_death_before_ready_stops_candidate(self):
        self.config['control']['server']['argv'][-1] = 'die'
        result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        self.assertFalse(any(e.get('phase', '').startswith('candidate') for e in events))

    def test_server_death_during_replay(self):
        self.config['control']['server']['argv'] = [sys.executable, '-c',
            'import subprocess,sys,time; p=subprocess.Popen(sys.argv[1:]); time.sleep(.5); p.kill()',
            sys.executable, str(self.path / 'server.py'), self.config['control']['server']['argv'][-2], 'ok']
        self.config['control']['replay'] = command('import time; time.sleep(2)')
        result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        self.assertIn('server died', result['error'])

    def test_ignored_term_and_descendant_cleanup(self):
        code = ('import subprocess,sys,time,signal; signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                'subprocess.Popen([sys.executable,"-c","import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(20)"]); time.sleep(20)')
        self.config['control']['prepare'] = command(code, .15)
        result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        cleanups = [e for e in events if e['event'] == 'cleanup']
        self.assertTrue(all(e['remaining'] == [] for e in cleanups))

    def test_descendant_outlives_successful_parent_is_killed(self):
        self.config['control']['prepare'] = command('import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(20)"])')
        result, events = self.run_fixture()
        self.assertTrue(result['pass_'], result)
        self.assertTrue(all(not e['remaining'] for e in events if e['event'] == 'cleanup'))

    def test_global_deadline_not_reset_between_phases(self):
        self.config['total_seconds'] = 1.3
        self.config['cleanup_seconds'] = .2
        self.config['control']['prepare'] = command('import time; time.sleep(.6)')
        self.config['control']['identity'] = command('import time; time.sleep(.6)')
        result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        starts = [e for e in events if e['event'] == 'spawn']
        self.assertLessEqual(max(e['deadline'] for e in starts), events[0]['work_deadline'])
        self.assertFalse(any(e.get('phase', '').startswith('candidate') for e in events))

    def test_literal_argv_has_no_shell_expansion(self):
        self.config['control']['identity']['argv'] = [sys.executable, '-c', 'import sys; print(sys.argv[1])', '$(touch SHOULD_NOT_EXIST)']
        result, events = self.run_fixture()
        self.assertTrue(result['pass_'], result)
        self.assertFalse((self.path / 'SHOULD_NOT_EXIST').exists())

    def test_real_cli_fixture_and_no_rerun_output(self):
        cfg = self.path / 'config.json'
        cfg.write_text(json.dumps(self.config))
        argv = [sys.executable, str(ROOT / 'supervisor.py'), str(cfg), str(self.path / 'out'), '--cpu-fixture']
        result = subprocess.run(argv, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        result2 = subprocess.run(argv, capture_output=True, timeout=10)
        self.assertNotEqual(result2.returncode, 0)

    def test_gpu_mode_cannot_use_fixture_entrypoint(self):
        self.config['mode'] = 'GPU_SCREEN'
        self.config['hardware']['argv'] = S.GPU_ARGV
        cfg = self.path / 'config.json'
        cfg.write_text(json.dumps(self.config))
        result = subprocess.run([sys.executable, str(ROOT / 'supervisor.py'), str(cfg), str(self.path / 'out'), '--cpu-fixture'], capture_output=True, timeout=2)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.path / 'out').exists())

    def test_no_implicit_cap_or_shell_string(self):
        del self.config['total_seconds']
        with self.assertRaises(KeyError):
            S.validate(self.config)
        self.config = fixture(self.path)
        self.config['candidate']['server']['argv'] = 'python -m server'
        with self.assertRaises(ValueError):
            S.validate(self.config)

    def test_candidate_feasibility_never_runs_control_or_replay(self):
        self.config['purpose'] = 'CANDIDATE_FEASIBILITY'
        self.config['arm_order'] = ['candidate']
        self.config['candidate']['functional'] = self.config['candidate'].pop('replay')
        del self.config['control']
        result, events = self.run_fixture()
        self.assertTrue(result['pass_'], result)
        phases = [e['phase'] for e in events if e['event'] == 'spawn']
        self.assertIn('candidate.functional', phases)
        self.assertFalse(any(p.startswith('control') or 'replay' in p for p in phases))

    def test_candidate_feasibility_refuses_hidden_control_order(self):
        self.config['purpose'] = 'CANDIDATE_FEASIBILITY'
        self.config['arm_order'] = ['control', 'candidate']
        with self.assertRaises(ValueError):
            S.validate(self.config)

    def test_functional_worker_hang_is_killed_at_explicit_phase_deadline(self):
        self.config['purpose'] = 'CANDIDATE_FEASIBILITY'
        self.config['candidate']['functional'] = command('import time; time.sleep(20)', .15)
        result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        self.assertIn('candidate.functional: absolute deadline', result['error'])
        self.assertFalse(any(e.get('phase') == 'candidate.qualify' for e in events))

    def test_startup_allowance_does_not_clip_declared_functional_budget(self):
        self.config['purpose'] = 'CANDIDATE_FEASIBILITY'
        self.config['candidate']['server']['seconds'] = .6
        self.config['candidate']['functional'] = command('import time; time.sleep(.7)', 1.3)
        result, events = self.run_fixture()
        self.assertTrue(result['pass_'], result)

    def test_preexisting_listener_cannot_satisfy_readiness(self):
        from urllib.parse import urlsplit
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', urlsplit(self.config['control']['ready_url']).port))
            listener.listen()
            result, events = self.run_fixture()
        self.assertFalse(result['pass_'])
        self.assertFalse(any(e.get('phase', '').endswith('.server') for e in events))

    def test_production_health_cadence_cannot_use_cpu_shortcut(self):
        self.config['mode'] = 'GPU_SCREEN'
        self.config['hardware']['argv'] = S.GPU_ARGV
        with self.assertRaisesRegex(ValueError, 'cadence'):
            S.validate(self.config)


if __name__ == '__main__':
    unittest.main()
