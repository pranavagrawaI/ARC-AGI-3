"""Review reproductions: real pipes/processes and independently timed contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from qualify import HERE, HybridSolver, ExampleGame, observation
from hybrid.transport import HybridLocalSolver
from tycho_nooa.protocol import WORKER_PREFIX, PROTOCOL_VERSION


def ready_worker(body):
    return ("import json,sys,time,os,subprocess\nfrom pathlib import Path\n"
        "sys.stdin.readline()\n"
        f"print({WORKER_PREFIX!r}+json.dumps({{'protocol_version':{PROTOCOL_VERSION},'type':'ready','metadata':{{}}}}),flush=True)\n"
        + body)


def is_running(pid):
    path = Path(f'/proc/{pid}/stat')
    return path.exists() and path.read_text().split(') ')[1].split()[0] != 'Z'


class ReviewRegressions(unittest.TestCase):
    def test_small_runner_selects_three_games_and_removes_action_ceiling(self):
        from hybrid import runner
        from taaf.game_api import GameAPI
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('benchmark.pkl','target.pkl'): (root/name).touch()
            invoked=[]
            async def run(**kwargs): invoked.append(kwargs)
            bm=SimpleNamespace(solver=HybridSolver(max_actions_per_game=16),
                games=[GameAPI(env_name=name+'-fixture')
                       for name in ('sb26','unselected','ls20','ar25')],
                job_dir=root,game_weights=[1]*4,n_passes=5,game_runs=[],run=run,
                request_stop=lambda:None)
            with patch.object(sys,'argv',['runner',str(root/'benchmark.pkl'),str(root/'target.pkl')]), \
                 patch.object(runner.pickle,'load',side_effect=[bm,SimpleNamespace(actual_run_as_submission=False)]), \
                 patch('signal.signal'), \
                 patch.dict(os.environ,HYBRID_RUN_MODE='small',TYCHO_MODE='orchestrator',
                            TYCHO_NOOA_TREATMENT='control',ARC_LLM_MODEL='fixture',ARC3_MAX_ACTIVE_STREAMS='10'):
                runner.main()
            config=json.loads((root/'tycho-dfranzen/candidate-config.json').read_text())
            self.assertEqual(config['game_names'],['sb26-fixture','ls20-fixture','ar25-fixture'])
            self.assertEqual(config['per_game_seconds'],2400)
            self.assertIsNone(config['per_game_action_cap'])
            self.assertEqual(config['concurrency'],3)
            self.assertEqual(config['max_active_games'],10)
            self.assertEqual(config['passes'],1)
            self.assertEqual(len(invoked),1)

    def test_expired_game_budget_is_recorded_without_claiming_model_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'worker_entry.py').write_text(ready_worker('time.sleep(90)\n'))
            game=ExampleGame(); game.start_game()
            solver=HybridSolver(runtime_root=str(root),worker_python=sys.executable,
                max_runtime_s_per_game=.3,analyzer_timeout=10)
            solver.job_dir=root; solver.setup()
            try:
                solver._play(game,0)
                self.assertEqual(game.game_run.state,'gave_up')
                self.assertEqual(len(game.game_run.history),0)
                receipt=json.loads(next(root.glob('hybrid-games/*/execution-receipt.json')).read_text())
                self.assertEqual(receipt['stop_reason'],'game_deadline')
                self.assertTrue(receipt['unresolved_provider_usage'])
                self.assertIn('TimeoutError',receipt['failure'])
                self.assertIsNotNone(receipt['worker_returncode'])
            finally: solver.teardown()

    def test_requested_construction_uses_remaining_absolute_budget(self):
        from hybrid.deadlines import RequestedBuilderConfig
        short = RequestedBuilderConfig().start(164,now=0)
        self.assertEqual(short.builder_seconds_remaining(now=30),64.5)
        long = RequestedBuilderConfig().start(900,now=0)
        self.assertEqual(long.builder_seconds_remaining(now=300),240)
        self.assertEqual(long.builder_seconds_remaining(now=850),22.5)
        self.assertEqual(long.builder_seconds_remaining(now=895),0)
        self.assertEqual(long.builder_seconds_remaining(now=950),0)
        self.assertEqual(long.trace_fields()['allocation_policy'],'actor-request-v3')

    def test_pipe_write_and_response_share_one_deadline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root/'worker.py'
            entry.write_text(ready_worker("time.sleep(0.7)\nsys.stdin.readline()\ntime.sleep(10)\n"))
            game = ExampleGame(); game.start_game()
            worker = HybridLocalSolver(game_id='pipe', workspace_root=root,
                initial_observation=observation(game), command=[sys.executable,str(entry)],
                timeout_seconds=2)
            try:
                worker.timeout_seconds = 1
                start = time.monotonic()
                worker._send_payload({'type':'fixture', 'data':'x'*1_000_000})
                with self.assertRaises(TimeoutError):
                    worker._receive('fixture')
                elapsed = time.monotonic()-start
                self.assertLess(elapsed,1.4, f'write and read used {elapsed:.3f}s of a 1s budget')
            finally:
                worker.close()

    def test_close_stops_model_python_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            entry = root/'worker.py'
            entry.write_text(ready_worker(
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],"
                "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
                "Path('child.pid').write_text(str(child.pid))\n"
                "sys.stdin.readline()\n"))
            game = ExampleGame(); game.start_game()
            worker = HybridLocalSolver(game_id='descendant', workspace_root=root,
                initial_observation=observation(game), command=[sys.executable,str(entry)],
                timeout_seconds=2)
            pid = None
            try:
                for _ in range(100):
                    if (worker.session_root/'child.pid').exists(): break
                    time.sleep(.01)
                pid = int((worker.session_root/'child.pid').read_text())
                self.assertTrue(is_running(pid))
                worker.close()
                for _ in range(100):
                    if not is_running(pid): break
                    time.sleep(.01)
                self.assertFalse(is_running(pid), 'model Python survived worker close')
                self.assertIsNotNone(worker._process.poll())
            finally:
                worker.close()
                if pid and is_running(pid): os.kill(pid,9)

    def test_artifact_write_failure_still_reaps_and_releases(self):
        import hybrid.solver as module
        import inference.agent.tool_agent as donor
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'worker_entry.py').write_text(ready_worker('time.sleep(90)\n'))
            game = ExampleGame(); game.start_game()
            solver = HybridSolver(runtime_root=str(root), worker_python=sys.executable,
                max_actions_per_game=0, analyzer_timeout=.2)
            solver.job_dir=root; solver.setup()
            gate = donor._PriorityGate(1)
            workers = []
            original = Path.write_text
            def write(path,*args,**kwargs):
                if path.name == 'compute.json': raise OSError('fixture disk write failure')
                return original(path,*args,**kwargs)
            def create(**kwargs):
                worker=HybridLocalSolver(**kwargs); workers.append(worker); return worker
            try:
                with patch.object(module,'_priority_gate',return_value=gate), \
                     patch.object(module,'HybridLocalSolver',side_effect=create), \
                     patch.object(Path,'write_text',write):
                    with self.assertRaisesRegex(OSError,'fixture disk write failure'):
                        solver._play(game,0)
                self.assertTrue(workers)
                self.assertTrue(all(w._process.poll() is not None for w in workers))
                self.assertFalse(solver._workers)
                self.assertEqual(gate._free,1)
                self.assertIsNotNone(game.game_run.final_score)
            finally:
                for worker in workers: worker.close()
                solver.teardown()


if __name__=='__main__': unittest.main(verbosity=2)
