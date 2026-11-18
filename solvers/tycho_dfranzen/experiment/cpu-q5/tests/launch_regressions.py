"""Failures reported on the shared board, reproduced before GPU launch."""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from qualify import HERE, HybridSolver, ExampleGame, Benchmark
from hybrid.policy import ContextProjection


class LaunchRegressions(unittest.TestCase):
    def test_one_game_persistence_fault_does_not_cancel_its_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            solver=HybridSolver()
            bm=Benchmark(games=[ExampleGame(label='failed'),ExampleGame(label='survivor')],
                         job_dir=Path(directory),solver=solver)
            def play(active_solver,game,index):
                if index==0: raise OSError('fixture persistence fault')
                import time
                time.sleep(.2)
                if active_solver._stop.is_set():
                    game.game_run.state='cancelled'
                else:
                    game.game_run.solver_note='sibling completed normally'
                game.finish_game()
            with patch.object(HybridSolver,'_play',new=play): asyncio.run(bm.run())
            self.assertEqual(bm.game_runs[0].state,'crashed')
            self.assertEqual(bm.game_runs[1].state,'gave_up')
            self.assertEqual(bm.game_runs[1].solver_note,'sibling completed normally')

    def test_long_newest_exchange_keeps_tool_pairs_and_latest_result(self):
        messages=[{'role':'system','content':'instructions'},
                  {'role':'user','content':'current observation must remain'}]
        for i in range(8):
            messages += [{'role':'assistant','content':None,'reasoning_content':'r'*24000,
                          'tool_calls':[{'id':str(i),'type':'function',
                          'function':{'name':'run_python','arguments':'{"code":"print(42)"}'}}]},
                         {'role':'tool','tool_call_id':str(i),'content':str(i)+'t'*30000}]
        projection=ContextProjection()
        selected=projection.project(messages)
        self.assertEqual(selected[1],messages[1])
        self.assertEqual(selected[-1],messages[-1])
        self.assertEqual([m['tool_call_id'] for m in selected if m['role']=='tool'],list(map(str,range(8))))
        self.assertLessEqual(projection.last_estimate,116*1024)
        self.assertLess(len(json.dumps(selected)),len(json.dumps(messages)))

    def test_policy_can_be_imported_first_in_a_fresh_process(self):
        result=subprocess.run([sys.executable,'-c',
            'import hybrid.policy; from tycho_nooa import TychoNooaAgent; print("ok")'],
            env={'PATH':'/usr/bin:/bin','PYTHONPATH':str(HERE/'runtime'),
                 'LITELLM_LOCAL_MODEL_COST_MAP':'True'},capture_output=True,text=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.strip(),'ok')


if __name__=='__main__': unittest.main(verbosity=2)
