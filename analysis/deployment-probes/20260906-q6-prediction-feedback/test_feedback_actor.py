"""Exercise feedback and gating through real CodeAct Actor calls."""
import asyncio
import json
from pathlib import Path
import pytest
from nooa.unifiedllm import FakeLLMClient
from test_payload import response
from solver_agent import MdArcSolverAgent

BAD = '''import numpy as np
def encode(g):
    return {"x": int(np.where(g == 12)[1].min()), "right": int(np.where(g == 11)[1].max())}
def predict(z,a):
    return {"x": z["x"]+1, "right": z["right"]-2}
'''
GOOD = BAD.replace('z["right"]-2', 'z["right"]')


def test_codeact_contradiction_repair_fresh_prediction_and_batch(tmp_path, monkeypatch):
    monkeypatch.setenv('ARC_STALL_REVIEW', '0')
    async def run():
        path = [{'expect':'move', 'z':None}]
        code = "self.submit_actions(['RIGHT'], rationale='predict: shift right', predicted_path="+repr(path)+")"
        fix = ("assert self.prediction_feedback()['status'] == 'mismatch'\n"
               "assert self.submit_actions(['RIGHT','RIGHT'], rationale='predict: shift right', predicted_path="+repr(path*2)+").startswith('REJECTED')\n"
               "self.write_helper('world_model.py', "+repr(GOOD)+")\n" + code)
        batch = ("assert self.prediction_feedback()['status'] == 'matched'\n"
                 "self.submit_actions(['RIGHT','RIGHT'], rationale='predict: two shifts', predicted_path="+repr(path*2)+")")
        actor = MdArcSolverAgent(llm=FakeLLMClient(scripted_responses=[response(code), response(fix), response(batch)]),run_dir=tmp_path)
        actor.write_helper('world_model.py', BAD)
        async def frame(turn, grid):
            state = {'turn':turn,'step':turn,'grid_rows':[''.join(format(v,'x') for v in grid[0])],
                     'state':'NOT_FINISHED','levels_completed':0,'available_actions':['RIGHT'], 'action_results':[]}
            with actor._states_path.open('a') as stream:stream.write(json.dumps(state)+'\n')
            await actor.handle({'user_messages':['Use latest_state and finish the next test.']})
        def observed(turn, before, after):
            with (actor.ipc_dir/'transitions.jsonl').open('a') as stream:
                stream.write(json.dumps({'turn':turn,'step':turn+1,'action':'RIGHT',
                                        'before':{'grid':before,'levels_completed':0},
                                        'after':{'grid':after,'levels_completed':0}})+'\n')
        try:
            g0,g1,g2 = [[12,0,0,11]],[[0,12,0,11]],[[0,0,12,11]]
            await frame(0,g0)
            observed(0,g0,g1)
            await frame(1,g1)
            observed(1,g1,g2)
            await frame(2,g2)
            entries=[json.loads(line) for line in actor._actions_path.read_text().splitlines()]
            assert [len(entry['actions']) for entry in entries] == [1,1,2]
            commits=[json.loads(line) for line in (actor.ipc_dir/'model_predictions.jsonl').read_text().splitlines()]
            assert commits[0]['model_sha256'] != commits[1]['model_sha256']
            assert commits[1]['model_sha256'] == commits[2]['model_sha256']
        finally:
            await actor.queue_manager.shutdown()
    asyncio.run(run())
