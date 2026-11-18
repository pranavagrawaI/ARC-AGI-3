"""Real TAAF -> worker -> native tools -> game -> acknowledgement qualification."""
from __future__ import annotations
import asyncio
import hashlib
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'runtime'))
for _key in list(os.environ):
    if _key.startswith(('ARC_LLM_', 'OPENAI_', 'OPENROUTER_', 'ANTHROPIC_', 'NOOA_LLM_')):
        os.environ.pop(_key)
os.environ['LITELLM_LOCAL_MODEL_COST_MAP']='True'
os.environ['TYCHO_SANDBOX_RUNTIME']='host'
os.environ['MULTIMODAL_CONTEXT']='current_grid'
os.environ['MULTIMODAL_UPSCALE']='10'
os.environ['ARC3_MAX_ACTIVE_STREAMS']='0'
os.environ['ARC3_PRIORITY_TAIL_FADE']='0'

import arcengine
from taaf.benchmark import Benchmark
from taaf.game_examples import ExampleGame
from hybrid.solver import HybridSolver
from hybrid.policy import ContextProjection, actor_images
from hybrid import policy
from hybrid.solver import observation
from taaf.deploy_inline import InlineTarget
from inference.framework.solver import HarnessSolver


def script(*steps):
    return json.dumps([{'tool':name,'arguments':arguments} for name,arguments in steps])


def action():
    return ('take_actions',{'actions':[{'action':'UP'}],'rationale':'qualification action'})


def preserve_fixture(game_dir, name, benchmark):
    destination=HERE/'evidence/fixtures'/name
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(game_dir,destination)
    shutil.copy2(benchmark,destination/'benchmark.json')


class HybridQualification(unittest.TestCase):
    def test_worker_error_preserves_completed_compute_without_acting(self):
        from tycho_nooa.protocol import WORKER_PREFIX, PROTOCOL_VERSION
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'worker_entry.py').write_text("import json,sys\n"
                "sys.stdin.readline()\n"
                f"print({WORKER_PREFIX!r}+json.dumps({{'protocol_version':{PROTOCOL_VERSION},'type':'ready','metadata':{{}}}}),flush=True)\n"
                "sys.stdin.readline()\n"
                f"print({WORKER_PREFIX!r}+json.dumps({{'protocol_version':{PROTOCOL_VERSION},'type':'error','sequence':0,'message':'fixture failure after computation','hybrid':{{'completion_tokens':13}}}}),flush=True)\n"
                "sys.stdin.readline()\n")
            bm=Benchmark(games=[ExampleGame(label='error-fixture')],job_dir=root,
                solver=HybridSolver(worker_python=sys.executable,runtime_root=str(root),
                    max_runtime_s_per_game=90,analyzer_timeout=60))
            asyncio.run(bm.run())
            run=bm.game_runs[0]
            self.assertEqual(run.state,'crashed')
            self.assertEqual(sum(run.actions_per_level),0)
            self.assertEqual(run.final_generated_tokens,13)
            game_dir=next((root/'hybrid-games').iterdir())
            self.assertFalse((game_dir/'host-actions.jsonl').exists())
            self.assertEqual(json.loads((game_dir/'compute.json').read_text())['completion_tokens'],13)
            (HERE/'evidence/error-accounting.json').write_text(json.dumps({
                'state':run.state,'engine_actions':0,'completed_tokens_preserved':13},indent=2)+'\n')

    def test_cancel_reaps_blocked_worker_and_finishes_queued_games(self):
        # A stuck model must not keep mutating games after Benchmark returns.
        import inference.agent.tool_agent as donor_agent
        from tycho_nooa.protocol import HOST_PREFIX, WORKER_PREFIX, PROTOCOL_VERSION
        os.environ['ARC3_MAX_ACTIVE_STREAMS']='1'
        donor_agent._PRIORITY_GATE=None
        os.environ.pop('TYCHO_NOOA_SCRIPTED_TOOL_CALLS',None)
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                entry=root/'worker_entry.py'
                entry.write_text("import json,os,sys,time\n"
                    "from pathlib import Path\n"
                    "Path('worker.pid').write_text(str(os.getpid()))\n"
                    "sys.stdin.readline()\n"
                    f"print({WORKER_PREFIX!r}+json.dumps({{'protocol_version':{PROTOCOL_VERSION},'type':'ready','metadata':{{}}}}),flush=True)\n"
                    "time.sleep(90)\n")
                bm=Benchmark(games=[ExampleGame(label='cancel-a'),ExampleGame(label='cancel-b')],
                    job_dir=root,solver=HybridSolver(worker_python=sys.executable,
                    runtime_root=str(root),concurrency=2,max_runtime_s_per_game=90,analyzer_timeout=60))
                async def cancel():
                    task=asyncio.create_task(bm.run())
                    for _ in range(100):
                        if list(root.glob('hybrid-games/*/worker-metadata.json')):
                            break
                        await asyncio.sleep(0.05)
                    else:
                        self.fail('blocked worker never became ready')
                    bm.request_stop()
                    try:
                        await asyncio.wait_for(task,timeout=5)
                    except asyncio.CancelledError:
                        pass
                started=time.monotonic()
                asyncio.run(cancel())
                self.assertTrue(all(run.state=='cancelled' for run in bm.game_runs),
                    [(run.state,run.solver_note) for run in bm.game_runs])
                pids=[int(p.read_text()) for p in root.glob('hybrid-games/*/worker.pid')]
                self.assertTrue(pids)
                self.assertTrue(all(not Path(f'/proc/{pid}').exists() for pid in pids))
                (HERE/'evidence/cancellation.json').write_text(json.dumps({
                    'states':[r.state for r in bm.game_runs],'workers_reaped':len(pids),
                    'seconds':time.monotonic()-started},indent=2)+'\n')
        finally:
            os.environ['ARC3_MAX_ACTIVE_STREAMS']='0'
            donor_agent._PRIORITY_GATE=None

    def test_generated_runner_unpickles_benchmark_and_preserves_results(self):
        os.environ['TYCHO_MODE']='no_world_model'
        os.environ['TYCHO_NOOA_TREATMENT']='control'
        os.environ['TYCHO_NOOA_SCRIPTED_TOOL_CALLS']=script(action(),
            ('write_file',{'path':'notes/level_0_insights.md','content':'Runner fixture completed.'}))
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            bm=Benchmark(games=[ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]],
                label='runner-fixture')],job_dir=root,solver=HarnessSolver(max_actions_per_game=1,
                max_runtime_s_per_game=90,analyzer_timeout=60))
            target=InlineTarget()
            target.actual_run_as_submission=False
            (root/'benchmark.pkl').write_bytes(pickle.dumps(bm))
            (root/'target.pkl').write_bytes(pickle.dumps(target))
            environment=dict(os.environ,HYBRID_RUN_MODE='public25',ARC_LLM_MODEL='scripted-fixture')
            result=subprocess.run([sys.executable,str(HERE/'runtime/runner_entry.py'),
                str(root/'benchmark.pkl'),str(root/'target.pkl')],env=environment,
                capture_output=True,text=True,timeout=30)
            self.assertEqual(result.returncode,0,result.stderr+result.stdout)
            saved=json.loads((root/'tycho-dfranzen/benchmark.json').read_text())
            self.assertEqual(saved['game_runs'][0]['state'],'won')
            (HERE/'evidence/runner.json').write_text(json.dumps({'exit_code':0,
                'state':saved['game_runs'][0]['state'],'generated_entrypoint':True},indent=2)+'\n')

    def test_end_to_end_workspace_action_evidence_and_scribe(self):
        os.environ['TYCHO_MODE']='no_world_model'
        os.environ['TYCHO_NOOA_TREATMENT']='control'
        os.environ['TYCHO_NOOA_SCRIPTED_TOOL_CALLS']=script(
            ('run_python',{'code':"print('native computation reached'); print('segmentation=', [(n['color'], n['pixels']) for n in dfr.segment([[0,1],[0,1]])['nodes']]); import scipy; print('scipy=', scipy.__version__)"}),
            ('write_file',{'path':'notes/reusable.py','content':'def twice(x):\n    return x*2\n'}),
            action(),
            ('run_python',{'code':"exec(open('notes/reusable.py').read()); print(twice(21))"}),
            action(),
            ('write_file',{'path':'notes/level_0_insights.md','content':'Two UPs complete the fixture.'}),
        )
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            game=ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]*2],label='hybrid-fixture')
            bm=Benchmark(label='hybrid-fixture',games=[game],job_dir=root,
                solver=HybridSolver(worker_python=sys.executable,runtime_root=str(HERE/'runtime'),
                    max_runtime_s_per_game=90,max_actions_per_game=2,analyzer_timeout=60))
            asyncio.run(bm.run())
            run=bm.game_runs[0]
            self.assertEqual(run.state,'won',run.solver_note)
            game_dir=next((root/'hybrid-games').iterdir())
            receipts=[json.loads(line) for line in (game_dir/'host-actions.jsonl').read_text().splitlines()]
            self.assertEqual([r['decision']['action'] for r in receipts],['UP','UP'])
            self.assertEqual(receipts[-1]['after']['levels_completed'],1)
            self.assertIsNotNone(receipts[-1]['after']['terminal_frame'])
            workspace=Path(json.loads((game_dir/'worker-metadata.json').read_text())['workspace'])
            self.assertTrue((workspace/'notes/reusable.py').is_file())
            traces=[json.loads(line) for line in (game_dir/'tool-trace.jsonl').read_text().splitlines()]
            self.assertTrue(any('42' in str(t.get('result')) for t in traces))
            self.assertTrue(any("[('0', 2), ('1', 2)]" in str(t.get('result')) for t in traces))
            self.assertTrue(any('scipy=' in str(t.get('result')) for t in traces))
            compute=json.loads((game_dir/'compute.json').read_text())
            self.assertIn('actor',compute['roles'])
            self.assertIn('scribe',compute['roles'])
            self.assertEqual(compute['completion_tokens'],sum(v['completion_tokens'] for v in compute['roles'].values()))
            requests=[json.loads(p.read_text()) for p in (game_dir/'decision-checkpoints/calls').glob('*/request.json')]
            self.assertTrue(requests)
            self.assertTrue(any('data:image/png;base64,' in json.dumps(r['request_body']['messages']) for r in requests))
            self.assertTrue((game_dir/'context-projection.jsonl').is_file())
            preserve_fixture(game_dir,'end-to-end',root/'benchmark.json')
            (HERE/'evidence/end-to-end.json').write_text(json.dumps({'actions':len(receipts),
                'state':run.state,'levels':receipts[-1]['after']['levels_completed'],
                'compute':compute,'provider_images':True,'workspace_computation':42,
                'donor_segmentation':[['0',2],['1',2]],'scipy_computation_available':True},indent=2)+'\n')

    def test_orchestrator_repairs_executable_prediction_over_http(self):
        os.environ.pop('TYCHO_NOOA_SCRIPTED_TOOL_CALLS',None)
        os.environ['TYCHO_MODE']='orchestrator'
        os.environ['TYCHO_NOOA_TREATMENT']='control'
        calls=[]
        counters={'actor':0,'builder':0,'passes':0}
        model_template = """from dataclasses import dataclass
import numpy as np
@dataclass
class State:
    grid: np.ndarray

def init_state(grid0, level): return State(grid0.copy())
def transition(state, action):
    value=int(state.grid[0,0]) + STEP
    return State(np.full_like(state.grid, 14 if value >= 3 else value))
def render(state): return state.grid
def outcome(state): return 'level_complete' if int(state.grid[0,0]) == 14 else 'ongoing'
def actions(state): return [{'action':'UP','row':None,'col':None}]
def heuristic(state): return 0
"""
        def tool(name, arguments):
            return [{'id':'fixture-'+str(len(calls)),'type':'function',
                'function':{'name':name,'arguments':json.dumps(arguments)}}]
        def prediction(value):
            return {'code':"import world_model as wm; import numpy as np; "
                +f"s=wm.init_state(np.full((64,64),{value},dtype=np.int16),0); "
                +"print('PROSPECTIVE', int(wm.render(wm.transition(s,{'action':'UP'}))[0,0]))"}
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):
                pass
            def do_GET(self):
                self.send_response(200);self.end_headers();self.wfile.write(b'healthy')
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                calls.append(body)
                names={tool['function']['name'] for tool in body.get('tools',[])}
                text=json.dumps(body['messages'])
                message={'role':'assistant','content':'',
                         'reasoning_content':'fixture thinking survives replay'}
                if 'take_actions' in names:
                    counters['actor']+=1
                    n=counters['actor']
                    if n in (1,4):
                        counters['passes']+=1
                        counters['builder']=0
                        message['tool_calls']=tool('invoke_builder',{'reason':
                            'Build fixture model' if n==1 else 'Repair: predicted 2 but observed 1'})
                    elif n in (2,5):
                        message['tool_calls']=tool('run_python',prediction(0 if n==2 else 1))
                    else:
                        message['tool_calls']=tool('take_actions',{
                            'actions':[{'action':'UP'}],'rationale':'execute prospective test'})
                elif 'notes/level_0_insights.md' in text:
                    message['tool_calls']=tool('write_file',{
                        'path':'notes/level_0_insights.md','content':'Fixture completed.'})
                else:
                    counters['builder']+=1
                    n=counters['builder']
                    revision=counters['passes']
                    if n==1:
                        message['tool_calls']=tool('write_file',{'path':'world_model.py',
                            'content':model_template.replace('STEP','2' if revision==1 else '1')})
                    elif n==2:
                        message['tool_calls']=tool('write_file',{'path':'notes/world_model.md',
                            'content':f'Fixture model revision {revision}; step is '+('2' if revision==1 else '1')})
                    else:
                        message['content']=f'confidence: fixture\nmodel: revision {revision}\nrecommended_action: UP'
                reply={'id':'fixture-'+str(len(calls)),'object':'chat.completion','created':0,
                    'model':'flashnext','choices':[{'index':0,'message':message,
                    'finish_reason':'tool_calls' if message.get('tool_calls') else 'stop'}],
                    'usage':{'prompt_tokens':100,'completion_tokens':10,'total_tokens':110,
                             'prompt_tokens_details':{'cached_tokens':20}}}
                wire=json.dumps(reply).encode()
                self.send_response(200);self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        os.environ.update(ARC_LLM_MODEL='flashnext',ARC_LLM_API_KEY='loopback-fixture',
            ARC_LLM_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',
            ARC_LLM_CONTEXT_WINDOW=str(128*1024),ARC_LLM_RETRIES='0',ARC_LLM_USE_RESPONSES='0',
            ARC_LLM_TIMEOUT_SECONDS='10',ARC3_HTTP_RETRY_INITIAL_SECONDS='10')
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory)
                game=ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]*3],label='orchestrator-fixture')
                bm=Benchmark(games=[game],job_dir=root,solver=HybridSolver(
                    worker_python=sys.executable,runtime_root=str(HERE/'runtime'),
                    max_runtime_s_per_game=900,max_actions_per_game=3,analyzer_timeout=900))
                asyncio.run(bm.run())
                self.assertEqual(bm.game_runs[0].state,'won',bm.game_runs[0].solver_note)
                game_dir=next((root/'hybrid-games').iterdir())
                compute=json.loads((game_dir/'compute.json').read_text())
                preserve_fixture(game_dir,'orchestrator-http',root/'benchmark.json')
                (HERE/'evidence/orchestrator-requests.json').write_text(json.dumps(calls,indent=2))
                self.assertTrue({'actor','builder','scribe'} <= set(compute['roles']),compute)
                self.assertGreater(compute['cached_tokens'],0)
                self.assertTrue(calls)
                self.assertTrue(all(c['max_tokens']==12*1024 for c in calls))
                self.assertTrue(all(c.get('top_p')==0.95 for c in calls))
                self.assertTrue(all(c.get('top_k')==20 for c in calls))
                self.assertTrue(all(c.get('chat_template_kwargs',{}).get('enable_thinking') for c in calls))
                self.assertTrue(any('data:image/png;base64,' in json.dumps(c['messages']) for c in calls))
                self.assertTrue(any(m.get('reasoning_content')=='fixture thinking survives replay'
                    for c in calls for m in c['messages']))
                traces=[json.loads(line) for line in (game_dir/'tool-trace.jsonl').read_text().splitlines()]
                prospective=[t for t in traces if 'PROSPECTIVE 2' in str(t.get('result'))]
                self.assertEqual(len(prospective),2, traces)
                receipts=[json.loads(line) for line in (game_dir/'host-actions.jsonl').read_text().splitlines()]
                self.assertEqual(len(receipts),3)
                self.assertEqual([r['after']['decision_frame'][0][0] for r in receipts],[1,2,14])
                self.assertEqual([t['sequence'] for t in prospective],[0,1])
                self.assertTrue(all(t['status']=='complete' for t in traces if t['name']=='invoke_builder'))
                for prediction_trace in prospective:
                    action_trace=next(t for t in traces if t['name']=='take_actions'
                        and t['sequence']==prediction_trace['sequence'])
                    self.assertLess(prediction_trace['call_index'],action_trace['call_index'])
                self.assertEqual(counters['passes'],2)
                self.assertLess(compute['roles']['builder']['calls'],12)
                workspace=Path(json.loads((game_dir/'worker-metadata.json').read_text())['workspace'])
                self.assertIn(' + 1', (workspace/'world_model.py').read_text())
                preserve_fixture(game_dir,'orchestrator-http',root/'benchmark.json')
                (HERE/'evidence/orchestrator-http.json').write_text(json.dumps({
                    'endpoint':'loopback only','calls':len(calls),'compute':compute,
                    'request_settings_verified':True,'builder_requests':counters['passes'],
                    'prospective_predictions':[2,2], 'expected_observed_values':[1,2,14],
                    'scope':'Scripted provider proves execution; not learned model capability.'},indent=2)+'\n')
        finally:
            server.shutdown();server.server_close();thread.join(timeout=2)
            for key in list(os.environ):
                if key.startswith('ARC_LLM_'):os.environ.pop(key)
            os.environ.pop('ARC3_HTTP_RETRY_INITIAL_SECONDS',None)

    def test_trickling_http_bodies_cannot_outlive_host_deadline(self):
        os.environ.pop('TYCHO_NOOA_SCRIPTED_TOOL_CALLS',None)
        os.environ['TYCHO_MODE']='no_world_model'
        os.environ['TYCHO_NOOA_TREATMENT']='control'
        results=[]
        for status in (200,500):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as directory:
                stop=threading.Event()
                requests=[]
                root=Path(directory)
                class Handler(BaseHTTPRequestHandler):
                    def log_message(self,*args): pass
                    def do_POST(self):
                        requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                        self.send_response(status)
                        self.send_header('Content-Type','application/json')
                        self.send_header('Content-Length','100000')
                        self.end_headers()
                        try:
                            # Frequent bytes defeat an inactivity-only timeout.
                            while not stop.wait(0.1):
                                self.wfile.write(b' ');self.wfile.flush()
                        except (BrokenPipeError,ConnectionResetError): pass
                server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
                thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
                os.environ.update(ARC_LLM_MODEL='flashnext',ARC_LLM_API_KEY='loopback-fixture',
                    ARC_LLM_BASE_URL=f'http://127.0.0.1:{server.server_port}/v1',
                    ARC_LLM_RETRIES='0',ARC_LLM_USE_RESPONSES='0',ARC_LLM_TIMEOUT_SECONDS='30')
                try:
                    solver=HybridSolver(worker_python=sys.executable,runtime_root=str(HERE/'runtime'),
                        max_runtime_s_per_game=5,analyzer_timeout=3)
                    bm=Benchmark(games=[ExampleGame(label=f'slow-{status}')],job_dir=root,solver=solver)
                    started=time.monotonic()
                    import hybrid.solver as solver_module
                    worker_type=solver_module.HybridLocalSolver
                    processes=[]
                    def tracked_worker(**kwargs):
                        worker=worker_type(**kwargs)
                        processes.append(worker._process)
                        return worker
                    with patch.object(solver_module,'HybridLocalSolver',side_effect=tracked_worker):
                        asyncio.run(bm.run())
                    elapsed=time.monotonic()-started
                    self.assertTrue(processes)
                    self.assertTrue(all(process.poll() is not None for process in processes))
                    self.assertLess(elapsed,10)
                    self.assertEqual(len(requests),1)
                    self.assertEqual(bm.game_runs[0].state,'crashed')
                    self.assertEqual(sum(bm.game_runs[0].actions_per_level),0)
                    self.assertFalse(solver._workers)
                    game_dir=next((root/'hybrid-games').iterdir())
                    self.assertFalse((game_dir/'host-actions.jsonl').exists())
                    preserve_fixture(game_dir,f'slow-http-{status}',root/'benchmark.json')
                    results.append({'http_status':status,'elapsed_seconds':elapsed,
                        'requests':len(requests),'engine_actions':0,
                        'completed_usage':None,'provider_cancellation':'not claimed'})
                finally:
                    stop.set();server.shutdown();server.server_close();thread.join(timeout=2)
                    for key in list(os.environ):
                        if key.startswith('ARC_LLM_'):os.environ.pop(key)
        (HERE/'evidence/slow-http.json').write_text(json.dumps(results,indent=2)+'\n')

    def test_block_trim_is_stable_and_keeps_tool_pairs(self):
        projection=ContextProjection(window=9000,drain=3000)
        messages=[{'role':'system','content':'fixed instructions'}]
        for index in range(12):
            messages += [{'role':'user','content':str(index)+'x'*1500},
                         {'role':'assistant','content':None,'tool_calls':[{'id':str(index)}]},
                         {'role':'tool','tool_call_id':str(index),'content':'y'*1500}]
        selected=projection.project(messages,output=1000)
        self.assertLess(len(selected),len(messages))
        self.assertEqual(selected[1]['role'],'user')
        next_request=projection.project(messages+[{'role':'user','content':'next'}],output=1000)
        self.assertEqual(next_request[:-1],selected)
        self.assertLessEqual(projection.last_estimate,8000)

    def test_image_accounting_counts_dimensions_not_base64_text(self):
        game=ExampleGame();game.start_game()
        policy.OBSERVATION=observation(game)
        images=actor_images()
        self.assertEqual(len(images),1)
        url=images[0]['image_url']['url']
        self.assertGreater(len(url),100)
        projection=ContextProjection()
        text,vision=__import__('inference.agent.tool_agent',fromlist=['_split_messages_for_estimate'])._split_messages_for_estimate(
            [{'role':'user','content':images}])
        self.assertEqual(vision,402)
        self.assertNotIn('base64',json.dumps(text))


if __name__=='__main__':
    unittest.main(verbosity=2)
