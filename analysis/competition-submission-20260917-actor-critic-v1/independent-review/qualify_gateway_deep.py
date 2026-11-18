"""Exact notebook bytes, official Agent.main and official REST gateway, synthetic games/model.
Gateway parquet writer is a local contract fixture, not Kaggle's private sidecar.
"""
import argparse,hashlib,json,os,shutil,subprocess,sys,threading
from pathlib import Path
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
OFFICIAL=ROOT/'analysis/deployment-probes/20260912-ls20-sparse-reasoning-public-v1/raw/ARC-AGI-3-Agents'

DEPENDENCY_GUARD = """import sys,runpy,json,hashlib
from pathlib import Path
import tycho_nooa,nooa,arc_agi,arcengine
modules=[tycho_nooa,nooa,arc_agi,arcengine]
origins={m.__name__:str(Path(m.__file__).resolve()) for m in modules}
assert all('site-packages' in Path(p).parts for p in origins.values()), origins
root=Path(sys.argv[1]).parent
manifest=json.loads((root/'expected-solver-sources.json').read_text())
site=Path(tycho_nooa.__file__).parent.parent
for row in manifest['packaged_sources']:
    assert hashlib.sha256((site/row['wheel_path']).read_bytes()).hexdigest()==row['sha256'], row['wheel_path']
(root/'dependency-origins.json').write_text(json.dumps(dict(python=sys.version,origins=origins,verified_solver_sources=len(manifest['packaged_sources'])),indent=2))
runpy.run_path(sys.argv[1],run_name='__main__')
"""

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--games',type=int,default=110);parser.add_argument('--mode',choices=['normal','reserve','critic-cap','deep'],default='normal')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    competition=out/'competition';games=competition/'environment_files';games.mkdir(parents=True)
    official=competition/'ARC-AGI-3-Agents'
    for name in json.loads((PACKAGE/'official-hashes.json').read_text()):
        dst=official/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(OFFICIAL/name,dst)
    for i in range(args.games):
        name=f'f{i:03d}';d=games/name/'00000001';d.mkdir(parents=True)
        (d/'metadata.json').write_text(json.dumps(dict(game_id=name+'-00000001',title=name,default_fps=30,tags=['fixture'],baseline_actions=[2,2])))
        (d/(name+'.py')).write_text(f'''from arcengine import ARCBaseGame,Level,Sprite
class {name.capitalize()}(ARCBaseGame):
    def __init__(self,seed=0):
        self.n=0
        super().__init__(game_id={name!r},levels=[Level(sprites=[Sprite(pixels=[[3]],name="piece")],grid_size=(8,8)),Level(sprites=[Sprite(pixels=[[4]],name="piece")],grid_size=(8,8))],available_actions=[1],seed=seed)
    def on_set_level(self,level): self.n=0
    def step(self):
        self.n+=1
        self.current_level.get_sprites()[0].set_position(self.n, 0)
        if self.n==2:self.next_level()
        self.complete_action()
''')
    shutil.copyfile(ROOT/'deployment/kaggle/tycho-nooa-full-wheelhouse-v11/solver-source-manifest.json',out/'expected-solver-sources.json')
    materialized=out/'notebook';materialized.mkdir()
    (materialized/'working').mkdir()
    os.environ['ARC_DISABLE_DOTENV']='1'
    from arc_agi import Arcade,OperationMode
    from arc_agi.server import create_app
    from werkzeug.serving import make_server
    arcade=Arcade(operation_mode=OperationMode.OFFLINE,environments_dir=str(games))
    def closed(card):
        import pandas as pd
        raw=card.model_dump(mode='json');(out/'gateway-scorecard.json').write_text(json.dumps(raw,indent=2))
        rows=[[f'{e["id"]}_0',e['id'],True,e['score']] for e in raw['environments']]
        pd.DataFrame(rows,columns=['row_id','game_id','end_of_game','score']).to_parquet(materialized/'working/submission.parquet',index=False)
    app,_=create_app(arcade,competition_mode=True,on_scorecard_close=closed)
    gateway=make_server('127.0.0.1',0,app,threaded=True)
    gt=threading.Thread(target=gateway.serve_forever,daemon=True);gt.start()
    requests=[]
    class Model(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def do_POST(self):
            r=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            actor=bool(r.get('tools'))
            requests.append(dict(purpose='actor' if actor else 'critic',request=r))
            with (out/'model-requests.jsonl').open('a') as f:f.write(json.dumps(requests[-1])+'\n')
            msg={'role':'assistant','content':json.dumps(dict(claim_ref='',question='',reason=''))}
            if not actor and args.mode=='critic-cap':msg['content']='invalid critic output fixture'
            deep=actor and args.mode=='deep' and sum(x['purpose']=='actor' for x in requests)==1
            if actor:
                msg.update(content=None,tool_calls=[dict(id=f'call{len(requests)}',type='function',function=dict(name='execute_python',arguments=json.dumps(dict(code=('r = await self.deep_investigate("Which rule moves the piece?", ["decision_packet"]); print(r["id"])' if deep else 'print(self.model_grid().shape)' if args.mode not in ('normal','deep') else 'self.submit_actions(["UP"], rationale="predict: fixture step")')))))])
            data=json.dumps(dict(id=f'fixture{len(requests)}',object='chat.completion',created=1,model='controlled-local',choices=[dict(index=0,finish_reason='tool_calls' if actor else 'stop',message=msg)],usage=dict(prompt_tokens=100,completion_tokens=50,total_tokens=150))).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    model=ThreadingHTTPServer(('127.0.0.1',0),Model);mt=threading.Thread(target=model.serve_forever,daemon=True);mt.start()
    notebook=json.loads((PACKAGE/'submission.ipynb').read_text());source='\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type']=='code')
    script=out/'execute-notebook.py';script.write_text(source)
    env={k:v for k,v in os.environ.items() if not k.startswith(('ARC_','NOOA_','SUBMISSION_','TYCHO_','KAGGLE_'))}
    env.update(PYTHONPATH=str(Path(__file__).resolve().parents[1]/('checks/local-deps-cp312' if sys.version_info[:2]==(3,12) else 'checks/local-deps')),ARC_DISABLE_DOTENV='1',PYTHONDONTWRITEBYTECODE='1',KAGGLE_IS_COMPETITION_RERUN='true',ARC_LOCAL_FIXTURE='1',ARC_LOCAL_ROOT=str(materialized),ARC_FIXTURE_COMPETITION=str(competition),ARC_FIXTURE_MODEL_URL=f'http://127.0.0.1:{model.server_port}/v1',SUBMISSION_GATEWAY_BASE_URL=f'http://127.0.0.1:{gateway.server_port}')
    if args.mode=='reserve':env['ARC_FIXTURE_ACTOR_CAP']='1'
    try:
        with (out/'stdout.log').open('w') as stdout,(out/'stderr.log').open('w') as stderr:
            result=subprocess.run([sys.executable,'-c',DEPENDENCY_GUARD,str(script)],cwd=out,env=env,stdout=stdout,stderr=stderr,timeout=1800)
        assert result.returncode==0, f'notebook exit {result.returncode}; see {out}/stderr.log'
        q=materialized/'working/qualification';r=json.loads((q/'launcher-result.json').read_text())
        assert len(r['games'])==args.games
        if args.mode=='deep':
            pass
        elif args.mode=='normal':
            assert all(g['state']=='WIN' and g['levels_completed']==2 for g in r['games'])
        else:
            assert all(g['stop_reason']=='model_admission_budget' for g in r['games'])
            assert sum(x['purpose']=='actor' for x in requests)==args.games*(12 if args.mode=='critic-cap' else 1)
            assert sum(x['purpose']=='critic' for x in requests)==args.games*(4 if args.mode=='critic-cap' else 1)
        assert sum(x['purpose']=='actor' for x in requests)>=args.games
        assert sum(x['purpose']=='critic' for x in requests)>=args.games
        sys.path.insert(0,str(materialized));import official_launcher_runtime as outer
        deep_records=[json.loads(l) for f in q.rglob('deep_investigations.jsonl') for l in f.read_text().splitlines()]
        ledgers=[json.loads(l) for f in q.glob('request-budget-*.jsonl') for l in f.read_text().splitlines()]
        summary=dict(server_requests_total=len(requests),server_requests_with_tools=sum(x['purpose']=='actor' for x in requests),server_requests_without_tools=sum(x['purpose']=='critic' for x in requests),shared_ledger_rows=len(ledgers),shared_ledger_actor=sum(x['request_purpose']=='actor' for x in ledgers),shared_ledger_critic=sum(x['request_purpose']=='critic' for x in ledgers),deep_investigation_records=len(deep_records),deep_max_tokens_requested=sorted({x['request'].get('max_tokens') for x in requests if x['purpose']=='critic' and x['request'].get('max_tokens')!=192}),deep_thinking_enabled=[x['request'].get('chat_template_kwargs') for x in requests if x['purpose']=='critic' and x['request'].get('max_tokens')!=192],expected_launcher_solver_requests=outer.expected_launcher_solver_requests(r))
        (out/'DEEP-SUMMARY.json').write_text(json.dumps(summary,indent=2))
        print(json.dumps(summary,indent=2))
        try:
            facts=outer.reconcile_launcher_request_count(r,dict(chat_completion_accesses=1,successful_chat_completion_accesses=1),dict(chat_completion_accesses=1+len(requests),successful_chat_completion_accesses=1+len(requests),successful_freeze_gc_accesses=1,ready_marker_count=1))
        except RuntimeError as e:
            (out/'RECONCILE-ERROR.txt').write_text(str(e));print('RECONCILE RuntimeError:',str(e)[:400])
            if args.mode=='deep':
                (out/'RESULT.json').write_text(json.dumps(dict(status='reconciliation_failed',**summary),indent=2));return
            raise
        for delta in (-1,1):
            try:outer.reconcile_launcher_request_count(r,dict(chat_completion_accesses=1,successful_chat_completion_accesses=1),dict(chat_completion_accesses=1+len(requests)+delta,successful_chat_completion_accesses=1+len(requests),successful_freeze_gc_accesses=1,ready_marker_count=1))
            except RuntimeError:pass
            else:raise AssertionError('Modified server count accepted')
        for f in q.rglob('worker-metadata.json'):
            pid=json.loads(f.read_text())['agent_pid']
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise AssertionError(f'worker {pid} survived')
        (out/'RESULT.json').write_text(json.dumps(dict(status='passed',games=args.games,actor=sum(x['purpose']=='actor' for x in requests),critic=sum(x['purpose']=='critic' for x in requests),supervisor=facts,scope='local synthetic gateway and model; no GPU or Kaggle'),indent=2))
    finally:
        model.shutdown();model.server_close();mt.join();gateway.shutdown();gateway.server_close();gt.join()

if __name__=='__main__':main()
