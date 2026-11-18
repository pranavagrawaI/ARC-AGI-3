"""Restore through original ToolAgent, sandbox and TAAF engine using recorded replies.
No network: every completion is from the archived past-only tape.
"""
from pathlib import Path
import ast, base64, copy, difflib, hashlib, io, json, os, re, sys, threading, time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

class BranchReached(BaseException):pass

def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()

def visible(value):
    """Compare decoded pixels, not version-dependent PNG compression bytes."""
    from PIL import Image
    value=copy.deepcopy(value)
    for m in value:
        if m.get('role')=='tool' and isinstance(m.get('content'),str):
            # Python's hash randomization changes printed set ordering only.
            def stable_set(match):
                parsed=ast.literal_eval(match[0])
                return repr(sorted(parsed)) if isinstance(parsed,set) else match[0]
            m['content']=re.sub(r"\{(?:'[A-Za-z0-9_]+'(?:, )?)+\}",stable_set,m['content'])
            m['content']=re.sub(r"('(run_elapsed_seconds|time_remaining_seconds)': )\d+(?:\.\d+)?",r'\1<CLOCK>',m['content'])
        if isinstance(m.get('content'),list):
            for part in m['content']:
                if part.get('type')=='image_url':
                    im=Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',',1)[1]))).convert('RGB')
                    part['image_url']['url']={'size':im.size,'pixels':hashlib.sha256(im.tobytes()).hexdigest()}
    return value

def restore(name,output,on_branch=None,env_dir=None,code_dir=None,taaf_dir=None):
    cfg=json.loads((HERE/'inputs'/f'{name}.json').read_text())
    for key in list(os.environ):
        if key.startswith(('ARC3_','LOCAL_ANALYZER_','MULTIMODAL_')):os.environ.pop(key)
    os.environ.update(cfg['env'])
    sys.path[:0]=[str(code_dir or cfg['code']),str(taaf_dir or ROOT/'analysis/m2-winners-20261001/audit/dfranzen-repo/tufa-arc-agi-framework/src')]
    import requests
    def no_network(*args,**kw):raise RuntimeError('restoration must not use network')
    requests.post=no_network
    import arc_agi
    from taaf.game_api import GameAPI,ArcadeSpec
    from inference.framework.solver import HarnessSolver,_HarnessGameSession
    from inference.agent import tool_agent as ta
    g=GameAPI(env_name='s5i5-18d95033',arcade_spec=ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE,
              environments_dir=str(env_dir or Path.home()/'projects/arc-agi-3-reference/competition-env/environment_files')))
    g.start_game()
    a=ta.ToolAgent(model='flashnext',base_url='http://restoration.invalid/v1',api_key='unused')
    solver=HarnessSolver(model='flashnext',concurrency=1,max_runtime_s_per_game=7254,save_request_logs=True)
    output.mkdir(parents=True,exist_ok=True)
    session=_HarnessGameSession(solver=solver,game=g,analyzer=a,game_index=0,pass_index=0,
                state_path=output/'state.json',transcript_path=output/'transcript.txt',analysis_html_relpath='analysis.html',
                stop_event=threading.Event(),viewer_data_path=output/'viewer.json')
    idx=0;comparisons=[];began=time.monotonic()
    branched=False
    original_chat=a._chat_completion
    def chat(messages,**kwargs):
        nonlocal idx,branched
        if branched:return on_branch(a,session,messages,kwargs,False,original_chat,cfg)
        expected=cfg['tape'][idx]['request'] if idx<len(cfg['tape']) else cfg['branch']
        vm,ve=visible(messages),visible(expected['messages'])
        differences=[]
        if vm!=ve:
            differences=list(difflib.unified_diff(json.dumps(ve,indent=1).splitlines(),json.dumps(vm,indent=1).splitlines()))[:25]
        comparisons.append({'request':idx,'actual_messages':len(messages),'expected_messages':len(expected['messages']),
                            'messages_identical':messages==expected['messages'],'tools_identical':kwargs['tools']==expected['tools'],
                            'visible_content_identical':vm==ve,'differences':differences,
                            'action_count':session.action_count,'analysis_step':session.analysis_step})
        if idx==len(cfg['tape']):
            (output/'actual-branch.json').write_text(json.dumps({'messages':messages,'tools':kwargs['tools']},indent=2))
            actual=[{'id':h.action.id.name,'data':dict(h.action.data or {})} for h in g.game_run.history]
            expected_actions=[h['action'] for h in cfg['benchmark_prefix']]
            qualified=all(r['visible_content_identical'] and r['tools_identical'] for r in comparisons) and actual==expected_actions
            snapshot={'qualified':qualified,'comparisons':comparisons,'actions_match':actual==expected_actions,
                      'retained_functions':getattr(a,'_kept_functions',{}),'saved_data':getattr(a,'_python_saved_data',{}),
                      'prefix_actions':actual,'level':g.current_state.levels_completed}
            (output/'checkpoint.json').write_text(json.dumps(snapshot,indent=2,default=str)+'\n')
            if on_branch is None:raise BranchReached()
            if not qualified:raise RuntimeError('historical restoration failed qualification')
            branched=True
            # Preserve original archived wire bytes on the first new request.
            messages[:]=copy.deepcopy(cfg['branch']['messages'])
            return on_branch(a,session,messages,kwargs,True,original_chat,cfg)
        reply=cfg['tape'][idx]['reply'];idx+=1
        return ta._ChatCompletionResult(**copy.deepcopy(reply))
    a._chat_completion=chat
    result={'checkpoint':name,'status':'RESTORING'}
    try:
        session.play()
        result['status']='SESSION_ENDED'
    except BranchReached:result['status']='BRANCH_REACHED'
    except BaseException as exc:
        result['status']='STOPPED' if hasattr(exc,'status') else 'FAILED'
        result['error']=repr(exc)
        if hasattr(exc,'status'):result['halt_status']=exc.status
    actual_actions=[{'id':h.action.id.name,'data':dict(h.action.data or {})} for h in g.game_run.history]
    expected_actions=[h['action'] for h in cfg['benchmark_prefix']]
    result.update({'requests_replayed':idx,'seconds':time.monotonic()-began,'comparisons':comparisons,
         'action_count':session.action_count,'actions_match':actual_actions[:len(expected_actions)]==expected_actions,
         'completed_levels':g.current_state.levels_completed,'retained_functions':getattr(a,'_kept_functions',{}),
         'saved_data':getattr(a,'_python_saved_data',{}),'original_system_matches':a._system_prompt==cfg['branch']['messages'][0]['content']})
    (output/'restoration.json').write_text(json.dumps(result,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons','retained_functions','saved_data']},indent=2))
    print('visible content matches',sum(r['visible_content_identical'] for r in comparisons),'/',len(comparisons),'retained',len(result['retained_functions']))
    return a,session,cfg,result

if __name__=='__main__':restore(sys.argv[1],HERE/'evidence'/('restore-'+sys.argv[1]+'-'+sys.argv[2]))
