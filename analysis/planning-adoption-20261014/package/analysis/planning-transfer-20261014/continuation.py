"""Restore archived actor state, then apply one named experimental variant.

Default mode is offline restoration. Live mode is explicit and uses the earlier
qualified absolute-deadline transport, with no retries. This script never starts
a server, launches a GPU, writes the submission queue, or submits a competition.
"""
import argparse
import copy
import dataclasses
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ARCHIVE = ROOT / 'analysis/counterexample-continuations-v3-20261013'
TOOLS = ROOT / 'solvers/m2_planning_tools'
sys.path[:0] = [str(ARCHIVE), str(TOOLS)]
import restore
from worker import Transport as ArchivedTransport, Stop, append, write
from variants import VARIANTS, guidance, wire_messages


class Transport(ArchivedTransport):
    """Retain the qualified transport and reject boolean usage at its boundary."""
    def post(self,url,**kwargs):
        kwargs=copy.deepcopy(kwargs)
        kwargs['json']['messages']=wire_messages(kwargs['json'].get('messages',[]))
        before=self.tokens
        reserved=min(int(kwargs['json'].get('max_tokens') or 12288),self.token_cap-before)
        response=super().post(url,**kwargs)
        usage=response.json()['usage']['completion_tokens']
        # Python bool subclasses int; the archived transport accepted it.
        if type(usage) is not int:
            self.tokens=before
            self.unresolved={'call':self.calls,'reserved_tokens':reserved}
            append(self.out/'wire.jsonl',{'event':'failure','call':self.calls,
                   'error':'invalid noninteger completion_tokens','unresolved':self.unresolved})
            raise Stop('invalid_usage')
        return response


def identity_files(checkpoints):
    files={Path(__file__),TOOLS/'variants.py',ARCHIVE/'restore.py',ARCHIVE/'worker.py'}
    files.update((TOOLS/'overlay').rglob('*.py'))
    for checkpoint in checkpoints:
        config_path=ARCHIVE/'inputs'/(checkpoint+'.json')
        files.add(config_path)
        config=json.loads(config_path.read_text())
        code=Path(os.environ.get('PLANNING_CODE_'+checkpoint.upper(),config['code']))
        files.update(code.rglob('*.py'))
    taaf=Path(os.environ.get('PLANNING_TAAF_DIR',ROOT/'analysis/m2-winners-20261001/audit/dfranzen-repo/tufa-arc-agi-framework/src'))
    files.update(taaf.rglob('*.py'))
    receipt_path=ROOT/'analysis/s5i5-inspection-20261012/engine-replay.json'
    receipt=json.loads(receipt_path.read_text())
    engine=Path(os.environ.get('PLANNING_ENGINE_SOURCE',receipt['engine_source']))
    if hashlib.sha256(engine.read_bytes()).hexdigest()!=receipt['engine_sha256']:
        raise RuntimeError('Native engine identity changed')
    files.update({receipt_path,engine})
    return sorted(path for path in files if '__pycache__' not in str(path))


def install(variant, saved_data):
    """Swap only the sandbox dispatch after original restoration qualifies."""
    import inference.utils
    from inference.agent import tool_agent
    root = TOOLS / 'overlay/inference'
    os.environ['ARC3_PLANNING_VARIANT'] = variant
    for name in ('bounded_search', 'prospective_probe'):
        qualified = 'inference.utils.' + name
        spec = importlib.util.spec_from_file_location(qualified, root / 'utils' / (name+'.py'))
        module = importlib.util.module_from_spec(spec)
        sys.modules[qualified] = module; spec.loader.exec_module(module)
        setattr(inference.utils, name, module)
    name = 'planning_transfer_sandbox'
    spec = importlib.util.spec_from_file_location(name, root / 'agent/python_tool_sandbox.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    def dispatch(**kwargs):
        # The archived raw control predates saved-data ownership. Every variant
        # receives the same host-owned per-episode store through this adapter.
        kwargs['saved_data']=saved_data
        return module.run_sandboxed_python(**kwargs)
    tool_agent.run_sandboxed_python = dispatch


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    state = {'mode':args.mode,'variant':args.variant,'checkpoint':args.checkpoint,'seed':args.seed,
             'status':'RESTORING','new_model_calls':0,'tokens':0,'unresolved':None,'new_native_actions':0}
    transport = None
    sessions = []
    fixture_sent = False
    protected = {str(p):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in identity_files([args.checkpoint])}
    write(out/'identity.json',{'protected_files':protected,'variant':args.variant,
          'limits':{'tokens':32768,'model_calls':32,'native_actions':150,'episode_seconds':900,'request_seconds':120},
          'intervention':'Common sandbox dispatch and host-owned data store, plus capability-specific system addendum after original restoration'})

    def branch(agent, session, messages, kwargs, first, original_chat, config):
        nonlocal transport, fixture_sent
        from inference.agent import tool_agent
        import requests
        import arcengine
        if first:
            sessions.append(session)
            if not hasattr(agent,'_python_saved_data'):
                agent._python_saved_data={}
            install(args.variant,agent._python_saved_data)
            extra = guidance(args.variant)
            before = copy.deepcopy(messages)
            messages[0]['content'] += extra
            agent._system_prompt += extra
            write(out/'first-request.json',{'messages':wire_messages(messages),'tools':kwargs['tools']})
            write(out/'intervention.json',{'original_messages_sha256':restore.digest(before),
                  'system_addendum':extra,'all_other_messages_identical':messages[1:]==before[1:],
                  'original_system_prefix_preserved':messages[0]['content']==before[0]['content']+extra})
            session.started_at=time.monotonic(); session.solver.max_runtime_s_per_game=900
            original_step=session.game._execute_action
            def step(action):
                if state['new_native_actions']>=150:
                    raise Stop('action_budget')
                state['new_native_actions']+=1
                frame=original_step(action)
                if frame.levels_completed>=2:
                    frame.raw.state=arcengine.GameState.WIN
                return frame
            session.game._execute_action=step
            if args.mode=='live':
                transport=Transport(out,args.url,args.seed)
                requests.post=transport.post
                agent._model=dataclasses.replace(agent._model,base_url=args.url)
                transport.set_deadline(transport.deadline)
        if args.mode=='restore':
            raise Stop('restored')
        if args.mode=='fixture':
            if fixture_sent:
                raise Stop('fixture_complete')
            fixture_sent=True
            code=Path(args.fixture).read_text()
            call={'id':'fixture','type':'function','function':{'name':'python','arguments':json.dumps({'code':code})}}
            return tool_agent._ChatCompletionResult(message={'role':'assistant','content':None,'tool_calls':[call]},
                finish_reason='tool_calls',usage={'completion_tokens':1,'prompt_tokens':1,'total_tokens':2})
        if transport.tokens>=transport.token_cap:
            raise Stop('token_budget')
        if transport.calls>=transport.request_cap:
            raise Stop('request_budget')
        return original_chat(messages,tools=kwargs['tools'],request_timeout_seconds=120,
            max_tokens=min(agent._max_output_tokens,transport.token_cap-transport.tokens))

    try:
        agent, session, config, restored = restore.restore(args.checkpoint,out,on_branch=branch,
            env_dir=os.environ.get('PLANNING_ENV_DIR'),
            code_dir=os.environ.get('PLANNING_CODE_'+args.checkpoint.upper()),
            taaf_dir=os.environ.get('PLANNING_TAAF_DIR'))
        state.update(restoration_status=restored['status'],status=restored.get('halt_status',restored['status']),
                     qualified_restoration=bool(restored['comparisons']) and all(r['visible_content_identical'] and r['tools_identical'] for r in restored['comparisons']) and restored['actions_match'])
        if 'error' in restored:
            state['error']=restored['error']
        if session.game.current_state.levels_completed>=2:
            state['status']='solved'
    except Exception as error:
        state.update(status='FAILED',error=type(error).__name__+': '+str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        if transport:
            state.update(new_model_calls=transport.calls,tokens=transport.tokens,unresolved=transport.unresolved)
        if sessions:
            session=sessions[0]
            run=session.game.game_run
            state.update(levels_completed=session.game.current_state.levels_completed,score=run.final_score,
                         saved_data=agent._python_saved_data,actions_per_level=run.actions_per_level)
            write(out/'actions.json',[{'id':h.action.id.name,'data':dict(h.action.data or {})} for h in run.history])
        state['protected_inputs_unchanged']=all(hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
                                               for path,expected in protected.items())
        write(out/'result.json',state)
        print(json.dumps({k:v for k,v in state.items() if k!='saved_data'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',choices=['b1','control'],required=True)
    parser.add_argument('--variant',choices=VARIANTS,default='control')
    parser.add_argument('--mode',choices=['restore','fixture','live'],default='restore')
    parser.add_argument('--seed',type=int,default=1701)
    parser.add_argument('--output',required=True)
    parser.add_argument('--fixture')
    parser.add_argument('--url',default='http://127.0.0.1:8001/v1')
    args=parser.parse_args()
    if args.mode=='fixture' and not args.fixture:
        parser.error('fixture mode requires --fixture')
    run(args)
