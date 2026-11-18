"""Offline accounting and context audit; parses proposals without executing them."""
import argparse
import ast
import hashlib
import json
from pathlib import Path


def audit(out):
    read=lambda n:json.loads((out/n).read_text())
    frozen=read('FROZEN.json')
    hashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in frozen['files'].items()}
    control=read('00-control.request.json');reviewed=read('02-reviewed.request.json');critic=read('01-critic.request.json')
    captured=read('captured-request.json');review=read('REVIEW.json')
    checks={'frozen_inputs_unchanged':all(hashes.values()),
        'control_messages_equal_capture':control['messages']==captured['messages'],
        'reviewed_messages_equal_control_plus_one':reviewed['messages'][:-1]==control['messages'],
        'actor_nonmessage_settings_equal':{k:v for k,v in control.items() if k!='messages'}=={k:v for k,v in reviewed.items() if k!='messages'},
        'critic_has_complete_capture':json.loads(critic['messages'][1]['content'])['actor_request_as_evidence']==captured,
        'critic_tools_absent':not critic.get('tools'),
        'reviewed_suffix_matches_frozen_template':reviewed['messages'][-1]['content']==(out/'actor-response-template.txt').read_text().format(
            review_id=review['binding']['review_id'],critique=json.dumps(review['critique']))}
    outputs={};metrics=[]
    for name in ('00-control','01-critic','02-reviewed'):
        response=read(name+'.response.json');message=response['choices'][0]['message'];usage=response['usage']
        metrics.append({'name':name,'seconds':read(name+'.timing.json')['seconds'],
            'prompt_tokens':usage['prompt_tokens'],'completion_tokens':usage['completion_tokens'],
            'cached_tokens':usage.get('prompt_tokens_details',{}).get('cached_tokens'),
            'reasoning_tokens':usage.get('completion_tokens_details',{}).get('reasoning_tokens'),
            'cost':usage['cost'],'finish_reason':response['choices'][0]['finish_reason']})
        if name=='01-critic':continue
        cells=[];actions=[];printed=[]
        for t in message.get('tool_calls') or []:
            args=json.loads(t['function']['arguments']);code=args.get('code','');cells.append(code)
            for node in ast.walk(ast.parse(code)):
                if not isinstance(node,ast.Call):continue
                if isinstance(node.func,ast.Attribute) and node.func.attr=='submit_actions' and node.args:
                    actions.append(ast.literal_eval(node.args[0]))
                if isinstance(node.func,ast.Name) and node.func.id=='print' and node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):
                    printed.append(node.args[0].value)
        outputs[name]={'assistant_text':message.get('content'),'proposed_cells':cells,
            'literal_action_proposals':actions,'literal_printed_statements':printed,
            'printed_statement_words':sum(len(s.split()) for s in printed),'code_executed':False}
    result={'checks':checks,'metrics':metrics,'total_cost':sum(m['cost'] for m in metrics),
            'scope':'Mechanical audit and emitted proposals only; semantic grading separate'}
    (out/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'ACTOR-PROPOSALS.json').write_text(json.dumps(outputs,indent=2)+'\n')
    if not all(checks.values()):raise ValueError('Mechanical audit failed')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    audit(p.parse_args().output)
