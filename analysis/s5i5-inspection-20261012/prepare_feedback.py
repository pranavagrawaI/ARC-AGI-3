"""Freeze a past-only counterexample packet at an existing request boundary.

Preparation only: no request is sent, no inference budget is spent.
"""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
M=json.loads((OUT/'model-check.json').read_text())
R=json.loads((OUT/'records.json').read_text())
path=ROOT/'solvers/m2_b1/experiment/pair-1/b1-r1/output/s5i5-18d95033_p0_requests.jsonl'
selected=None
with path.open() as f:
    for line_number,line in enumerate(f,1):
        req=json.loads(line)
        if req.get('analysis_step')!=19 or req.get('action')!=43:continue
        messages=req['messages']
        flat=json.dumps(messages)
        if 'def growS' not in flat or 'yellow grows from init:' not in flat:continue
        if 'nodes 538647' in flat:continue
        if messages[-1]['role'] not in ['tool','user']:continue
        selected=(line_number,req);break
assert selected,'No causally valid request boundary found'
line_number,control=selected
row=M['available_when_failed_predictor_was_written']['row']
assert row['local_action']==16
feedback=(
    'Historical consistency check of your current simulator, before relying on its search results:\n'
    'At an already observed L2 transition, MOUSE(row=57,col=27) extended sky. '
    'Predecessor: '+json.dumps(row['input_state'],sort_keys=True)+'\n'
    'Your growS predecessor prediction: '+json.dumps(row['outcomes']['failed_push_model']['prediction'],sort_keys=True)+'\n'
    'Recorded successor: '+json.dumps(row['actual_successor'],sort_keys=True)+'\n'
    'The green bar moved upward with yellow; your predictor left green at its old row. '
    'This is a contradiction in the candidate simulator, not evidence that the puzzle goal is impossible. '
    'Check the candidate against already observed transitions before using a no-solution result to reject a goal. '
    'Report the number of selected transitions, actual comparisons, disagreements, rejected successors and unmodeled actions. '
    'Zero comparisons must not count as a successful check. Do not claim that historical agreement proves unseen transitions correct.'
)
treatment={**control,'messages':control['messages']+[{'role':'user','content':feedback}]}
for name,data in [('continuation-control-request.json',control),('continuation-feedback-request.json',treatment)]:
    (OUT/name).write_text(json.dumps(data,indent=2)+'\n')
packet={'source_request_log':str(path.relative_to(ROOT)),'source_line':line_number,
        'source_request_log_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'analysis_step':control['analysis_step'],'action':control['action'],
        'feedback':feedback,'evidence_local_action':16,
        'available_history_actions':22,'future_frames_included':False,'sent':False,
        'control_request_sha256':hashlib.sha256((OUT/'continuation-control-request.json').read_bytes()).hexdigest(),
        'treatment_request_sha256':hashlib.sha256((OUT/'continuation-feedback-request.json').read_bytes()).hexdigest(),
        'next_test':{'contrast':'Original-model ordinary continuation versus same request plus one historical counterexample',
                     'measures':['Repair cost in generated tokens','Prediction accuracy on held-out transitions','Correctness and coverage of the self-check'],
                     'limitation':'Packet preparation and local detection do not demonstrate incremental feedback benefit; no inference run authorized by this packet.'}}
(OUT/'feedback-packet.json').write_text(json.dumps(packet,indent=2)+'\n')
assert treatment['messages'][:-1]==control['messages']
assert {k:v for k,v in treatment.items() if k!='messages'}=={k:v for k,v in control.items() if k!='messages'}
print('prepared, not sent: source request line',line_number,'analysis step',control['analysis_step'],'action',control['action'])
