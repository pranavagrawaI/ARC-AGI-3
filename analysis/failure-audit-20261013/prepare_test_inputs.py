"""Prepare three context arms; this is not an inference launcher."""
from pathlib import Path
import hashlib
import json

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent/'test-inputs'
OUT.mkdir(exist_ok=True)
prior=ROOT/'analysis/s5i5-inspection-20261012'
control=json.loads((prior/'continuation-control-request.json').read_text())
diagnostic=json.loads((prior/'continuation-feedback-request.json').read_text())
check=json.loads((prior/'model-check.json').read_text())['available_when_failed_predictor_was_written']['row']
reminder='Previously observed transition: MOUSE(row=57,col=27).\nPredecessor: '+json.dumps(check['input_state'],sort_keys=True)+'\nObserved successor: '+json.dumps(check['actual_successor'],sort_keys=True)+'\nContinue using this observation.'
observation={**control,'messages':control['messages']+[{'role':'user','content':reminder}]}
arms={'control':control,'observation-reminder':observation,'prediction-counterexample':diagnostic}
hashes={}
for label,data in arms.items():
    path=OUT/(label+'.json');path.write_text(json.dumps(data,indent=2)+'\n')
    hashes[label]=hashlib.sha256(path.read_bytes()).hexdigest()
    if label!='control':assert data['messages'][:-1]==control['messages']
manifest={'source_checkpoint':'s5i5 B1 r1 request-log line 73, analysis step 19, action 43',
          'arms':hashes,'sent':False,'future_evidence_included':False,
          'critical_fact_was_already_present_in_original_context':True,
          'primary_question':'Does explicit prediction contradiction add repair benefit beyond ordinary continuation or repeating the observation?',
          'measures':['Generated tokens until a valid predictor','Coverage and agreement on past transitions',
                      'Prediction on held-out transitions','Action-plan feasibility'],
          'validity_rules':['Count actual comparisons; zero coverage is invalid','Keep no-successor-on-noop separate from successor matches',
                            'Keep original model, quantization, sampling and visible evidence fixed',
                            'Do not expose later repairs, routes, or engine internals to the model'],
          'execution_not_ready_until':['Restore the historical sandbox/helper/data state at this checkpoint',
                                       'Freeze inference caps and original runtime configuration',
                                       'Obtain run authority for any consequential compute'],
          'scope':'Context packets only. This manifest does not authorize or launch model inference.'}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('three context arms prepared; none sent')
