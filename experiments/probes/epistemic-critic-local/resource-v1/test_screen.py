"""Three-arm development diagnostic using the existing no-retry runner."""
import importlib.util
import json
from pathlib import Path
import sys
import hashlib

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen_screen',HERE.parent/'framework-v1/screen.py')
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
runner.OUT=runner.ROOT/'analysis/local-probes/20260911-critic-resource-screen-v1'
OUT=runner.OUT

def prepare():
    OUT.mkdir(exist_ok=False)
    base=(HERE.parent/'framework-v1/system.md').read_text()
    rule=(HERE/'revision-rule.md').read_text();resources=(HERE/'shared-context.md').read_text()
    prompts={'base':base,'revision':base+'\n'+rule,'resources':base+'\n'+rule+'\n'+resources}
    cases=[
      {'id':'threshold','observations':['Three clicks at different coordinates changed only row 63. No level completed. Completion condition is unknown.'],
       'actor_reasoning':'The goal is to reach the counter threshold. It is either 16 or 64. I will keep clicking until I find it.',
       'expected':'concern','rubric':'Withdraw objective certainty or bound policy without requiring reaching the supposed threshold, claiming threshold false, or inventing counts.'},
      {'id':'uncertainty-control','observations':['Three clicks changed only row 63. No completion condition has been established.'],
       'actor_reasoning':'A larger threshold remains possible, but this policy has not earned more actions. I will leave that hypothesis open and try one untested interaction; I do not need to establish the whole goal first.',
       'expected':'no_concern','rubric':'Permit stopping a policy while preserving hypothesis; do not demand complete evidence or a compulsory alternative model.'},
      {'id':'time-paralysis','observations':['Host telemetry: 35 seconds remain for admitting model calls and 180 environment actions remain. Recent model calls took 8 to 12 seconds each. Goal is unknown. One untested LEFT is available.'],
       'actor_reasoning':'Because actions are scarce, I will take four more model calls to settle the complete goal theory before choosing any action. That is the safest way to conserve our budget.',
       'expected':'concern','rubric':'Account for 32-48 seconds plus subsequent action work, no guarantee of settling goal; allow bounded inquiry/action under uncertainty without making LEFT mandatory or claiming inevitable timeout.'},
      {'id':'deadline-falsification','observations':['An allocated two-probe investigation ended without level completion. The model-admission budget is nearly exhausted. Larger thresholds have not been tested.'],
       'actor_reasoning':'The allocated time ended without success, so the counter-threshold hypothesis is false. I will erase it from the notes and exclude counter-based objectives in future reasoning.',
       'expected':'concern','rubric':'Distinguish abandoning resource allocation from falsifying hypothesis. Preserve uncertainty without demanding further threshold pursuit.'}]
    runner.write('CASES.json',cases);order=[]
    template=json.loads(next((runner.ROOT/'analysis/local-probes/20260911-critic-framework-screen-v1').glob('*.request.json')).read_text())
    for i,c in enumerate(cases):
        arms=list(prompts);arms=arms[i%3:]+arms[:i%3]
        for arm in arms:
            name=f'{len(order):02d}-{c["id"]}-{arm}'
            body=dict(template);body['messages']=[{'role':'system','content':prompts[arm]},
              {'role':'user','content':json.dumps({k:c[k] for k in ('observations','actor_reasoning')})}]
            runner.write(name+'.request.json',body);order.append({'name':name,'case':c['id'],'arm':arm})
    runner.write('ORDER.json',order)
    reserve=sum((len((OUT/(r['name']+'.request.json')).read_bytes())+4096)*.42/1e6+384*3/1e6 for r in order)
    assert reserve<.25
    (OUT/'EXPERIMENT-CARD.md').write_text('''# Resource-aware epistemic revision diagnostic
Question: Does self-checking the critic revision improve warranted corrections, and does explaining resource implications add value?
Posture: Development diagnostic.
Contrast: Framework base / base plus revision check / revision check plus shared resource explanation; four identical authored cases per arm, cyclic order, one sample each.
Prediction: Revision check removes circular threshold advice. Resource explanation improves deadline-aware inquiry without converting scarcity into evidence of falsity.
Measure: Grade raw responses against CASES.json, separating assessment, revision validity, unsupported premises, restraint, schema delivery, latency, tokens and cost.
Cap: 12 calls, $0.25 total conservative reservation, 480 seconds admission window, 30 seconds per HTTP call, 384 generated tokens per call, zero gameplay actions. Existing Qwen3.8-27B/CoreWeave FP8 endpoint, reasoning off.
Stop: All calls complete or transport/provider/finish/accounting failure; no retries or relaunch. Preserve semantic output failures and continue.
Invalid if: Frozen input drift, wrong model/provider, diagnosis leaked into input, unaccounted requests. Semantic failures do not become no_concern.
Result map: Improved corrections -> fresh-case or Actor uptake test; unchanged -> no incremental benefit demonstrated; harmful/ambiguous -> no promotion, preserve exact counterexamples. No gameplay efficacy or generalization claim from authored development cases.
Outputs: This directory, exact requests/responses, frozen inputs, execution ledger, manual result report.
''')
    sources=[Path(__file__),HERE/'revision-rule.md',HERE/'shared-context.md',HERE.parent/'framework-v1/screen.py',HERE.parent/'paired_commitment.py',HERE.parent/'claim_screen.py',HERE.parent/'reasoning_screen.py']
    runner.write('FROZEN.json',{'reserved_usd':reserve,'files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in list(OUT.iterdir())+sources if p.is_file()}})
    print(json.dumps({'prepared':12,'reserved_usd':reserve}))

if __name__=='__main__':
    {'prepare':prepare,'execute':runner.execute}[sys.argv[1]]()
