import json,sys,time
from pathlib import Path
from unittest.mock import patch
HERE=Path('/home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021/trial-16-candidate-s10-r2-local')
sys.path[:0]=[str(HERE),str(HERE/'tests')]
import test_pipeline
OUT=Path(__file__).resolve().parent
root=OUT/'late-publication'/'run';root.mkdir(parents=True)
fixture=test_pipeline.Pipeline();config=fixture.config();config['primary_seconds']=.2
fixture.config=lambda:config
original=Path.write_text;calls=[]
def write(path,text,*args,**kwargs):
 if path.name=='primary-bundle-receipt.json':
  receipt=json.loads(text);entered=time.monotonic()
  # Delay only the final real filesystem write, after helper's final clock check.
  time.sleep(max(0,receipt['deadline']-time.monotonic())+.05)
  result=original(path,text,*args,**kwargs)
  calls.append(dict(entered=entered,actual_write_end=time.monotonic(),saved_receipt=receipt))
  return result
 return original(path,text,*args,**kwargs)
with patch.object(Path,'write_text',write):result=fixture.run_fixture(root)
returned=time.monotonic()
call=calls[0];receipt=call['saved_receipt']
events=[json.loads(line) for line in (root/'output/supervisor.jsonl').read_text().splitlines()]
publication_clock_predicate=(receipt['start']<=receipt['end']<=receipt['deadline']<=min(events[0]['work_deadline'],receipt['start']+config['primary_seconds']))
report=dict(result=result,publication=call,returned=returned,late_seconds=returned-receipt['deadline'],
 offline_publication_clock_predicate_accepts=publication_clock_predicate,
 scope='CPU pipeline with explicit hardware/model/window/final verdict substitutes; actual runner, primary serialization/persistence, and bounded delay only at final receipt write; no GPU/network/model')
import primary_bundle
assert not result['pass_'] and result['status']=='INVALID_INCONCLUSIVE' and result['U_resp'] is None
assert receipt['end']<=receipt['deadline']<call['actual_write_end']
assert 'primary_bundle_error' in result
try:primary_bundle.verify_publication(root/'output',(root.parent/primary_bundle.NAME).read_bytes(),events[0]['work_deadline'],config['primary_seconds'])
except (ValueError,FileNotFoundError) as error:report['new_offline_rejection']=repr(error)
else:raise AssertionError('stale publication accepted')
(OUT/'late-publication-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
