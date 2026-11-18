import json,sys
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[2]/'trial-16-candidate-s10-local'
sys.path[:0]=[str(P),str(P/'tests')]
import test_pipeline,observation
DEST=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).parent/'oom-boundary-run'
DEST.mkdir(parents=True,exist_ok=False)
fixture=test_pipeline.Pipeline()
base=DEST/'baseline';base.mkdir();baseline=fixture.run_fixture(base)
assert baseline['pass_'],baseline
actual=observation.host_observer.snapshot
armed=[False]
def increased(scope,label):
 row=actual(scope,label)
 if label=='candidate.replay.after':armed[0]=True
 if armed[0]:
  row['oom_events']['oom']+=1;row['oom_events']['oom_kill']+=1
  row['raw']['events']=''.join(f'{k} {v}\n' for k,v in row['oom_events'].items())
 return row
mut=DEST/'oom_after_replay';mut.mkdir()
with patch('observation.host_observer.snapshot',side_effect=increased):result=fixture.run_fixture(mut)
receipt=json.loads((mut/'output/candidate-release.json').read_text())
summary=dict(scope='Actual CPU candidate/query/port/process/observer cleanup pipeline; model/window outcome substituted exactly as declared in existing test_pipeline. Only snapshot raw/stored OOM counters injected after replay, without inducing host OOM.',baseline_status=baseline['status'],mutated_result=result,final_receipt_pass=receipt['pass_'],final_receipt_error=receipt['error'],final_resource_samples=len(receipt['samples']),host_series_report_present=(mut/'output/host-series.result.json').exists())
(DEST/'SUMMARY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
