import json,sys,time
from pathlib import Path
from unittest.mock import patch
HERE=Path('/home/pranav/projects/arc-agi-3/analysis/serving-alternatives-20261021/trial-16-candidate-s10-local')
sys.path[:0]=[str(HERE),str(HERE/'tests')]
import test_pipeline,host_observer
OUT=Path(__file__).resolve().parent
root=OUT/'oom-boundary';root.mkdir()
original=host_observer.snapshot
state={'increased':False,'injected_labels':[]}
def snapshot(spec,label):
 row=original(spec,label)
 if label=='candidate.replay.before':state['increased']=True
 if state['increased']:
  # Input injection only: simulate an available complete counter increase.
  # No real cgroup mutation or OOM; production boundary/control flow unchanged.
  counts=dict(row['oom_events']);counts['oom']+=1;counts['oom_kill']+=1
  row['oom_events']=counts;row['raw']['events']=''.join(f'{key} {value}\n' for key,value in counts.items())
  state['injected_labels'].append(label)
 return row
with patch('host_observer.snapshot',side_effect=snapshot):
 result=test_pipeline.Pipeline().run_fixture(root)
out=root/'output'
events=[json.loads(line) for line in (out/'supervisor.jsonl').read_text().splitlines()]
bounds=[json.loads(line) for line in (out/'host-boundaries.jsonl').read_text().splitlines()]
release=json.loads((out/'candidate-release.json').read_text())
server=[e for e in events if e['event']=='spawn' and e['phase']=='candidate.server'][0]
observer=[e for e in events if e['event']=='spawn' and e['phase']=='observer'][0]
report=dict(result=result,injected_labels=state['injected_labels'],boundary_labels=[r['label'] for r in bounds],
 cleanup=[e for e in events if e['event']=='cleanup' and e['pgid'] in (server['pid'],observer['pid'])],
 release_samples=len(release['samples']),release_error=release['error'],
 observer_stop_exists=(out/'observer.stop').exists(),series_exists=(out/'host-series.jsonl').exists(),
 series_report_exists=(out/'host-series.result.json').exists(),
 primary_exists=(OUT/'000-serving-candidate16-primary.json').exists(),
 scope='CPU fixture: raw OOM boundary-input injection, actual candidate/observer processes and group cleanup; no real OOM/GPU/model/wire window')
assert result['status']=='INVALID_INCONCLUSIVE' and not result['pass_']
assert result['surviving_phase_groups']==[] and result['retained_queries']==[]
assert len(report['cleanup'])==2 and all(e['remaining']==[] for e in report['cleanup'])
assert report['release_samples']==0 and 'OOM' in report['release_error']
assert report['series_exists'] and not report['observer_stop_exists'] and not report['series_report_exists']
assert bounds[-1]['label']=='final_cleanup' and 'OOM counter increased; stop load test' in bounds[-1]['invalidity']
assert result['verification_error']=="ValueError('cleanup incomplete')"
(OUT/'oom-review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
