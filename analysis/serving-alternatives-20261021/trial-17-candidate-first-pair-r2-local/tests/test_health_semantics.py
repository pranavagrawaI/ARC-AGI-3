"""Compose real retained readiness receipts with generic lifecycle validation."""
import copy,json,shutil,tempfile,unittest
from pathlib import Path
import pair_evidence
HERE=Path(__file__).resolve().parents[1]
BASE=HERE.parent
CANDIDATE=BASE/'trial16-three-pass-audit-result-20261024/evidence/serving-candidate-s10-16/output'
CONTROL=BASE/'trial-14-phase-handles-local/remote-readback/archive-v1-kernel140449504/members/serving-post-control-load14/output'

def retained(root,label):
 rows=[json.loads(line) for line in (root/'supervisor.jsonl').read_bytes().splitlines()]
 spawns=[r for r in rows if r['event']=='spawn' and r.get('phase','').startswith(label+'.health-')]
 pids={r['pid'] for r in spawns};pgids={r['pgid'] for r in spawns}
 beginning='candidate_root_retained' if label=='candidate' else 'native_spawn'
 return [rows[0]]+[r for r in rows[1:] if r in spawns or r['event']==beginning or
  (r['event']=='ready' and r['arm']==label) or (r['event']=='exit' and r['pid'] in pids) or
  (r['event']=='cleanup' and r['pgid'] in pgids)]

class HealthSemantics(unittest.TestCase):
 def setUp(self):self.config=json.loads((HERE/'packet-spec.json').read_bytes())['config']
 def validate(self,root,rows,label):
  pair_evidence.phase_deadlines(rows)
  return pair_evidence.health(root,rows,self.config,label)
 def test_actual_candidate_refusal_then200_and_deadlines_together(self):
  rows=retained(CANDIDATE,'candidate')
  self.assertEqual(next(r for r in rows if r['event']=='exit')['returncode'],75)
  self.assertTrue(self.validate(CANDIDATE,rows,'candidate'))
 def test_actual_control_refusal_pinned503_then200_and_deadlines_together(self):
  rows=retained(CONTROL,'control')
  self.assertTrue(any(r['event']=='exit' and r['returncode']==1 for r in rows))
  self.assertTrue(self.validate(CONTROL,rows,'control'))
 def test_malformed503_unexpectedHTTP_missing200_and_nonhealth_failure_rejected(self):
  rows=retained(CONTROL,'control')
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)
   for p in CONTROL.glob('control.health-*.*'):shutil.copyfile(p,root/p.name)
   failed=next(r for r in rows if r['event']=='exit' and r['returncode']==1)
   path=root/(failed['phase']+'.stdout');raw=path.read_bytes()
   for bad in (raw+b'not-empty',raw.replace(b'503',b'502',1)):
    path.write_bytes(bad)
    with self.assertRaises(ValueError):self.validate(root,rows,'control')
   path.write_bytes(raw)
   final=next(r for r in reversed(rows) if r['event']=='exit')
   changed=copy.deepcopy(rows);next(r for r in changed if r['event']=='exit' and r['pid']==final['pid'])['returncode']=75
   with self.assertRaises(ValueError):self.validate(root,changed,'control')
   changed=copy.deepcopy(rows)
   changed.extend([dict(event='spawn',phase='control.identity',pid=900000,pgid=900000,t=1.,deadline=3.),
    dict(event='exit',phase='control.identity',pid=900000,t=2.,returncode=1),
    dict(event='cleanup',pgid=900000,t=2.1,remaining=[])])
   with self.assertRaisesRegex(ValueError,'control.identity'):self.validate(root,changed,'control')
 def test_health_cleanup_clocks_and_pid_join_remain_binding(self):
  rows=retained(CANDIDATE,'candidate');spawn=next(r for r in rows if r['event']=='spawn')
  for mutation in ('late_cleanup','wrong_pid','remaining_group'):
   changed=copy.deepcopy(rows)
   if mutation=='wrong_pid':next(r for r in changed if r['event']=='exit')['pid']+=1000000
   else:
    cleanup=next(r for r in changed if r['event']=='cleanup' and r['pgid']==spawn['pgid'])
    if mutation=='late_cleanup':cleanup['t']=spawn['deadline']+.001
    else:cleanup['remaining']=[spawn['pid']]
   with self.assertRaises(ValueError):self.validate(CANDIDATE,changed,'candidate')
