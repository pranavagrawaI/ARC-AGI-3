import copy,hashlib,json,tempfile,unittest
from pathlib import Path
import pair_evidence
class Replay(unittest.TestCase):
 def fixture(self,root):
  (root/'candidate').mkdir();argv=['python3','/run/replay.py','run','/run/wire','/run/output/candidate','--arm','candidate']
  (root.parent/'config.json').write_text(json.dumps(dict(candidate=dict(replay=dict(argv=argv,seconds=1510)))))
  server=dict(pid=80,pgid=80,starttime=90,uid=1000,cgroup='cg',pid_namespace='ns',boot_id='boot')
  identity=dict(server,pid=100,pgid=100,sid=100,starttime=110,cmdline_sha256=hashlib.sha256(b''.join(v.encode()+b'\0' for v in argv)).hexdigest())
  capture=dict(server_pgid='80',server_start='90',boot_id='boot',session_id='session',gpu_uuid='GPU-x',origin=20.,end=1520.)
  header=dict(pid=100,replay_pgid=100,replay_starttime=110,server_pgid='80',server_start='90',session_id='session',gpu_uuid='GPU-x',boot_id='boot')
  (root/'candidate/events.jsonl').write_text(json.dumps(header)+'\n')
  events=[dict(event='start',work_deadline=9900.),dict(event='ready',arm='candidate',t=10.),dict(event='capacity_server_bound',arm='candidate',t=11.,identity=server),dict(event='spawn',phase='candidate.replay',pid=100,pgid=100,identity=identity,argv=argv,deadline=1525.,t=15.),dict(event='exit',phase='candidate.replay',pid=100,t=1521.),dict(event='cleanup',pgid=100,remaining=[],t=1522.),dict(event='candidate_cleanup_start',cleanup_start=1523.)]
  events[2].update(session_id='session',gpu_uuid='GPU-x')
  return events,capture
 def test_cross_arm_pid_source_and_late_cleanup_rejected(self):
  with tempfile.TemporaryDirectory() as td:
   root=Path(td)/'output';root.mkdir();events,capture=self.fixture(root)
   self.assertEqual(pair_evidence.replay_join(root,events,'candidate',capture)['starttime'],110)
   mutations=[]
   value=copy.deepcopy(events);value[3]['identity']['starttime']=111;mutations.append(value)
   value=copy.deepcopy(events);value[3]['identity']['cmdline_sha256']='0'*64;mutations.append(value)
   value=copy.deepcopy(events);value[2]['identity']['pgid']=81;mutations.append(value)
   value=copy.deepcopy(events);value[5]['t']=1600.;mutations.append(value)
   value=copy.deepcopy(events);value[3]['argv']=['python3','/run/control.py'];mutations.append(value)
   for value in mutations:
    with self.subTest(value=value),self.assertRaises(ValueError):pair_evidence.replay_join(root,value,'candidate',capture)
