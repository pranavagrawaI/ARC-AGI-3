import copy,hashlib,json,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import candidate_evidence,observation,resources
class HostEvidence(unittest.TestCase):
    def fixture(self,root):
        paths={key:'/fixture/'+key for key in ('meminfo','maximum','current','stat','events','procs')};paths['disk']='/tmp'
        observed=dict(cgroup='0::/fixture\n',namespaces={'pid':'pid:[1]'})
        scope=dict(run_id='CPU',observer_scope=observed,paths=paths,file_keys={key:dict(inode=1) for key in paths if key!='disk'},required=['meminfo','maximum','current','stat','events','procs'])
        (root/'observer-scope.json').write_text(json.dumps(scope));(root/'candidate-scope.json').write_text(json.dumps(dict(run_id='CPU',observer=observed,paths=paths,file_keys={key:dict(inode=1) for key in paths})))
        def row(label,start,end):
            raw=dict(meminfo='MemAvailable: 67108864 kB\n',maximum=str(64*1024**3),current=str(40*1024**3),stat='inactive_file 0\n',events='oom 0\noom_kill 0\n',procs='')
            value=dict(label=label,start=start,end=end,run_id='CPU',scope=observed,paths=paths,raw=raw,oom_events={'oom':0,'oom_kill':0},file_clocks={key:dict(start=start,end=end) for key in scope['required']})
            if label!='sample':value.update(valid=True,invalidity=[],original140=resources.cgroup({key:raw[key] for key in ('meminfo','maximum','current','stat')}))
            return value
        boundaries=[row('before_candidate_preparation',1,2),row('candidate.prepare.before',3,4),row('final_cleanup',9,10)]
        rows=[row('sample',2,2.1),row('sample',5,5.1),row('sample',10,10.1)]
        events=[dict(event='start',work_deadline=100)]+[dict(event='host_boundary',**{key:r[key] for key in ('label','start','end','valid','invalidity')}) for r in boundaries]
        self.persist(root,boundaries,rows)
        return boundaries,rows,events
    def persist(self,root,boundaries,rows):
        (root/'host-boundaries.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in boundaries));raw=''.join(json.dumps(row)+'\n' for row in rows).encode();(root/'host-series.jsonl').write_bytes(raw)
        (root/'host-series.result.json').write_text(json.dumps(dict(pass_=True,error=None,samples=len(rows),bytes=len(raw),series_sha256=hashlib.sha256(raw).hexdigest())))
    def test_complete_false140_and_coherent_raw_oom_increase_distinguished(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);boundaries,rows,events=self.fixture(root);config={'run_id':'CPU'}
            value=candidate_evidence.host_evidence(root,config,events);self.assertFalse(value['original140']);self.assertFalse(value['oom_increased'])
            rows[-1]['raw']['events']='oom 1\noom_kill 1\n';rows[-1]['oom_events']={'oom':1,'oom_kill':1};self.persist(root,boundaries,rows)
            self.assertTrue(candidate_evidence.host_evidence(root,config,events)['oom_increased'])
            rows[-1]['oom_events']['oom']=0;self.persist(root,boundaries,rows)
            with self.assertRaisesRegex(ValueError,'OOM consistency'):candidate_evidence.host_evidence(root,config,events)
    def test_malformed_or_missing_host_evidence_is_not_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);boundaries,rows,events=self.fixture(root);rows[1]['raw']['maximum']='nonsense';self.persist(root,boundaries,rows)
            with self.assertRaisesRegex(ValueError,'cgroup max'):candidate_evidence.host_evidence(root,{'run_id':'CPU'},events)
            rows[1]['raw'].pop('events');self.persist(root,boundaries,rows)
            with self.assertRaises((ValueError,KeyError)):candidate_evidence.host_evidence(root,{'run_id':'CPU'},events)
    def test_returned_invalid_boundary_is_saved_before_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);boundaries,rows,events=self.fixture(root);runner=object.__new__(observation.Observation)
            runner.output=root;runner.check_observer=lambda:None;runner.observer_spec={};runner.initial_events={'oom':0,'oom_kill':0};runner.boundary_count=0;runner.work_end=100;runner.config={'observer':{'boundary_timeout_seconds':5}};runner.emit=lambda *args,**kwargs:None
            row=boundaries[1];row['raw']['events']='oom 1\noom_kill 1\n';row['oom_events']={'oom':1,'oom_kill':1}
            with patch('observation.host_observer.snapshot',return_value=row),self.assertRaisesRegex(ValueError,'OOM'):runner.boundary('injected')
            retained=json.loads((root/'host-boundaries.jsonl').read_text().splitlines()[-1]);self.assertFalse(retained['valid']);self.assertEqual(retained['oom_events']['oom_kill'],1)
