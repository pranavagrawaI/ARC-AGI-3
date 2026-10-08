"""Plant complete raw receipts and corrupt one premise at a time."""
import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent/'mtp-depth-20261006')]
import audit
import native_fixture as native
import probe


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(raw):
    root=raw/'headroom';root.mkdir()
    write(root/'hardware.json',{'valid':True})
    write(root/'cohort.json',[{'recorded_prompt':100}]*10)
    write(root/'server-info-before.json',{'internal_states':[{'effective_max_running_requests_per_dp':10,
          'max_mamba_cache_size':60,'memory_usage':{'token_capacity':1000000}}]})
    shutil.copytree(HERE/'native-source',root/'runtime-source')
    write(root/'runtime-source/identity.json',{'files':json.loads((HERE/'native-contract.json').read_text())['files']})
    blocks=[]
    for label,n in [('WARMUP',10),*probe.PLAN]:
        directory=root/label;directory.mkdir();rows=[];events=[]
        tokens=32 if label=='WARMUP' else 4096
        for i in range(30):
            text=(f'sglang:num_running_reqs {n}\nsglang:num_queue_reqs 0\n'
                  f'sglang:realtime_tokens_total{{mode="decode"}} {i*(10 if label=="WARMUP" else 100)}\n'
                  'sglang:generation_tokens_total 0\n'
                  'sglang:realtime_tokens_total{mode="prefill_compute"} 100\n')
            path=directory/'metrics'/f'{i}.prom';path.parent.mkdir(exist_ok=True);path.write_text(text)
            event=dict(t=float(i),wall=0,kind='metrics',pending=n,**probe.parse_metrics(text.encode()),
                       raw_metrics=str(path.relative_to(directory)),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            rows.append(event);events.append(event)
        responses=[]
        for j in range(n):
            write(directory/f'g{j}-request.json',dict(max_tokens=tokens,ignore_eos=True,seed=20261008,temperature=.7,top_p=.95,top_k=20))
            usage=dict(completion_tokens=tokens,prompt_tokens=100,prompt_tokens_details={'cached_tokens':95})
            write(directory/f'g{j}-response.json',{'usage':usage});responses.append({'usage':usage})
        summary=dict(label=label,n=n,status='complete',responses=responses,plateau=probe.plateau(rows,n),retractions=0)
        endpoints=[]
        for kind,total in [('metrics_before',0),('metrics_after',n*tokens)]:
            text=(f'sglang:num_running_reqs 0\nsglang:num_queue_reqs 0\n'
                  f'sglang:realtime_tokens_total{{mode="decode"}} {total}\n'
                  f'sglang:generation_tokens_total {total}\n'
                  'sglang:realtime_tokens_total{mode="prefill_compute"} 100\n')
            path=directory/'metrics'/f'{kind}.prom';path.write_text(text)
            endpoints.append(dict(t=-1 if kind=='metrics_before' else 31,wall=0,kind=kind,**probe.parse_metrics(text.encode()),
                       raw_metrics=str(path.relative_to(directory)),raw_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        summary['accounting']=probe.accounting(endpoints[0],rows,endpoints[1],n*tokens,n)
        events=[endpoints[0],*events,endpoints[1]]
        if label.startswith('PROFILE'):
            native.start_profile(native.profiler(),dict(output_dir=str(root/'traces'),num_steps=64,activities=['CPU','GPU'],
                with_stack=False,record_shapes=False,profile_by_stage=False,profile_id=label,detailed_annotations=True))
            summary.update(profile_attempted=True,traces=probe.verify_trace(label,root))
            events.append(dict(kind='profile_request',t=8,wall=0))
            events.append(dict(kind='profile_response',raw='Start profiling.\n',t=8,wall=0))
        write(directory/'summary.json',summary)
        (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        if label!='WARMUP':blocks.append(summary)
    write(root/'state.json',dict(run_id=probe.RUN_ID,blocks=blocks))
    manifest(root)
    return root


def manifest(root):
    write(root/'output-hashes.json',{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in root.rglob('*') if p.is_file() and p.name!='output-hashes.json'})


class AuditTests(unittest.TestCase):
    def test_raw_receipts_qualify_and_missing_cuda_fails_even_with_updated_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);root=fixture(raw)
            result=audit.audit(raw)
            self.assertTrue(result['qualified'],result)
            self.assertEqual(len(result['blocks']),5)
            self.assertEqual(result['blocks'][3]['plateau']['U'],100)
            next((root/'traces').glob('PROFILE10-*')).unlink();manifest(root)
            self.assertFalse(audit.audit(raw)['qualified'])

    def test_corrupt_projection_fails_even_with_updated_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);root=fixture(raw)
            path=root/'N10A/events.jsonl';events=[json.loads(x) for x in path.read_text().splitlines()]
            events[12]['decode_tokens']+=100
            path.write_text(''.join(json.dumps(e)+'\n' for e in events));manifest(root)
            result=audit.audit(raw)
            self.assertFalse(result['qualified'])
            self.assertIn('projection mismatch',result['error'])

    def test_missing_response_and_wrong_runtime_do_not_qualify(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);root=fixture(raw)
            (root/'N10A/g0-response.json').unlink();manifest(root)
            self.assertFalse(audit.audit(raw)['qualified'])
        with tempfile.TemporaryDirectory() as tmp:
            raw=Path(tmp);root=fixture(raw)
            (root/'runtime-source/srt/managers/scheduler.py').write_text('wrong runtime');manifest(root)
            result=audit.audit(raw)
            self.assertFalse(result['qualified']);self.assertIn('native source mismatch',result['error'])

if __name__=='__main__':unittest.main()
