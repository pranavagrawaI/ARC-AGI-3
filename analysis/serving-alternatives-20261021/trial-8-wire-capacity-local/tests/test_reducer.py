"""Analytical raw-file fixtures, not model-rate or corpus feasibility evidence."""
import base64
import copy
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import zlib
import catalog
import reduce_api

def png():
    def chunk(name,data):return struct.pack('>I',len(data))+name+data+struct.pack('>I',zlib.crc32(name+data))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',640,640,8,2,0,0,0))+chunk(b'IDAT',zlib.compress((b'\0'+b'\0'*(640*3))*640))+chunk(b'IEND',b'')

def fixture(root,label,origin,completion_times):
    directory=root/label;directory.mkdir();image=png();records={};streams={};events=[];responses=[]
    def emit(t,kind,**fields):events.append(dict(t=t+origin,kind=kind,**fields))
    for i in range(10):
        g=f'g{i}';streams[g]=[]
        for n in range(10):
            key=f'{g}:{n}';streams[g].append(key)
            body={'model':'flashnext','messages':[{'role':'user','content':[{'type':'text','text':key},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image).decode()}}]}],'max_tokens':300}
            records[key]=dict(game=g,index=n,cap=300,prompt=1000,local_prompt=1000,images=[catalog.digest(image)],wire_sha=catalog.wire_hash(body),extends_previous=True)
            rid=f'{label}-{g}-{n}';out=directory/rid;out.mkdir()
            raw=catalog.canonical(body);wire=b'POST /v1/chat/completions HTTP/1.1\r\nX-Request-Id: '+rid.encode()+b'\r\nContent-Length: '+str(len(raw)).encode()+b'\r\n\r\n'+raw
            (out/'wire.http').write_bytes(wire);(out/'wire.json').write_bytes(catalog.canonical({'headers':{'X-Request-Id':rid},'body':body}))
            if n<=len(completion_times):
                sent=0 if n==0 else completion_times[n-1]
                if n==0:emit(0,'admit',game=g)
                emit(sent,'send',rid=rid,key=key,wire_ref=rid+'/wire.json',transport_ref=rid+'/wire.http')
                complete=n<len(completion_times);ended=completion_times[n] if complete else 1500
                receipt={'rid':rid,'key':key,'complete':complete,'status':200,'error':None if complete else "CutoffCancelled('owned socket cancelled at fixed cutoff')",'started':origin+sent+.001,'ended':origin+ended-.001 if complete else origin+ended+.01,'writes':[{'t':origin+sent+.001,'ended':origin+sent+.002,'bytes':len(wire)}],
                         'body_completed_at':origin+ended-.001 if complete else None,
                         'termination':'COMPLETE' if complete else 'INTENTIONAL_CUTOFF',
                         'error_kind':None if complete else 'INTENTIONAL_CUTOFF',
                         'cause':None if complete else "CutoffCancelled('owned socket cancelled at fixed cutoff')",
                         'cause_type':None if complete else 'CutoffCancelled',
                         'cutoff_shutdown_at':None if complete else origin+1500.001,
                         'error_observed_at':None if complete else origin+1500.01}
                if complete:
                    details={'cached_tokens':800}
                    details.update({'image_tokens':400} if label=='control' else {'multimodal_tokens':{'image':400}})
                    response={'id':'response-'+rid,'usage':{'prompt_tokens':1000,'completion_tokens':300,'total_tokens':1300,'prompt_tokens_details':details},'choices':[{'finish_reason':'length','message':{'content':'CPU fixture'}}]}
                    data=catalog.canonical(response);(out/'body.bin').write_bytes(data)
                    receipt.update(headers=[['Content-Length',str(len(data))]],received_bytes=len(data),persisted_bytes=len(data))
                    emit(ended,'response',rid=rid,status=200,response_ref=rid+'/body.bin',http_ref=rid+'/http.json');responses.append(out/'body.bin')
                else:
                    emit(1500,'cutoff_pending',rid=rid)
                    receipt.update(headers=[['Content-Length','228']],received_bytes=0,persisted_bytes=0)
                    (out/'body.bin').write_bytes(b'')
                (out/'http.json').write_bytes(catalog.canonical(receipt))
    emit(0,'replay_start',session_id='CPU_ONLY',gpu_uuid='CPU_ONLY',server_pgid='100' if label=='control' else '200',server_start=str(origin))
    # Boundary markers precede any event at the same timestamp, fixed a priori.
    events.insert(0,events.pop())
    emit(600,'measure_start',boundary=origin+600);emit(1500,'measure_end',boundary=origin+1500)
    for t in range(1,1500):emit(t,'heartbeat')
    emit(1500.1,'capture_end',threads_alive=0,failed=False)
    events.sort(key=lambda e:(e['t'], -1 if e['kind']=='measure_end' else 0))
    for seq,e in enumerate(events):e.update(seq=seq,pid=123,boot_id='CPU_ONLY')
    (directory/'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
    cat={'records':records,'streams':streams,'initial_holders':list(streams),'initial_queue':[],'slots':10}
    return cat,responses

class ReducerTests(unittest.TestCase):
    def build(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);root=Path(t.name)
        cat,refs=fixture(root,'control',0,[100,700,1000,1300]);fixture(root,'candidate',2000,[100,700,800,1000,1100,1300,1400])
        return root,cat,refs
    def test_known_package_ratio_and_omitted_native_metric(self):
        root,cat,_=self.build()
        # Analytical API fixture does not claim real package/source qualification.
        with patch.object(catalog,'load',return_value=cat),patch.object(reduce_api.provenance,'check',return_value={'CPU_FIXTURE_ONLY':True}):value=reduce_api.compare(root)
        self.assertEqual(value['verdict'],'QUALITY_INVESTIGATION_ONLY');self.assertEqual(value['ratio'],2)
        self.assertEqual(value['control']['completed'],30);self.assertEqual(value['control']['U_resp'],10)
        self.assertEqual(value['control']['available_rate_metrics'],['U_resp'])
        self.assertNotIn('per_stream_rate',value['control'])
    def test_missing_actual_package_receipts_cannot_pass_comparison(self):
        root,cat,_=self.build()
        with patch.object(catalog,'load',return_value=cat):value=reduce_api.compare(root)
        self.assertEqual(value['verdict'],'FEASIBILITY_ONLY');self.assertIsNone(value['ratio'])
    def test_pending_disk_loss_cannot_qualify_package_ratio(self):
        root,cat,_=self.build()
        rows=list(map(json.loads,(root/'control/events.jsonl').read_text().splitlines()))
        rid=next(row['rid'] for row in rows if row['kind']=='cutoff_pending')
        path=root/'control'/rid/'http.json';receipt=json.loads(path.read_text())
        receipt.update(received_bytes=228,headers=[['Content-Length','228']],
                       error="OSError('persistence failed after final body read')",complete=False)
        path.write_bytes(catalog.canonical(receipt));(path.parent/'body.bin').write_bytes(b'')
        with patch.object(catalog,'load',return_value=cat),patch.object(reduce_api.provenance,'check',return_value={'CPU_FIXTURE_ONLY':True}):
            value=reduce_api.compare(root)
        self.assertEqual(value['verdict'],'FEASIBILITY_ONLY')
        self.assertIsNone(value['ratio']);self.assertFalse(value['control']['valid'])
    def test_pending_cutoff_requires_owned_cancellation_not_arbitrary_error(self):
        root,cat,_=self.build()
        rows=list(map(json.loads,(root/'control/events.jsonl').read_text().splitlines()))
        rid=next(row['rid'] for row in rows if row['kind']=='cutoff_pending')
        path=root/'control'/rid/'http.json';original=json.loads(path.read_text())
        self.assertTrue(reduce_api.arm(root/'control',cat,'control')['valid'])
        changes=[{'error':"OSError('disk failed')"},{'error_kind':'PERSISTENCE'},
                 {'cutoff_shutdown_at':None},{'received_bytes':228},
                 {'body_completed_at':1499.99}]
        for change in changes:
            path.write_bytes(catalog.canonical(dict(original,**change)))
            with self.subTest(change=change):self.assertFalse(reduce_api.arm(root/'control',cat,'control')['valid'])
    def test_missing_image_total_fails_closed(self):
        root,cat,refs=self.build();p=refs[0];v=json.loads(p.read_bytes());del v['usage']['prompt_tokens_details'];p.write_bytes(catalog.canonical(v))
        result=reduce_api.arm(root/'control',cat,'control');self.assertFalse(result['valid'])
    def test_corrupt_raw_transport_cannot_use_intended_payload(self):
        root,cat,refs=self.build();p=refs[0].parent/'wire.http';p.write_bytes(p.read_bytes().replace(b'X-Request-Id:',b'Y-Request-Id:'))
        self.assertFalse(reduce_api.arm(root/'control',cat,'control')['valid'])
    def test_cutoff_clock_not_extended(self):
        root,cat,_=self.build();p=root/'control/events.jsonl';rows=list(map(json.loads,p.read_text().splitlines()))
        next(e for e in rows if e['kind']=='measure_end')['boundary']+=1
        p.write_text(''.join(json.dumps(e)+'\n' for e in rows));self.assertFalse(reduce_api.arm(root/'control',cat,'control')['valid'])
    def test_body_clock_prevents_warmup_sample_leak(self):
        root,cat,refs=self.build();p=root/'control/events.jsonl';rows=list(map(json.loads,p.read_text().splitlines()))
        # Put ten body completions just before warmup, events just after. They
        # must not count toward the >=30 measured-completion requirement.
        for event in rows:
            if event['kind']=='response' and event['t']==700:
                event['t']=600.1
                h=root/'control'/event['http_ref'];v=json.loads(h.read_text());v['ended']=v['body_completed_at']=599.99;h.write_bytes(catalog.canonical(v))
        rows.sort(key=lambda e:e['t'])
        for seq,event in enumerate(rows):event['seq']=seq
        p.write_text(''.join(json.dumps(e)+'\n' for e in rows))
        value=reduce_api.arm(root/'control',cat,'control')
        self.assertFalse(value['valid'])
        self.assertEqual(value.get('completed'),20)
