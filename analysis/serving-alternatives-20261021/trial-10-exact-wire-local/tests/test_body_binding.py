import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import catalog,replay,reduce_api,test_replay,test_reducer

class BodyBinding(unittest.TestCase):
 run_fixture=test_replay.ReplayTests.run_fixture
 def test_prepare_validates_declared_body_hash(self):
  body={'model':'flashnext','messages':[]};raw=catalog.canonical(body)
  record={'prepared_body_sha256':catalog.digest(raw),'wire_sha':catalog.wire_hash(body),'images':[]}
  cat={'records':{'a:0':record}}
  with tempfile.TemporaryDirectory() as d,patch.object(catalog,'verify_corpus'),patch.object(catalog,'load',return_value=cat),patch.object(catalog,'payload_for',return_value=body):
   output=Path(d)/'ok';index=replay.prepare(d,output);self.assertEqual(index['a:0']['sha256'],record['prepared_body_sha256'])
   record['prepared_body_sha256']='wrong'
   with self.assertRaisesRegex(ValueError,'bytes/order'):replay.prepare(d,Path(d)/'bad')
 def test_index_cannot_authorize_changed_prepared_body_before_connect(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);bodies=root/'b';bodies.mkdir();body={'model':'flashnext','messages':[]};raw=catalog.canonical(body)
   record={'prepared_body_sha256':catalog.digest(raw),'wire_sha':catalog.wire_hash(body),'images':[]}
   changed=raw+b'\n';(bodies/'a.json').write_bytes(changed);(bodies/'index.json').write_text(json.dumps({'a:0':{'file':'a.json','sha256':catalog.digest(changed)}}))
   obj=replay.Replay({'records':{'a:0':record},'streams':{'a':['a:0']},'initial_holders':['a'],'initial_queue':[]},bodies,root/'out','http://127.0.0.1:1','control',slots=1)
   try:
    with patch.object(replay,'CaptureConnection',side_effect=AssertionError('must not connect')):
     with self.assertRaisesRegex(ValueError,'bytes/order'):obj.once('a:0','local',0)
   finally:obj.log.close()
 def test_actual_outbound_reserialization_stops_replay(self):
  original=replay.CaptureConnection.send
  def changed(connection,data):
   if data.startswith(b'{'):
    value=json.loads(data);data=json.dumps(dict(reversed(list(value.items()))),separators=(',',':')).encode()
   return original(connection,data)
  with patch.object(replay.CaptureConnection,'send',changed):
   root,server,obj,events,ok,elapsed=self.run_fixture('ok',(.1,.15))
  self.assertFalse(ok);self.assertTrue(obj.failed)
  self.assertTrue(any('bytes/order' in e.get('error','') for e in events))
  self.assertFalse(any(e['kind']=='response' for e in events))
 def test_reducer_rejects_semantic_equal_http_body_and_sha_receipt_drift(self):
  helper=test_reducer.ReducerTests();self.addCleanup(helper.doCleanups);root,cat,_=helper.build()
  self.assertTrue(reduce_api.arm(root/'control',cat,'control')['valid'])
  events=list(map(json.loads,(root/'control/events.jsonl').read_text().splitlines()))
  dispatched=next(e for e in events if e['kind']=='send')
  p=root/'control'/dispatched['transport_ref'];raw=p.read_bytes();head,body=raw.split(b'\r\n\r\n',1)
  changed=json.dumps(dict(reversed(list(json.loads(body).items()))),ensure_ascii=False,separators=(',',':')).encode()
  self.assertEqual(json.loads(body),json.loads(changed));self.assertEqual(len(body),len(changed))
  p.write_bytes(head+b'\r\n\r\n'+changed);self.assertFalse(reduce_api.arm(root/'control',cat,'control')['valid'])
  p.write_bytes(raw);wire=p.with_name('wire.json');value=json.loads(wire.read_text());value['outbound_body_sha256']='wrong';wire.write_text(json.dumps(value));self.assertFalse(reduce_api.arm(root/'control',cat,'control')['valid'])
