import json,tempfile,unittest
from pathlib import Path
from run_pair import authorize_launch,admit,assess_output,request_receipts
ROOT=Path(__file__).resolve().parents[2]
GOLD=ROOT/'solvers/m2_f1x_reap/experiment/screen-1/merge/output'
class PairSafety(unittest.TestCase):
 def test_push_uncertainty_consumes_single_attempt(self):
  with tempfile.TemporaryDirectory() as d:
   folder=Path(d);calls=[]
   def push():calls.append('accepted');raise TimeoutError('response lost')
   with self.assertRaises(TimeoutError):authorize_launch(folder,{'cap':6000},push)
   with self.assertRaises(FileExistsError):authorize_launch(folder,{'cap':6000},push)
   self.assertEqual(calls,['accepted']);self.assertTrue((folder/'launch-attempt.json').exists())
 def test_admission_requires_exclusive_slot_and_whole_pair_quota(self):
  self.assertTrue(admit([],9735.707,0,12000))
  self.assertFalse(admit([('other','RUNNING')],9735.707,0,12000))
  self.assertFalse(admit([],9735.707,1,12000))
  self.assertFalse(admit([],100000,0,12000))
 def fixture(self,root):
  bench=json.loads((GOLD/'benchmark.json').read_text());(root/'benchmark.json').write_text(json.dumps(bench))
  (root/'serve.log').write_text('SERVER operational')
  (root/'kernel.log').write_text('GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition capability: (12, 0)\nSERVER READY at 676s')
  for g in bench['game_runs']:(root/(g['game_id']+'_p0_requests.jsonl')).write_text('{"event":"response","usage":{"completion_tokens":1}}\n')
  return bench,{g['game_id'] for g in bench['game_runs']}
 def test_healthy_control_fixture_and_unhealthy_platform(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);bench,ids=self.fixture(root)
   self.assertTrue(assess_output(root,'COMPLETE',ids)['valid'])
   self.assertFalse(assess_output(root,'ERROR',ids)['valid'])
 def test_wrong_gpu_cannot_authorize_treatment(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);bench,ids=self.fixture(root)
   (root/'kernel.log').write_text('GPU: NVIDIA Tesla T4 capability: (7, 5)\nSERVER READY at 676s')
   self.assertFalse(assess_output(root,'COMPLETE',ids)['valid'])
 def test_short_gameplay_and_incomplete_collection_cannot_authorize_treatment(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);bench,ids=self.fixture(root)
   bench['end_time']=bench['start_time'];(root/'benchmark.json').write_text(json.dumps(bench))
   self.assertFalse(assess_output(root,'COMPLETE',ids)['valid'])
   self.fixture(root);(root/'serve.log').unlink()
   self.assertFalse(assess_output(root,'COMPLETE',ids)['valid'])
 def test_analyzer_failure_notes_cannot_authorize_treatment(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);bench,ids=self.fixture(root)
   bench['game_runs'][0]['solver_note']='analyzer failed 11 times consecutively';(root/'benchmark.json').write_text(json.dumps(bench))
   self.assertFalse(assess_output(root,'COMPLETE',ids)['valid'])
 def test_real_archived_traces_have_expected_response_count(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'output';out.mkdir()
   for p in GOLD.glob('*_requests.jsonl'):(out/p.name).symlink_to(p)
   receipt=request_receipts(out)
   self.assertEqual(receipt['totals']['responses'],1436)
   self.assertEqual(receipt['totals']['completion_tokens'],2699790)
   self.assertEqual(receipt['totals'].get('automatic_image_request_exposures',0),0)
 def test_image_receipt_binds_actual_png_and_rejects_missing_image(self):
  import base64,hashlib
  with tempfile.TemporaryDirectory() as d:
   out=Path(d)/'output';out.mkdir();png=(Path(__file__).parent/'tn36-peak.png').read_bytes()
   parts=[{'type':'text','text':'Animation image (frame 1 of 6): real frame'},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}]
   events=[{'event':'request','messages':[{'role':'user','content':parts}]},{'event':'response','usage':{'completion_tokens':7}}]
   p=out/'example_p0_requests.jsonl';p.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
   receipt=request_receipts(out);self.assertEqual(receipt['images'][0]['png_sha256'],hashlib.sha256(png).hexdigest())
   events[0]['messages'][0]['content']=parts[:1];p.write_text('\n'.join(json.dumps(e) for e in events)+'\n')
   with self.assertRaises(ValueError):request_receipts(out)
if __name__=='__main__':unittest.main()
