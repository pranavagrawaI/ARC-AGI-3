import importlib,sys,tempfile,json,unittest
from pathlib import Path
from unittest.mock import patch
import candidate_reduce,catalog,api_arm
OLD=Path(__file__).resolve().parents[2]/'trial-10-exact-wire-local/tests'
sys.path.append(str(OLD));sys.modules['reduce_api']=api_arm
fixture=importlib.import_module('test_reducer').fixture
class CandidateArm(unittest.TestCase):
    def test_public_candidate_only_reduction_preserves_full_window_and_images(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cat,refs=fixture(root,'candidate',2000,[100,700,800,1000,1100,1300,1400])
            with patch.object(catalog,'load',return_value=cat):value=candidate_reduce.arm(root)
            self.assertTrue(value['valid'],value);self.assertEqual(value['completed'],60);self.assertEqual(value['U_resp'],20)
            self.assertNotIn('ratio',value);self.assertNotIn('control',value)
            body=refs[0];data=json.loads(body.read_bytes());data['usage']['prompt_tokens_details']['multimodal_tokens']['image']=0;body.write_bytes(catalog.canonical(data))
            with patch.object(catalog,'load',return_value=cat):self.assertFalse(candidate_reduce.arm(root)['valid'])
    def test_complete_bad_api_body_negative_missing_body_inconclusive(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);cat,refs=fixture(root,'candidate',2000,[100,700,800,1000,1100,1300,1400]);body=refs[0]
            data=json.loads(body.read_bytes());data['choices'][0]['finish_reason']='INVALID';raw=catalog.canonical(data);body.write_bytes(raw)
            path=body.parent/'http.json';receipt=json.loads(path.read_text());receipt.update(headers=[['Content-Length',str(len(raw))]],received_bytes=len(raw),persisted_bytes=len(raw));path.write_bytes(catalog.canonical(receipt))
            with patch.object(catalog,'load',return_value=cat):value=candidate_reduce.complete_api_failure(root)
            self.assertEqual(value['kind'],'COMPLETE_API_CONTRACT_FAILURE');self.assertTrue(value['raw_verified'])
            body.unlink()
            with patch.object(catalog,'load',return_value=cat),self.assertRaises(FileNotFoundError):candidate_reduce.complete_api_failure(root)
    def test_real_complete_http500_negative_without_claiming_a_passed_window(self):
        import test_replay
        helper=test_replay.ReplayTests();self.addCleanup(helper.doCleanups)
        root,server,obj,events,ok,elapsed=helper.run_fixture('error',(.1,.8));self.assertFalse(ok)
        (root/'out').rename(root/'candidate')
        # The generic retained server's successful sibling uses control accounting;
        # make that explicit CPU response candidate-valid so only real500 is negative.
        for record in obj.cat['records'].values():record.update(local_prompt=999,cap=5)
        for path in (root/'candidate').glob('*/http.json'):
            receipt=json.loads(path.read_text())
            if receipt['status']==200 and receipt['complete']:
                body=path.parent/'body.bin';value=json.loads(body.read_bytes());value['usage']['prompt_tokens_details']['multimodal_tokens']={'image':0};raw=catalog.canonical(value);body.write_bytes(raw)
                receipt.update(headers=[['Content-Length',str(len(raw))]],received_bytes=len(raw),persisted_bytes=len(raw));path.write_bytes(catalog.canonical(receipt))
        with patch.object(catalog,'load',return_value=obj.cat):value=candidate_reduce.complete_api_failure(root)
        self.assertEqual(value['kind'],'COMPLETE_HTTP_FAILURE');self.assertEqual(value['status'],500)
        # A known failure drain need not be an accepted1500s window/heartbeat.
        events[-1]['t']+=3
        candidate_reduce.failure_log(events)
        with self.assertRaises(ValueError):api_arm.contract.validate_log(events,'replay_start')
