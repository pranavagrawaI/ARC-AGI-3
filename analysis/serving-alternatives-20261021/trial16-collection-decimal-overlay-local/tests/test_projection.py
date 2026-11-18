import base64,copy,datetime as dt,hashlib,importlib.util,json,os,socket,sys,unittest
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch
from binding import ROOT,DEADLINE
sys.path.extend([str(ROOT/'tests'),str(ROOT/'collection_tools')])
from quota_http_fixture import HttpApi
import decimal_quota as d
import quota_wire,reconciliation
HERE=Path(__file__).resolve().parents[1]
BODY=(HERE/'evidence/quota-response-body.json').read_bytes();RAW=json.loads(BODY)
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
class Projection(unittest.TestCase):
    def read(self,body):
        api=HttpApi([body])
        with patch.object(socket.socket,'connect',side_effect=AssertionError('NETWORK FORBIDDEN')):value=d.read(api)
        self.assertEqual(len(api.sends),1);self.assertEqual(len(api.prior_calls),1);self.assertIs(api.clients[0][0]._response_processor,api.clients[0][1])
        target=os.environ.get('OVERLAY_PROOF_DIR')
        if target:
            self.case_count=getattr(self,'case_count',0)+1;path=Path(target)/self._testMethodName/str(self.case_count);path.mkdir(parents=True)
            (path/'RECEIPT.json').write_text(json.dumps(value,indent=2))
        return value
    def test_exact_captured_ns_and_required_boundaries(self):
        cases=[('42.306700500s',42306700),('0.000000001s',0),('0.000000999s',0),('0.000001000s',1),('108000.123456789s',108000123456),('77433.978s',77433978000),('1.5s',1500000)]
        for text,expected in cases:
            raw=copy.deepcopy(RAW);raw['gpuQuota']['timeUsed']=text
            receipt=self.read(BODY if text=='42.306700500s' else raw)
            self.assertNotIn('parse_error',receipt,receipt);self.assertEqual(d.verify_projection(receipt)['used_s'],Decimal(text[:-1]))
            self.assertEqual(receipt['typed_projection']['used_s']['total_microseconds'],expected)
            self.assertEqual(receipt['wire_seconds']['used_s'],str(Decimal(text[:-1])))
        self.assertEqual(hashlib.sha256(BODY).hexdigest(),'b2dbe486e364d6e4b6cdf782d99a92f3edcdf4e75e49ac9268de5c012bfe3276')
    def test_integer_component_and_nanosecond_authority_mismatch_rejected(self):
        receipt=self.read(BODY);policy=reconciliation.load_policy();self.assertEqual(d.validate(receipt,policy)['used_s'],Decimal('42.306700500'))
        for key in ('days','seconds','microseconds','total_microseconds'):
            value=copy.deepcopy(receipt);value['typed_projection']['used_s'][key]+=1
            with self.subTest(key=key),self.assertRaises(ValueError):d.validate(value,policy)
        value=copy.deepcopy(receipt);value['wire_seconds']['used_s']='42.306700'
        with self.assertRaises(ValueError):d.validate(value,policy)
        raw=copy.deepcopy(RAW);raw['gpuQuota']['totalTimeAllowed']='108000.000000001s'
        value=self.read(raw);self.assertNotIn('parse_error',value)
        with self.assertRaises(ValueError):d.validate(value,policy)
    def test_terminal_checks_exact_reserved_ns_status_and_window(self):
        overlay=load('overlay_reduce',HERE/'reduce.py');receipt=self.read(BODY)
        self.assertTrue(overlay.reporting({'status':{'status':'ERROR'},'typed_quota':receipt})['terminal_collection_complete'])
        self.assertFalse(overlay.reporting({'status':{'status':'RUNNING'},'typed_quota':receipt})['terminal_collection_complete'])
        raw=copy.deepcopy(RAW);raw['gpuQuota']['timeReserved']='0.000000001s';reserved=self.read(raw)
        self.assertEqual(reserved['typed_projection']['reserved_s']['total_microseconds'],0)
        self.assertFalse(overlay.reporting({'status':{'status':'ERROR'},'typed_quota':reserved})['terminal_collection_complete'])
        raw['quotaRefreshTime']='2026-11-07T00:00:00Z';changed=self.read(raw)
        self.assertFalse(overlay.reporting({'status':{'status':'ERROR'},'typed_quota':changed})['terminal_collection_complete'])
        with self.assertRaises(ValueError):d.validate(receipt,reconciliation.load_policy(),dt.datetime.fromisoformat(receipt['utc'])+dt.timedelta(seconds=121))
    def test_frozen_validator_and_failed_receipt_remain_failed(self):
        old=json.loads((HERE/'evidence/READBACK.json').read_text())['typed_quota']
        with self.assertRaises(ValueError):reconciliation.validate_quota(old,reconciliation.load_policy(),prelaunch=False)
        new=self.read(BODY)
        with self.assertRaises((ValueError,KeyError)):reconciliation.validate_quota(new,reconciliation.load_policy(),prelaunch=False)
        original=load('frozen_collection_reduce',ROOT/'collection_tools/reduce.py')
        self.assertFalse(original.reporting({'status':{'status':'ERROR'},'typed_quota':old})['terminal_collection_complete'])
        self.assertFalse(original.reporting({'status':{'status':'ERROR'},'typed_quota':new})['terminal_collection_complete'])
    def test_original_source_archive_and_deadline_gates_exact(self):
        old=(ROOT/'collection_tools/reduce.py').read_text();new=(HERE/'reduce.py').read_text()
        self.assertEqual(new[new.index('def reduce('):],old[old.index('def reduce('):])
        old=(ROOT/'collection_tools/monitor.py').read_text();new=(HERE/'monitor.py').read_text()
        self.assertEqual(new[new.index('def main('):],old[old.index('def main('):])
        old=(ROOT/'collection_tools/readback.py').read_text();new=(HERE/'readback.py').read_text()
        self.assertEqual(new[new.index('def main('):new.index("        out['typed_quota']")],old[old.index('def main('):old.index("        out['typed_quota']")])
        attempt=json.loads((ROOT/'launch-attempt.json').read_text())
        self.assertEqual(dt.datetime.fromisoformat(attempt['utc'])+dt.timedelta(seconds=5700),dt.datetime.fromisoformat(DEADLINE))
    def test_parser_pin_and_transport_rejections(self):
        with patch.object(d,'SERIALIZER_SHA256','wrong'):
            result=d.read(HttpApi([BODY]));self.assertIn('parse_error',result)
        for api in (HttpApi([BODY],endpoint='https://api.kaggle.com/v1/other'),HttpApi([BODY],response_type=object()),HttpApi([b'{bad JSON']),HttpApi([BODY],mutate_typed=lambda q:setattr(q.gpu_quota,'time_used',dt.timedelta(seconds=42,microseconds=306701)))):
            result=d.read(api);self.assertIn('parse_error',result)
            with self.assertRaises((ValueError,KeyError,TypeError)):d.validate(result,reconciliation.load_policy())
