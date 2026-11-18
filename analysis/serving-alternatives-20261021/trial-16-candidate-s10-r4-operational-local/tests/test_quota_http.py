import base64,copy,datetime as dt,hashlib,json,os,socket,unittest
from pathlib import Path
from types import SimpleNamespace
import quota_wire,reconciliation
from unittest.mock import patch
import launch
from quota_http_fixture import HttpApi,BODY,RAW,requests,KaggleClient
@unittest.skipIf(KaggleClient is None,'installed SDK required')
class HttpQuota(unittest.TestCase):
    def read(self,api):
        with patch.object(socket.socket,'connect',side_effect=AssertionError('NETWORK FORBIDDEN')):receipt=launch.quota_readback(api)
        proof=os.environ.get('PACKET_PROOF_DIR')
        if proof:
            self.case_count=getattr(self,'case_count',0)+1;target=Path(proof)/'http-transport'/self._testMethodName/str(self.case_count);target.mkdir(parents=True)
            (target/'RECEIPT.json').write_text(json.dumps(receipt,indent=2))
        return receipt
    def test_actual_transport_preserves_days_and_exact_captured_body(self):
        api=HttpApi();receipt=self.read(api)
        self.assertNotIn('parse_error',receipt,receipt)
        self.assertEqual(receipt['total_allowed_s'],108000);self.assertEqual(receipt['minimum_allowed_s'],108000)
        self.assertEqual(len(api.sends),1);self.assertEqual(len(api.prior_calls),1)
        self.assertIs(api.clients[0][0]._response_processor,api.clients[0][1])
        self.assertEqual(base64.b64decode(receipt['http']['body_base64']),BODY)
        self.assertEqual(receipt['http']['sha256'],'34d4ca81483ff40c6341d6f20f6464e82a70a458d66622aa5d43aa6879c83355')
        self.assertEqual(receipt['http']['bytes'],277);self.assertNotIn('CPU_SECRET',json.dumps(receipt))

    def test_fractional_usage_joins_wire_and_total_seconds_without_reserialization(self):
        from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsResponse as Q
        for value in ('1.5s','77433.978s'):
            body=copy.deepcopy(RAW);body['gpuQuota']['timeUsed']=value;api=HttpApi([body])
            with self.subTest(value=value),patch.object(Q,'to_dict',side_effect=AssertionError('SDK reserialization is not authority')):
                receipt=self.read(api)
            self.assertNotIn('parse_error',receipt,receipt);self.assertEqual(receipt['used_s'],float(value[:-1]));self.assertEqual(receipt['total_allowed_s'],108000)
            reconciliation.validate_quota(receipt,reconciliation.load_policy(),prelaunch=False)
            with self.assertRaises(ValueError):reconciliation.validate_quota(receipt,reconciliation.load_policy())
            self.assertEqual(len(api.sends),1)
    def test_wrong_endpoint_type_disagreement_missing_and_oversize_fail_closed(self):
        cases=[('endpoint',HttpApi(endpoint='https://api.kaggle.com/v1/other')),('query',HttpApi(endpoint=quota_wire.ENDPOINT+'?token=CPU_SECRET')),
               ('response-type',HttpApi(response_type=SimpleNamespace())),('missing-callback',HttpApi(missing=True)),
               ('typed-disagreement',HttpApi(mutate_typed=lambda q:setattr(q.gpu_quota,'time_used',dt.timedelta(seconds=1)))),
               ('oversize',HttpApi([b' '*(quota_wire.MAX_BODY+1)+BODY])),('corrupt-json',HttpApi([b'{not JSON'])),
               ('missing-body',HttpApi([b''])),('wrong-status',HttpApi(status=403)),('wrong-content-type',HttpApi(content_type='text/plain')),
               ('prior-callback-error',HttpApi(prior_error=True))]
        for name,api in cases:
            with self.subTest(name=name):
                receipt=self.read(api);self.assertIn('parse_error',receipt);self.assertEqual(len(api.sends),1)
                self.assertIs(api.clients[0][0]._response_processor,api.clients[0][1]);self.assertNotIn('CPU_SECRET',json.dumps(receipt))
                with self.assertRaises((ValueError,KeyError,TypeError)):reconciliation.validate_quota(receipt,reconciliation.load_policy())
                if receipt['http'] is not None:self.assertLessEqual(len(base64.b64decode(receipt['http']['body_base64'])),quota_wire.MAX_BODY)
        self.assertEqual(len(cases[-1][1].prior_calls),1)
    def test_durable_wire_tamper_or_missing_metadata_cannot_qualify(self):
        receipt=self.read(HttpApi());policy=reconciliation.load_policy()
        for key in ('http','schema','capture_count','sdk_response_type'):
            value=copy.deepcopy(receipt);value.pop(key)
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError,TypeError)):reconciliation.validate_quota(value,policy)
        value=copy.deepcopy(receipt);value['sdk_response_type']='OtherResponse'
        with self.assertRaises(ValueError):reconciliation.validate_quota(value,policy)
        for key,replacement in [('sha256','changed'),('bytes',1),('status',500),('endpoint','other'),('complete',False),('body_base64','%%%')]:
            value=copy.deepcopy(receipt);value['http'][key]=replacement
            with self.subTest(key=key),self.assertRaises((ValueError,KeyError,TypeError)):reconciliation.validate_quota(value,policy)
