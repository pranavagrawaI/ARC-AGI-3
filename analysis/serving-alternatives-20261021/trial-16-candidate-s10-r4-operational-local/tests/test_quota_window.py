"""Real installed SDK objects and fresh_checks; only remote endpoints are fixtures."""
import contextlib,copy,datetime as dt,json,os,shutil,socket,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import launch,reconciliation as r
try:
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsResponse
    import requests
except ImportError:ApiGetAcceleratorQuotaStatisticsResponse=None;requests=None
HERE=Path(__file__).resolve().parents[1]
from quota_http_fixture import HttpApi,BODY,RAW
class Record:
    def __init__(self,raw):self.raw=raw;self.ref=raw.get('ref','')
    def to_dict(self):return self.raw
class FakeApi(HttpApi):
    def __init__(self,raws=None,named=False):
        super().__init__(raws or [BODY,BODY]);self.raws=self.bodies;self.calls=[];self.named=named
    @contextlib.contextmanager
    def build_kaggle_client(self):
        self.calls.append('quota')
        with super().build_kaggle_client() as client:yield client
    def kernels_list(self,**kwargs):
        self.calls.append('recent');rows=[{'ref':'owner/completed'}]
        if not self.named:rows.insert(0,dict(r.STUB))
        return [Record(v) for v in rows]
    def kernels_list_with_response(self,**kwargs):
        self.calls.append(kwargs['search']);return Record({'kernels':[dict(r.STUB)]} if kwargs['search']=='serving-candidate-feas5-20261023' else {})
    def kernels_status(self,ref):
        self.calls.append(ref)
        if ref==r.PRIOR_REF:
            response=requests.Response();response.status_code=404;response._content=b'{"error":{"code":404,"message":"No runs found for this kernel."}}'
            raise requests.HTTPError('CPU_404',response=response)
        return SimpleNamespace(status=SimpleNamespace(name='ERROR' if ref==r.PREDECESSOR_REF else 'COMPLETE'))
@unittest.skipIf(ApiGetAcceleratorQuotaStatisticsResponse is None or requests is None,'installed SDK interpreter required')
class Window(unittest.TestCase):
    def run_fresh(self,api):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        (self.root/'CPU-FIXTURE-INPUTS.json').write_text(json.dumps([{'HTTP_body_base64':__import__('base64').b64encode(v).decode()} if isinstance(v,bytes) else v for v in api.raws],indent=2))
        def forbidden(*a,**k):raise AssertionError('NETWORK FORBIDDEN')
        with patch.object(socket.socket,'connect',forbidden),patch.object(launch,'HERE',self.root),patch.object(launch.collection_space,'check',return_value={'eligible':True,'CPU_ONLY':True}):
            try:return launch.fresh_checks(api)
            except BaseException as error:
                (self.root/'CPU-FIXTURE-REJECTION.json').write_text(json.dumps({'error':repr(error),'calls':api.calls},indent=2));raise
            finally:
                proof=os.environ.get('PACKET_PROOF_DIR')
                if proof:
                    self.case_count=getattr(self,'case_count',0)+1
                    destination=Path(proof)/self._testMethodName/str(self.case_count);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(self.root,destination)
    def test_old_predecessor_and_new_window_pass_real_fresh_checks(self):
        api=FakeApi();e=self.run_fresh(api)
        self.assertEqual(e['judgment']['classification'],'RECONCILED_FAILED_SAVE_NO_RUN_OBSERVED');self.assertEqual(api.calls.count('quota'),2)
        self.assertEqual(r.load_policy()['historical_predecessor_used_s'],77433.978)
        for field in ('quota_before','quota_after'):
            self.assertEqual(e[field]['used_s'],0);self.assertEqual(e[field]['reserved_s'],0);self.assertEqual(e[field]['total_allowed_s'],108000)
            self.assertEqual(e[field]['refresh_time_utc'],'2026-10-31T00:00:00+00:00');self.assertEqual(__import__('base64').b64decode(e[field]['http']['body_base64']),BODY)
        self.assertFalse((self.root/'launch-attempt.json').exists())
    def test_missing_or_changed_window_allowance_and_moved_quota_fail_closed(self):
        variants=[]
        def change(label,fn):
            raw=copy.deepcopy(RAW);fn(raw);variants.append((label,raw))
        change('missing-refresh',lambda v:v.pop('quotaRefreshTime'))
        change('changed-refresh',lambda v:v.update(quotaRefreshTime='2026-11-07T00:00:00.000Z'))
        change('missing-total',lambda v:v['gpuQuota'].pop('totalTimeAllowed'))
        change('changed-total',lambda v:v['gpuQuota'].update(totalTimeAllowed='21600s'))
        change('insufficient-total',lambda v:v['gpuQuota'].update(totalTimeAllowed='5000s'))
        change('missing-minimum',lambda v:v['gpuQuota'].pop('minimumTimeAllowed'))
        change('changed-minimum',lambda v:v['gpuQuota'].update(minimumTimeAllowed='20000s'))
        change('moved-used',lambda v:v['gpuQuota'].update(timeUsed='1s'))
        change('reserved',lambda v:v['gpuQuota'].update(timeReserved='1s'))
        change('missing-used',lambda v:v['gpuQuota'].pop('timeUsed'))
        change('malformed-duration',lambda v:v['gpuQuota'].update(totalTimeAllowed='NaNs'))
        for label,raw in variants:
            for phase,raws in [('before',[raw,RAW]),('after',[RAW,raw])]:
                with self.subTest(label=label,phase=phase),self.assertRaises((ValueError,KeyError)):
                    self.run_fresh(FakeApi(raws))
                self.assertFalse((self.root/'launch-attempt.json').exists())
                receipt=json.loads((self.root/('quota-before-prelaunch.json' if phase=='before' else 'quota-after-prelaunch.json')).read_text())
                self.assertEqual(receipt['schema'],'QUOTA_HTTP_TYPED_V1');self.assertIn('parse_error',receipt) if label in ('missing-refresh','missing-total','missing-minimum','missing-used','malformed-duration') else self.assertIsNotNone(receipt['http'])
    def test_all_named_sessions_also_require_second_exact_quota(self):
        api=FakeApi(named=True);pair=self.run_fresh(api);self.assertIn('quota_after',pair);self.assertEqual(api.calls.count('quota'),2)
        self.assertTrue((self.root/'quota-pair-prelaunch.json').exists())
        changed=copy.deepcopy(RAW);changed['gpuQuota']['timeUsed']='1s'
        with self.assertRaisesRegex(ValueError,'quota moved'):
            self.run_fresh(FakeApi([RAW,changed],named=True))
    def test_raw_typed_and_receipt_chronology_cannot_disagree(self):
        api=FakeApi();e=self.run_fresh(api);policy=r.load_policy();now=dt.datetime.now(dt.timezone.utc)
        variants=[]
        for field in ('quota_before','quota_after'):
            for delta in ({'used_s':1},{'total_allowed_s':21600},{'refresh_time_utc':'2026-11-07T00:00:00+00:00'}, {'utc':(now-dt.timedelta(seconds=121)).isoformat()}, {'utc':(now+dt.timedelta(seconds=1)).isoformat()}):
                v=copy.deepcopy(e);v[field].update(delta);variants.append(v)
        v=copy.deepcopy(e);v['quota_before']['utc']=e['finished_utc'];v['quota_after']['utc']=e['started_utc'];variants.append(v)
        for index,v in enumerate(variants):
            with self.subTest(index=index),self.assertRaises(ValueError):r.evaluate(v,policy,now)
        for key in ('total_allowed_s','refresh_time_utc'):
            changed=copy.deepcopy(policy);changed['current_quota'].pop(key)
            with self.assertRaises((ValueError,KeyError)):r.evaluate(e,changed,now)
