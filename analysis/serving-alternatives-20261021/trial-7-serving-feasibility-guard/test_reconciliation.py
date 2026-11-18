import copy
import datetime as dt
import unittest
import reconciliation as r

class ReconciliationTests(unittest.TestCase):
    def fixture(self):
        now=dt.datetime.now(dt.timezone.utc)
        e=dict(started_utc=(now-dt.timedelta(seconds=12)).isoformat(),finished_utc=now.isoformat(),prior_kernel_ref=r.PRIOR_REF,predecessor_kernel_ref=r.PREDECESSOR_REF,predecessor_status={'status':'ERROR'},
               quota_before={'used_s':59275.88,'reserved_s':0},quota_after={'used_s':59275.88,'reserved_s':0},
               recent=[{'ref':'','raw':dict(r.STUB),'status':'REDACTED_UNKNOWN'},{'ref':'owner/completed','status':'COMPLETE'}],
               exact_search={'kernels':[dict(r.STUB)]},negative_search={},prior_status={'status':404,'body':'{"error":{"code":404,"message":"No runs found for this kernel.","status":"NOT_FOUND"}}'})
        return now,e,r.load_policy()

    def test_scoped_judgment_preserves_identity_gap(self):
        now,e,policy=self.fixture()
        result=r.evaluate(e,policy,now)
        self.assertEqual(result['classification'],'RECONCILED_FAILED_SAVE_NO_RUN_OBSERVED')
        self.assertFalse(result['terminal_identity_proven']);self.assertFalse(result['kernel_absence_proven'])

    def test_changed_extra_stub_active_or_generic_errors_fail(self):
        now,e,policy=self.fixture()
        variants=[]
        def change(fn):
            v=copy.deepcopy(e);fn(v);variants.append(v)
        change(lambda v:v['recent'].append(copy.deepcopy(v['recent'][0])))
        change(lambda v:v['recent'][0]['raw'].update(lastRunTime='2026-10-23T00:00:00Z'))
        change(lambda v:v['recent'][0]['raw'].update(ref=''))
        change(lambda v:v['recent'][1].update(status='RUNNING'))
        change(lambda v:v['prior_status'].update(status=403))
        change(lambda v:v['prior_status'].update(body='{"error":{"code":404,"message":"Not found"}}'))
        change(lambda v:v['prior_status'].update(body='{}'))
        change(lambda v:v['prior_status'].update(body='not JSON'))
        change(lambda v:v['exact_search'].update(kernels=[]))
        change(lambda v:v.update(negative_search={'kernels':[dict(r.STUB)]}))
        change(lambda v:v['quota_before'].update(used_s=59266))
        change(lambda v:v['quota_after'].update(used_s=59266))
        change(lambda v:v['quota_after'].update(reserved_s=1))
        change(lambda v:v.update(started_utc=(now-dt.timedelta(seconds=121)).isoformat()))
        change(lambda v:v.update(finished_utc=(now+dt.timedelta(seconds=1)).isoformat()))
        change(lambda v:v.update(prior_kernel_ref='owner/other'))
        change(lambda v:v.update(predecessor_kernel_ref='owner/other'))
        change(lambda v:v['predecessor_status'].update(status='RUNNING'))
        for index,v in enumerate(variants):
            with self.subTest(index=index),self.assertRaises(ValueError):r.evaluate(v,policy,now)
        wrong=policy|{'prior_kernel_ref':'owner/other'}
        with self.assertRaises(ValueError):r.evaluate(e,wrong,now)


class LiveBoundaryTests(unittest.TestCase):
    def test_fresh_checks_records_exact_scoped_judgment_and_refuses_changed_stub(self):
        import contextlib
        import json
        from pathlib import Path
        import tempfile
        from types import SimpleNamespace
        from unittest.mock import patch
        import requests
        import launch
        class Record:
            def __init__(self,raw):self.raw=raw;self.ref=raw.get('ref','')
            def to_dict(self):return self.raw
        class Duration:
            def __init__(self,value):self.value=value
            def total_seconds(self):return self.value
        class FakeApi:
            changed=False
            active_predecessor=False
            def __init__(self):self.status_calls=[]
            @contextlib.contextmanager
            def build_kaggle_client(self):
                quota=SimpleNamespace(gpu_quota=SimpleNamespace(time_used=Duration(59275.88),time_reserved=Duration(0)))
                endpoint=SimpleNamespace(get_accelerator_quota_statistics=lambda:quota)
                yield SimpleNamespace(kernels=SimpleNamespace(kernels_api_client=endpoint))
            def kernels_list(self,**kwargs):
                stub=dict(r.STUB)
                if self.changed:stub['lastRunTime']='other'
                return [Record(stub),Record({'ref':'owner/completed'})]
            def kernels_list_with_response(self,**kwargs):
                return Record({'kernels':[dict(r.STUB)]} if kwargs['search']=='serving-candidate-feas5-20261023' else {})
            def kernels_status(self,ref):
                self.status_calls.append(ref)
                if ref==r.PREDECESSOR_REF and self.active_predecessor:return SimpleNamespace(status=SimpleNamespace(name='RUNNING'))
                if ref==r.PRIOR_REF:
                    response=requests.Response();response.status_code=404
                    response._content=b'{"error":{"code":404,"message":"No runs found for this kernel."}}'
                    raise requests.HTTPError('404',response=response)
                return SimpleNamespace(status=SimpleNamespace(name='COMPLETE'))
        with tempfile.TemporaryDirectory() as directory,patch.object(launch,'HERE',Path(directory)):
            api=FakeApi();evidence=launch.fresh_checks(api)
            self.assertEqual(evidence['judgment']['classification'],'RECONCILED_FAILED_SAVE_NO_RUN_OBSERVED')
            self.assertIn(r.PREDECESSOR_REF,api.status_calls)
            self.assertEqual(json.loads((Path(directory)/'failed-save-reconciliation-prelaunch.json').read_text())['judgment'],evidence['judgment'])
            api.changed=True
            with self.assertRaises(ValueError):launch.fresh_checks(api)
            failure=json.loads((Path(directory)/'failed-save-reconciliation-prelaunch.json').read_text())
            self.assertNotIn('judgment',failure)
            api.changed=False;api.active_predecessor=True
            with self.assertRaises(ValueError):launch.fresh_checks(api)
            self.assertEqual(json.loads((Path(directory)/'predecessor-status-prelaunch.json').read_text())['status'],'RUNNING')

if __name__=='__main__':unittest.main()
