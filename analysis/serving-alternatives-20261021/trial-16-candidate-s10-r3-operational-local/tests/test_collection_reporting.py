import copy,json,sys,unittest,datetime as dt
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collection_tools'))
import reduce as collection_reduce
import reconciliation
from test_quota_window import RAW
class Reporting(unittest.TestCase):
    def test_same_window_terminal_usage_separate_from_historical_charge(self):
        quota=dict(utc=dt.datetime.now(dt.timezone.utc).isoformat(),raw_response=copy.deepcopy(RAW),**reconciliation.quota_values(RAW))
        value=dict(status={'status':'ERROR'},typed_quota=quota)
        self.assertTrue(collection_reduce.reporting(value)['terminal_collection_complete'])
        quota['used_s']=12;quota['raw_response']['gpuQuota']['timeUsed']='12s'
        self.assertTrue(collection_reduce.reporting(value)['terminal_collection_complete'])
        for mutation in ('missing','running','reserved','old-window','allowance','historical-mislabel'):
            v=copy.deepcopy(value)
            if mutation=='missing':v.pop('typed_quota')
            if mutation=='running':v['status']['status']='RUNNING'
            if mutation=='reserved':v['typed_quota']['reserved_s']=1;v['typed_quota']['raw_response']['gpuQuota']['timeReserved']='1s'
            if mutation=='old-window':v['typed_quota']['raw_response']['quotaRefreshTime']='2026-10-24T00:00:00.000Z'
            if mutation=='allowance':v['typed_quota']['raw_response']['gpuQuota']['totalTimeAllowed']='108000s'
            if mutation=='historical-mislabel':v['typed_quota']['used_s']=77433.978;v['typed_quota']['raw_response']['gpuQuota']['timeUsed']='77433.978s'
            result=collection_reduce.reporting(v)
            self.assertFalse(result['terminal_collection_complete'],mutation);self.assertFalse(result['slot_release_claim']);self.assertTrue(result['collection_gaps'])
