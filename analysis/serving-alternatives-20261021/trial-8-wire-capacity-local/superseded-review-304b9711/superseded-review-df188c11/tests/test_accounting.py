import copy
import json
import unittest
from pathlib import Path
import accounting
import catalog

class Accounting(unittest.TestCase):
    def response(self,arm,cached=0):
        d={'cached_tokens':cached}
        d.update({'image_tokens':800} if arm=='control' else {'multimodal_tokens':{'image':800}})
        return {'usage':{'prompt_tokens':999,'completion_tokens':5,'total_tokens':1004,'prompt_tokens_details':d},'choices':[{'finish_reason':'length','message':{'content':'hello'}}]}
    def test_totals_and_inferred_label_hits_and_misses(self):
        record={'images':['a','b'],'local_prompt':999,'cap':5}
        for arm in ('control','candidate'):
            for cached in (0,999):
                result=accounting.observe(arm,record,self.response(arm,cached))
                self.assertEqual(result['inferred_image_object_count'],2)
                self.assertEqual(result['observed_api_image_tokens'],800)
                self.assertEqual(result['image_count_observation'],'INFERRED_AGGREGATE_ONLY')
    def test_missing_wrong_truncated_rejected(self):
        record={'images':['a','b'],'local_prompt':999,'cap':5}
        for arm in ('control','candidate'):
            for mutation in ('missing','wrong','prompt','finish','length'):
                b=self.response(arm)
                if mutation=='missing':del b['usage']['prompt_tokens_details']
                if mutation=='wrong':b['usage']['prompt_tokens_details']={'image_tokens':400,'multimodal_tokens':{'image':400}}
                if mutation=='prompt':b['usage']['prompt_tokens']=998
                if mutation=='finish':b['choices'][0]['finish_reason']=None
                if mutation=='length':b['usage']['completion_tokens']=6
                with self.assertRaises((ValueError,KeyError)):accounting.observe(arm,record,b)
    def test_all_1369_catalog_proof(self):
        new,proof=catalog.build()
        self.assertEqual(proof['records'],1369)
        self.assertTrue(proof['every_count_delta_11'])
        self.assertTrue(proof['source_wire_image_caps_unchanged'])
        self.assertEqual(new['streams'],json.loads((catalog.PRIOR/'trial-4-saturated-local/catalog.json').read_text())['streams'])
    def test_audit_key_drift_rejected(self):
        old=json.loads((catalog.PRIOR/'trial-4-saturated-local/catalog.json').read_text())
        rows=list(map(json.loads,(catalog.PRIOR/'trial-5-serving-screen/prompt-audit/records.jsonl').read_bytes().splitlines()))
        rows[-1]=copy.deepcopy(rows[0])
        with self.assertRaisesRegex(ValueError,'keyset'):catalog.corrected(old,rows)
