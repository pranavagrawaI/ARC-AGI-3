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
        new=catalog.load()
        proof=json.loads((catalog.HERE/'catalog-proof.json').read_text())
        self.assertEqual(len(new['records']),1369)
        self.assertEqual(proof['catalog_sha'],catalog.CATALOG_SHA)
        self.assertTrue(proof['semantic_wire_png_caps_fifo_unchanged'])
        self.assertTrue(all(len(r['prepared_body_sha256'])==64 for r in new['records'].values()))
    def test_audit_key_drift_rejected(self):
        import tempfile
        old=catalog.load();old['records'].pop(next(iter(old['records'])))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'drift.json';path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError,'catalog identity'):catalog.load(path)
