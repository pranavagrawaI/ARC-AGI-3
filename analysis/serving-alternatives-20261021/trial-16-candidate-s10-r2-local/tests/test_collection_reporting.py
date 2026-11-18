import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'collection_tools'))
import reduce as collection_reduce
class Reporting(unittest.TestCase):
    def test_missing_typed_or_nonterminal_is_no_terminal_claim(self):
        for value in ({'status':{'status':'RUNNING'},'typed_quota':{'used_s':77433.978,'reserved_s':0}}, {'status':{'status':'ERROR'}}, {'status':{'status':'ERROR'},'typed_quota':{'used_s':77433.978,'reserved_s':1}}):
            result=collection_reduce.reporting(value);self.assertFalse(result['terminal_collection_complete']);self.assertFalse(result['slot_release_claim']);self.assertTrue(result['collection_gaps'])
        value=collection_reduce.reporting({'status':{'status':'ERROR'},'typed_quota':{'used_s':77433.978,'reserved_s':0}})
        self.assertTrue(value['terminal_collection_complete']);self.assertFalse(value['slot_release_claim'])
