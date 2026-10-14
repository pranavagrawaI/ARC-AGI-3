"""Audit known external outcomes, including empty and invalid coverage."""
import json
from pathlib import Path
import tempfile
import unittest
from audit import audit_episode, summarize


class AuditTests(unittest.TestCase):
    def test_paired_screen_counts_gains_losses_and_invalid_pairs(self):
        matrix=[];rows=[]
        for seed in [1,2]:
            for variant in ['control','planner','probing','both']:
                row={'checkpoint':'b1','seed':seed,'variant':variant}
                matrix.append(row);rows.append({**row,'valid':True,'solved':variant!='control'})
        result=summarize(rows,matrix)
        self.assertEqual(result['comparisons']['planner']['gate'],'PASS_SCREEN')
        self.assertEqual(result['comparisons']['planner']['gains'],2)
        rows[1]['valid']=False
        self.assertEqual(summarize(rows,matrix)['comparisons']['planner']['gate'],'INCOMPLETE_OR_INVALID')
        rows[1]['valid']=True;rows[0]['solved']=True;rows[1]['solved']=False
        self.assertEqual(summarize(rows,matrix)['comparisons']['planner']['losses'],1)

    def test_empty_and_duplicate_results_never_pass(self):
        self.assertEqual(summarize([],[])['comparisons']['planner']['gate'],'INCOMPLETE_OR_INVALID')
        row={'checkpoint':'b1','seed':1,'variant':'control','valid':True}
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            summarize([row,row],[row,row])

    def fixture(self, root):
        expected={'checkpoint':'b1','seed':1,'variant':'planner'}
        files={
            'result.json':{**expected,'mode':'live','status':'token_budget','unresolved':None,'qualified_restoration':True,
                           'protected_inputs_unchanged':True,'new_model_calls':1,'tokens':10,'levels_completed':1,'score':2.777},
            'checkpoint.json':{'qualified':True,'actions_match':True,'prefix_actions':[],
                               'comparisons':[{'visible_content_identical':True,'tools_identical':True}]},
            'restoration.json':{'actions_match':True},
            'first-request.json':{'messages':[],'tools':[]},
            'intervention.json':{'all_other_messages_identical':True,'original_system_prefix_preserved':True},
            'actions.json':[]}
        for name,value in files.items():(root/name).write_text(json.dumps(value))
        events=[{'event':'request','call':1,'body':{'max_tokens':10,'seed':1,'messages':[],'tools':[]}},
                {'event':'response','call':1,'status':200,'body':{'usage':{'completion_tokens':10}}}]
        (root/'wire.jsonl').write_text('\n'.join(json.dumps(row) for row in events))
        return expected,files,events

    def test_valid_receipts_and_corrupt_or_fixture_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);expected,files,events=self.fixture(root)
            self.assertTrue(audit_episode(root,expected,verify_native=lambda actions:1)['valid'])
            self.assertFalse(audit_episode(root,expected,verify_native=lambda actions:2)['valid'])
            original=files['checkpoint.json']
            (root/'checkpoint.json').write_text(json.dumps({**original,'comparisons':[]}))
            self.assertFalse(audit_episode(root,expected,verify_native=lambda actions:1)['valid'])
            (root/'checkpoint.json').write_text(json.dumps(original))
            for change in [{'mode':'fixture'},{'unresolved':{'call':1}},{'tokens':11},{'seed':2},{'status':'http_error'}]:
                (root/'result.json').write_text(json.dumps({**files['result.json'],**change}))
                self.assertFalse(audit_episode(root,expected,verify_native=lambda actions:1)['valid'])
            (root/'result.json').write_text(json.dumps(files['result.json']))
            events[1]['body']['usage']['completion_tokens']=True
            (root/'wire.jsonl').write_text('\n'.join(json.dumps(row) for row in events))
            self.assertFalse(audit_episode(root,expected,verify_native=lambda actions:1)['valid'])
            (root/'wire.jsonl').unlink()
            self.assertFalse(audit_episode(root,expected,verify_native=lambda actions:1)['valid'])


if __name__=='__main__':unittest.main()
