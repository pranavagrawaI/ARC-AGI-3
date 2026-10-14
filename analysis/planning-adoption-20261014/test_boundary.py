"""Public wire projection and no-success-on-empty audit regressions."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent


def module(name,file):
    spec=importlib.util.spec_from_file_location(name,HERE/'runtime'/file)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


class BoundaryTests(unittest.TestCase):
    def test_known_wire_metadata_is_removed_without_mutating_actor_messages(self):
        variants=module('wire_boundary','variants.py')
        original=[{'role':'user','content':'keep','_arc3_control':'resume'},
                  {'role':'assistant','reasoning_content':'preserve','tool_calls':[{'index':0,'id':'x','type':'function','function':{'name':'python','arguments':'{}'}}]}]
        snapshot=copy.deepcopy(original)
        result=variants.wire_messages(original)
        self.assertEqual(original,snapshot)
        self.assertEqual(result[0],{'role':'user','content':'keep'})
        self.assertEqual(result[1]['reasoning_content'],'preserve')
        self.assertEqual(result[1]['tool_calls'][0],{'id':'x','type':'function','function':{'name':'python','arguments':'{}'}})

    def test_empty_and_partial_matrix_are_never_qualified(self):
        audit=module('boundary_audit','audit.py')
        self.assertEqual(audit.summarize([],[])['comparisons']['both']['gate'],'INCOMPLETE_OR_INVALID')
        expected={'id':'missing','checkpoint':'b1','seed':1701,'variant':'control'}
        row=audit.audit_episode(HERE/'does-not-exist',expected)
        self.assertFalse(row['valid']);self.assertIn('missing result.json',row['issues'])


if __name__=='__main__':unittest.main()
