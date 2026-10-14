"""Native replay checks actual completion rather than trusting result labels."""
import json
from pathlib import Path
import unittest
from audit import native_level_count

ROOT=Path(__file__).resolve().parents[2]


class NativeAuditTests(unittest.TestCase):
    def test_archived_prefix_and_qualified_route_have_different_terminal_depths(self):
        directory=ROOT/'analysis/counterexample-continuations-v3-20261013/output/continuations/episodes/b1-2903-reminder'
        actions=json.loads((directory/'actions.json').read_text())
        self.assertEqual(native_level_count(actions),1)
        route=json.loads((Path(__file__).parent/'evidence/b1-2903-reminder-guided.json').read_text())['path']
        clicks={'Og':12,'Os':6,'Sg':27,'Ss':21,'Yg':42,'Ys':36,'Ng':57,'Ns':51}
        continuation=[{'id':'ACTION6','data':{'x':clicks[key+operation],'y':57}} for operation,key in route]
        self.assertEqual(native_level_count(actions+continuation),2)


if __name__=='__main__':unittest.main()
