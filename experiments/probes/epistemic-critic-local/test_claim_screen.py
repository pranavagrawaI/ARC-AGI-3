import json
import unittest
from claim_screen import datasets, initial_requests, decode_json


class ClaimScreenTests(unittest.TestCase):
    def test_code_fence_changes_no_semantics(self):
        response = '{"judgments":[{"id":"one","status":"supported_in_scope"}]}'
        self.assertEqual(decode_json('```json\n'+response+'\n```'), json.loads(response))
        self.assertEqual(decode_json(response), json.loads(response))
        with self.assertRaises(ValueError):
            decode_json('Here is my answer: '+response)

    def test_measurements_support_literal_oracles(self):
        s5, cd = datasets()
        self.assertEqual([t['changed_cells'] for t in s5['facts']['transitions']], [1, 2, 1])
        self.assertEqual(s5['facts']['current_changed_rows'][0]['color_counts']['4'], 4)
        red = next(c for c in cd['facts']['current_color_measurements'] if c['color'] == 2)
        self.assertEqual((red['cell_count'], red['row_bounds'], red['column_bounds']), (30, [24, 32], [25, 38]))

    def test_single_and_list_conditions_share_evidence_and_claims(self):
        data = datasets()
        requests = dict(initial_requests(data))
        for d in data:
            batch = json.loads(requests[d['id']+'-list']['messages'][1]['content'])
            for c in d['claims']:
                single = json.loads(requests[d['id']+'-single-'+c['id']]['messages'][1]['content'])
                self.assertEqual(single['facts'], batch['facts'])
                self.assertIn(single['claims'][0], batch['claims'])

    def test_extraction_has_no_evidence_or_answers(self):
        data = datasets()
        for name, body in initial_requests(data):
            packet = json.loads(body['messages'][1]['content'])
            self.assertNotIn('expected', packet)
            self.assertEqual(len(body['messages']), 2)
            if name.endswith('-extract'):
                self.assertEqual(set(packet), {'narrative'})


if __name__ == '__main__':
    unittest.main()
