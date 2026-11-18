import json
from pathlib import Path
import tempfile
import unittest

from ledger_screen import measure, read_prefix


def transition(step, before, after):
    return {'step': step, 'action': 'UP', 'before': {'grid': before, 'levels_completed': 0},
            'after': {'grid': after, 'levels_completed': 0}}


class LedgerTests(unittest.TestCase):
    def test_literal_counts_and_coordinates(self):
        rows = [transition(1, [[0, 0, 0]], [[1, 0, 0]]),
                transition(2, [[1, 0, 0]], [[1, 2, 2]]),
                transition(3, [[1, 2, 2]], [[1, 2, 3]])]
        packet = measure(rows)
        self.assertEqual([r['changed_cells'] for r in packet['ledger']], [1, 2, 1])
        self.assertEqual(packet['ledger'][1]['changes_y_x_before_after'], [[0, 1, 0, 2], [0, 2, 0, 2]])
        self.assertEqual(packet['current_frame']['grid_rows'], ['123'])

    def test_shape_change_is_not_zero_impact(self):
        result = measure([transition(1, [[0]], [[0, 0]])])
        self.assertIsNone(result['ledger'][0]['changed_cells'])
        self.assertFalse(result['ledger'][0]['comparable_shapes'])

    def test_hex_frame_roundtrip(self):
        grid = [[0, 9, 10, 15], [4, 5, 6, 7]]
        packet = measure([transition(1, grid, grid)])
        decoded = [[int(v, 16) for v in row] for row in packet['current_frame']['grid_rows']]
        self.assertEqual(decoded, grid)

    def test_future_data_never_parsed(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'transitions.jsonl'
            path.write_text(json.dumps(transition(1, [[0]], [[1]])) + '\nINVALID FUTURE RECORD\n')
            self.assertEqual(len(read_prefix(path, 1)), 1)

    def test_noncontiguous_history_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'transitions.jsonl'
            path.write_text(json.dumps(transition(2, [[0]], [[1]])) + '\n')
            with self.assertRaises(ValueError):
                read_prefix(path, 1)


if __name__ == '__main__':
    unittest.main()
