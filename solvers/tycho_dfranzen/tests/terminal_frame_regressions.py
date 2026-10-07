"""Winning frames reach the actual Actor request without becoming decision frames."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / 'runtime'))
from port_regressions import play
import arcengine
import numpy as np
from taaf.game_examples import ExampleGame
from tycho.workspace.workspace import GameWorkspace, grid_text, _parse_grid_text
from tycho_nooa.models import GameTurn
from hybrid.terminal_frames import context


class CompletionGame(ExampleGame):
    def _execute_action(self, action):
        before = self._current_level
        result = super()._execute_action(action)
        if self._current_level > before and result.raw.state != arcengine.GameState.WIN:
            result.raw.frame.insert(0, np.full((64, 64), 7, dtype=np.int8))
        return result


class TerminalFrames(unittest.TestCase):
    def test_post_action_terminal_is_in_next_real_request(self):
        with patch.dict(os.environ, {'TYCHO_TERMINAL_FRAMES': '1'}):
            directory, run, receipts, requests, _ = play(
                [[{'action': 'UP'}], [{'action': 'UP'}]],
                CompletionGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]] * 2,
                               label='terminal-fixture'), 2, 'lean_prune')
        with directory:
            self.assertEqual(run.state, 'won', run.solver_note)
            root = next((Path(directory.name) / 'hybrid-games').iterdir())
            terminal = json.loads(next(root.rglob('level_0/terminal.json')).read_text())
            self.assertEqual(terminal['terminal_grid'], [[7] * 64] * 64)
            texts = [json.dumps(request['messages']) for request in requests]
            self.assertNotIn('OBSERVED WINNING TRANSITIONS', texts[0])
            self.assertIn('OBSERVED WINNING TRANSITIONS', texts[1])
            self.assertIn('level_0/turn_000.txt', texts[1])
            self.assertIn('level_0/terminal.json (AFTER winning action)', texts[1])
            self.assertIn(grid_text(terminal['terminal_grid']).splitlines()[0], texts[1])
            before = next(root.rglob('level_0/turn_000.txt')).read_text()
            after = next(root.rglob('level_1/turn_000.txt')).read_text()
            self.assertEqual(_parse_grid_text(before), [[0] * 64] * 64)
            self.assertEqual(_parse_grid_text(after), [[4] * 64] * 64)

    def test_only_completed_levels_and_latest_two_grids_are_presented(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = GameWorkspace('terminal', root=directory, render=False)
            for level in range(4):
                workspace.record_terminal(level, [[level]], 'UP')
            turn = GameTurn(grid=[[9]], available_actions=['UP'], level=3,
                            turn_in_level=0, levels_completed=3)
            with patch.dict(os.environ, {'TYCHO_TERMINAL_FRAMES': '1'}):
                prompt = context(workspace, turn)
            self.assertIn('Completed level 1:', prompt)
            self.assertIn('Completed level 2:', prompt)
            self.assertNotIn('Completed level 0:', prompt)
            self.assertNotIn('Completed level 3:', prompt)
            with patch.dict(os.environ, {'TYCHO_TERMINAL_FRAMES': '0'}):
                self.assertEqual(context(workspace, turn), '')


if __name__ == '__main__':
    unittest.main()
