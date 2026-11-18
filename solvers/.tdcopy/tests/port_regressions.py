"""dfranzen affordance port: GAME_OVER legality, event guidance, interior no-op guard.

Writes nothing under evidence/; every artifact lives in a temporary directory.
"""
from __future__ import annotations
import asyncio
import json
import os
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'runtime'))
for _key in list(os.environ):
    if _key.startswith(('ARC_LLM_', 'OPENAI_', 'OPENROUTER_', 'ANTHROPIC_', 'NOOA_LLM_')):
        os.environ.pop(_key)
os.environ.update(LITELLM_LOCAL_MODEL_COST_MAP='True', TYCHO_SANDBOX_RUNTIME='host',
    MULTIMODAL_CONTEXT='current_grid', MULTIMODAL_UPSCALE='10',
    ARC3_MAX_ACTIVE_STREAMS='0', ARC3_PRIORITY_TAIL_FADE='0')

import arcengine
from taaf.benchmark import Benchmark
from taaf.game_examples import ExampleGame
from hybrid.solver import HybridSolver
from hybrid import policy


def script(*steps):
    """A list is a take_actions batch; a (tool, arguments) tuple is any other call."""
    return json.dumps([{'tool':step[0],'arguments':step[1]} if isinstance(step, tuple) else
                       {'tool':'take_actions','arguments':{'actions':step,'rationale':'port fixture'}}
                       for step in steps])


@dataclass
class DeathGame(ExampleGame):
    """Any wrong action is fatal. Like the real
    GameAPI (verified on ls20), any non-RESET action after GAME_OVER raises."""
    _dead: bool = False

    def _execute_action(self, action):
        if self._dead and action.id != arcengine.GameAction.RESET:
            raise RuntimeError('arcengine returned an empty frame (non-RESET after GAME_OVER)')
        if action.id == arcengine.GameAction.RESET:
            self._dead = False
            return super()._execute_action(action)
        target = self.target_sequence_per_level[self._current_level]
        if action.id != target[self._progress]:
            self._dead = True
            state = self._build_state(state=arcengine.GameState.NOT_FINISHED)
            state.raw.state = arcengine.GameState.GAME_OVER
            return state
        return super()._execute_action(action)

    def _build_state(self, state):
        built = super()._build_state(state)
        built.raw.available_actions = [1, 2]  # DOWN is offered, and fatal
        return built


def play(steps, game, max_actions, treatment='control'):
    os.environ.update(TYCHO_MODE='no_world_model', TYCHO_NOOA_TREATMENT=treatment,
                      TYCHO_NOOA_SCRIPTED_TOOL_CALLS=script(*steps))
    directory = tempfile.TemporaryDirectory()
    root = Path(directory.name)
    bm = Benchmark(label='port-fixture', games=[game], job_dir=root,
        solver=HybridSolver(worker_python=sys.executable, runtime_root=str(HERE/'runtime'),
            max_runtime_s_per_game=90, max_actions_per_game=max_actions, analyzer_timeout=60))
    asyncio.run(bm.run())
    game_dir = next((root/'hybrid-games').iterdir())
    receipts = [json.loads(line) for line in (game_dir/'host-actions.jsonl').read_text().splitlines()]
    requests = [json.loads(p.read_text())['request_body']
                for p in sorted((game_dir/'decision-checkpoints/calls').glob('*/request.json'))]
    tools = [json.loads(line) for line in (game_dir/'tool-trace.jsonl').read_text().splitlines()]
    return directory, bm.game_runs[0], receipts, requests, tools


def text(request):
    return json.dumps(request['messages'])


class GameOverPort(unittest.TestCase):
    def test_non_reset_after_game_over_is_refused_not_crashed(self):
        game = DeathGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]*2], label='death')
        directory, run, receipts, requests, tools = play(
            [[{'action':'DOWN'}], [{'action':'UP'}], [{'action':'RESET'}],
             [{'action':'UP'}], [{'action':'UP'}],
             ('write_file', {'path':'notes/level_0_insights.md','content':'UP UP; DOWN is fatal.'})],
            game, max_actions=4)
        with directory:
            self.assertEqual(run.state, 'won', run.solver_note)
            self.assertEqual([r['decision']['action'] for r in receipts], ['DOWN', 'RESET', 'UP', 'UP'])
            self.assertEqual(receipts[0]['after']['state'], 'GAME_OVER')
            self.assertEqual(receipts[0]['after']['available_actions'], ['RESET'])
            refused = [t for t in tools if "Input should be 'RESET'" in json.dumps(t)]
            self.assertTrue(refused, 'the illegal UP must surface to the Actor as a tool error')
            after_death = [r for r in requests if 'GAME OVER' in text(r)]
            self.assertTrue(after_death, 'game-over diagnosis must reach the Actor')
            self.assertIn('BAR RULE', text(after_death[0]))

    def test_system_prompt_states_the_task_and_flag_semantics(self):
        game = ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]*2], label='rules')
        directory, run, receipts, requests, tools = play([[{'action':'UP'}], [{'action':'UP'}]], game, 2)
        with directory:
            system = json.dumps(requests[0]['messages'][0])
            for phrase in ('as few actions as possible', 'GAME_OVER', 'only legal action is RESET',
                           'Levels usually build on mechanics'):
                self.assertIn(phrase, system)


class TenConcurrentGames(unittest.TestCase):
    def test_ten_workers_play_concurrently_to_completion(self):
        os.environ.update(TYCHO_MODE='orchestrator', TYCHO_NOOA_TREATMENT='control',
                          TYCHO_NOOA_SCRIPTED_TOOL_CALLS=script([{'action':'UP'}], [{'action':'UP'}],
                              ('write_file', {'path':'notes/level_0_insights.md','content':'UP UP'})))
        with tempfile.TemporaryDirectory() as directory:
            games = [ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]*2], label=f'g{i}')
                     for i in range(10)]
            bm = Benchmark(label='ten', games=games, job_dir=Path(directory),
                solver=HybridSolver(worker_python=sys.executable, runtime_root=str(HERE/'runtime'),
                    concurrency=10, max_runtime_s_per_game=120, max_actions_per_game=2, analyzer_timeout=90))
            asyncio.run(bm.run())
            self.assertEqual([r.state for r in bm.game_runs], ['won']*10,
                             [r.solver_note for r in bm.game_runs])
            starts = sorted(json.loads(p.read_text()).get('started_epoch', 0) for p in
                            Path(directory, 'hybrid-games').glob('*/worker-metadata.json'))
            self.assertEqual(len(starts), 10)


class LeanArm(unittest.TestCase):
    def test_no_world_model_no_scribe_runs_actor_only(self):
        game = ExampleGame(target_sequence_per_level=[[arcengine.GameAction.ACTION1]*2], label='lean')
        os.environ['TYCHO_NOOA_TREATMENT'] = 'no_scribe'
        try:
            os.environ['TYCHO_MODE'] = 'no_world_model'
            directory, run, receipts, requests, tools = play(
                [[{'action':'UP'}], [{'action':'UP'}]], game, 2, treatment='no_scribe')
        finally:
            os.environ['TYCHO_NOOA_TREATMENT'] = 'control'
        with directory:
            self.assertEqual(run.state, 'won', run.solver_note)
            game_dir = next(Path(directory.name, 'hybrid-games').iterdir())
            compute = json.loads((game_dir/'compute.json').read_text())
            self.assertEqual(set(compute['roles']), {'actor'})
            offered = {t['function']['name'] if 'function' in t else t['name']
                       for r in requests for t in r.get('tools', [])}
            self.assertNotIn('invoke_builder', offered)
            self.assertIn('as few actions as possible', json.dumps(requests[0]['messages'][0]))


class EventGuidance(unittest.TestCase):
    @staticmethod
    def payload(grid, *, levels=0, state='NOT_FINISHED'):
        from tycho_nooa.protocol import project_arc_observation
        return project_arc_observation(frames=[grid], available_action_ids=[1],
            state=state, levels_completed=levels, full_reset=False, previous_levels_completed=levels)

    def setUp(self):
        self.base = [[0]*16 for _ in range(16)]
        policy.BATCH_STOP = None

    def notes(self, before, after):
        policy.PREVIOUS, policy.OBSERVATION = before, after
        return policy.observation_notes()

    def test_level_up_brings_transfer_guidance(self):
        before = self.payload(self.base)
        after = self.payload([[1]*16 for _ in range(16)], levels=1)
        self.assertIn('completed the previous level', self.notes(before, after))

    def test_border_only_change_is_named(self):
        bar = [row[:] for row in self.base]
        bar[15][5] = 3
        self.assertIn('only the 4-cell border', self.notes(self.payload(self.base), self.payload(bar)))
        moved = [row[:] for row in self.base]
        moved[8][8] = 3
        self.assertNotIn('border', self.notes(self.payload(self.base), self.payload(moved)))

    def test_batch_stop_reason_is_reported_once(self):
        policy.BATCH_STOP = (1, 'previous_action_no_change')
        first = self.notes(self.payload(self.base), self.payload(self.base))
        self.assertIn('previous_action_no_change', first)
        self.assertNotIn('previous_action_no_change', self.notes(self.payload(self.base), self.payload(self.base)))

    def test_interior_guard(self):
        bar = [row[:] for row in self.base]
        bar[0][3] = 3
        self.assertFalse(policy.interior_changed(self.base, bar))
        inner = [row[:] for row in self.base]
        inner[4][4] = 3
        self.assertTrue(policy.interior_changed(self.base, inner))
        self.assertTrue(policy.interior_changed(self.base, [[0]*8 for _ in range(8)]))


class BatchGuardSeam(unittest.TestCase):
    def test_border_only_successor_stops_the_queue(self):
        from tycho_nooa.action_batch import ActionBatchDecision, PendingActionBatch
        from tycho_nooa.models import GameTurn
        from tycho_nooa.protocol import ActionResultMessage, PROTOCOL_VERSION, project_arc_observation
        base = [[0]*16 for _ in range(16)]
        bar = [row[:] for row in base]
        bar[15][2] = 7
        def obs(grid):
            return project_arc_observation(frames=[grid], available_action_ids=[1], state='NOT_FINISHED',
                levels_completed=0, full_reset=False, previous_levels_completed=0)
        batch = ActionBatchDecision.model_validate({'actions':[{'action':'UP'},{'action':'UP'}],
                                                    'rationale':'guard'})
        pending = PendingActionBatch(batch=batch, anchor_level=0, anchor_turn_in_level=0, anchor_sequence=0)
        first = batch.actions[0].decision()
        result = ActionResultMessage(protocol_version=PROTOCOL_VERSION, type='action_result', sequence=0,
            level=0, turn_in_level=0, before=obs(base), decision=first, after=obs(bar))
        turn = GameTurn(grid=bar, available_actions=['RESET','UP'], sequence=1, level=0, turn_in_level=1,
            levels_completed=0, state='NOT_FINISHED', previous_action='UP')
        decision, reason = pending.release(turn, result)
        self.assertIsNone(decision)
        self.assertEqual(reason, 'previous_action_no_change')


if __name__ == '__main__':
    unittest.main()
