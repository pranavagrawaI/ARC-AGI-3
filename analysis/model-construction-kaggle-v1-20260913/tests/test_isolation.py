import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from constructor import ConstructionLimits, run_construction
from real_trace import load_case
from sandbox_eval import evaluate_candidate, execute_inspection, sealed_rollout, replay_visible


def response(name, arguments):
    return {'usage': {'prompt_tokens': 10, 'completion_tokens': 10}, 'choices': [{'message': {'role': 'assistant', 'tool_calls': [
        {'id': 'attempt', 'type': 'function', 'function': {'name': name, 'arguments': arguments}}
    ]}}]}


class IsolationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.case = load_case()
        cls.witness = (HERE / 'tests/witness_candidate.py').read_text()

    def test_candidate_cannot_read_evidence_or_verifier(self):
        # Executes at every ABI call, including all suffix predictions.
        guard = '''
import pathlib, json, sys
assert not pathlib.Path('/work/case.json').exists()
assert not pathlib.Path('/evidence/case.json').exists()
assert not pathlib.Path('/home/pranav').exists()
assert set(p.name for p in pathlib.Path('/work').iterdir()) == {'runner.py', 'candidate.py', 'args.json'}
request = json.loads(pathlib.Path('/work/args.json').read_text())
assert set(request) == {'method', 'args'}
assert not any(name in vars(sys.modules['__main__']) for name in ('observed', 'compatible', 'sealed_rollout', 'fixture'))
if request['method'] == 'predict':
    assert len(request['args']) == 2
    assert isinstance(request['args'][1], str)
'''
        result = evaluate_candidate(guard + self.witness, self.case, include_suffix=True)
        self.assertEqual(result['suffix']['status'], 'passed')

    def test_forged_assessment_cannot_pass(self):
        source = '''
import sys
sys.modules['__main__'].compatible = lambda *a: True
sys.modules['__main__'].observed = lambda *a: {'block': [0,0], 'bar_left': 0}
def initialize(observation): return {}
def predict(state, action):
    return [{'state': {}, 'commitments': {'block': [0,0], 'bar_left': 0},
             'visible': {'status': 'passed'}, 'suffix': {'status': 'passed'}}]
def assimilate(*args): return {}
'''
        result = evaluate_candidate(source, self.case, include_suffix=True)
        self.assertEqual(result['visible']['status'], 'contradicted')
        self.assertEqual(result['suffix']['status'], 'not_run')

    def test_printed_forged_envelope_cannot_pass(self):
        source = "import os\nprint('{\"visible\": {\"status\": \"passed\"}, \"suffix\": {\"status\": \"passed\"}}', flush=True)\nos._exit(0)"
        with self.assertRaises(ValueError):
            evaluate_candidate(source, self.case, include_suffix=True)

    def test_output_flood_is_killed_during_execution(self):
        with self.assertRaisesRegex(RuntimeError, 'output limit'):
            execute_inspection("import os\nwhile True: os.write(1, b'x'*8192)", self.case)
        with self.assertRaisesRegex(RuntimeError, 'output limit'):
            evaluate_candidate("import os\nwhile True: os.write(2, b'x'*8192)", self.case, include_suffix=False)

    def test_failed_attempts_consume_budgets_and_are_durable(self):
        for name, counter in [('execute_python', 'inspections'), ('submit_candidate', 'submissions')]:
            events = []
            result = run_construction(self.case, transport=lambda body: response(name, '{bad json'),
                limits=ConstructionLimits(reserve_requests=0, reserve_completion_tokens=0, max_model_requests=2, max_inspections=1, max_submissions=1),
                event_sink=events.append)
            self.assertEqual(getattr(result, counter), 1)
            self.assertEqual(len(events), 2)
            self.assertEqual(events[0]['call']['function']['arguments'], '{bad json')
            self.assertIn('budget exhausted', events[1]['rendered_result_message']['content'])

    def test_request_snapshots_do_not_acquire_future_tool_results(self):
        seen = []
        def transport(body):
            seen.append(body)
            return response('execute_python', json.dumps({'code': 'print(123)'}))
        result = run_construction(self.case, transport=transport,
            limits=ConstructionLimits(reserve_requests=0, reserve_completion_tokens=0, max_model_requests=2, max_inspections=2))
        self.assertEqual(len(seen[0]['messages']), 3)
        self.assertEqual(len(seen[1]['messages']), 5)
        self.assertEqual(result.transcript[0]['request'], seen[0])
        self.assertEqual(json.loads(result.tool_events[0]['rendered_result_message']['content'])['stdout'], '123\n')

    def test_execution_failure_consumes_attempt(self):
        with patch('constructor.execute_inspection', side_effect=TimeoutError('bounded')) as call:
            result = run_construction(self.case,
                transport=lambda body: response('execute_python', '{"code":"while True: pass"}'),
                limits=ConstructionLimits(reserve_requests=0, reserve_completion_tokens=0, max_model_requests=2, max_inspections=1))
            self.assertEqual(result.inspections, 1)
            self.assertEqual(call.call_count, 1)

    def test_prefix_predicts_before_assimilation_and_no_future_args(self):
        case = self.case.evaluator_payload(include_suffix=False)
        calls = []
        class Candidate:
            def initialize(self, observation):
                self.initial = observation
                return 0
            def predict(self, state, action):
                calls.append(('predict', state))
                return [{'state': state+1, 'commitments': {'block': [0,0], 'bar_left': 0}}]
            def assimilate(self, *args):
                calls.append(('assimilate', args))
                raise AssertionError('contradicted forecast must not assimilate')
        _, verdict, _, _ = replay_visible(Candidate(), case)
        self.assertEqual(verdict['status'], 'contradicted')
        self.assertEqual(calls, [('predict', 0)])

    def test_joint_branches_and_temporal_ancestry(self):
        def state(block, bar):
            grid = [[0]*64 for _ in range(64)]
            grid[block[0]][block[1]] = 12
            grid[61][bar] = 11
            return {'grid': grid}
        fixture = {'suffix': [
            {'step': 6, 'action': 'UP', 'after': state([1,1], 2)},
            {'step': 7, 'action': 'UP', 'after': state([3,3], 4)}],
            'branch_cap': 32, 'required_targets': ['block', 'bar_left']}
        start = [{'state': 'root', 'ancestry': [0], 'history': []}]
        calls = []
        class Candidate:
            def predict(self, state, action):
                calls.append(state)
                if state == 'root':
                    return [{'state': 'live', 'commitments': {'block': [1,1], 'bar_left': 2}},
                            {'state': 'dead', 'commitments': {'block': [9,9], 'bar_left': 2}}]
                return [{'state': 'end', 'commitments': {'block': [3,3], 'bar_left': 4 if state == 'dead' else 9}}]
        result, _ = sealed_rollout(Candidate(), start, fixture)
        self.assertEqual(calls, ['root', 'live', 'dead'])
        self.assertEqual(result['compatible_depth'], 1)
        self.assertEqual(result['first_contradiction_step'], 7)
        class Joint:
            def predict(self, state, action):
                return [{'state': '', 'commitments': {'block': [1,1], 'bar_left': 9}},
                        {'state': '', 'commitments': {'block': [9,9], 'bar_left': 2}}]
        result, _ = sealed_rollout(Joint(), start, fixture)
        self.assertEqual(result['compatible_depth'], 0)


if __name__ == '__main__':
    unittest.main()
