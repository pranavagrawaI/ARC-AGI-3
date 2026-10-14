"""Independent expected transitions and failure-stage contracts."""
import copy
import importlib.util
from pathlib import Path
import unittest

PATH = Path(__file__).resolve().parents[1] / 'overlay/inference/utils/prospective_probe.py'
spec = importlib.util.spec_from_file_location('probe_under_test', PATH)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
probe = module.prospective_probe


class ProbeTests(unittest.TestCase):
    def run_probe(self, predicted=1, actual=1, receipt=None, **overrides):
        rows, order = [], []
        state = {'n': 0}
        def observe():
            order.append('observe')
            return state
        def predict(before, action):
            order.append('predict')
            before['n'] = 999  # Cannot corrupt the saved before-state.
            return {'n': predicted}
        def execute(action):
            self.assertEqual(rows[-1]['status'], 'PREDICTED')
            self.assertEqual(rows[-1]['predicted'], {'n': predicted})
            order.append('execute')
            state['n'] = actual
            return receipt if receipt is not None else {'executed_count': 1}
        def record(row):
            order.append('record:' + row['status'])
            rows.append(copy.deepcopy(row))
        arguments = dict(observe=observe, predict=predict, state_key=lambda s:s['n'],
                         execute=execute, record=record)
        arguments.update(overrides)
        result = probe({'action': 'UP'}, **arguments)
        return result, rows, order

    def test_prediction_is_recorded_before_one_action_and_state_is_frozen(self):
        result, rows, order = self.run_probe()
        self.assertEqual(result['status'], 'MATCH')
        self.assertEqual(rows[0]['before'], {'n': 0})
        self.assertEqual(order, ['observe','predict','record:PREDICTED','execute','record:ACTED','observe','record:MATCH'])
        self.assertIsNone(rows[0]['actual'])
        self.assertEqual(rows[0]['predicted'], {'n': 1})

    def test_mismatch_records_both_states(self):
        result, _, _ = self.run_probe(predicted=2, actual=1)
        self.assertEqual((result['status'], result['matched']), ('MISMATCH', False))
        self.assertEqual(result['actual'], {'n': 1})

    def test_refused_unknown_and_terminal_do_not_claim_agreement(self):
        for receipt, expected in [({'executed_count':0},'REFUSED'), ({},'ACTION_UNCONFIRMED'),
                                  ({'executed_count':2},'ACTION_UNCONFIRMED'),
                                  ({'executed_count':True},'ACTION_UNCONFIRMED'),
                                  ({'executed_count':1,'level_completed':True},'TERMINAL')]:
            result, _, order = self.run_probe(receipt=receipt)
            self.assertEqual((result['status'],result['matched']), (expected,None))
            self.assertEqual(order.count('observe'), 1)

    def test_native_single_action_receipt_and_malformed_counts(self):
        result,_,_=self.run_probe(receipt={'executed':True,'executed_actions':['MOUSE(row=57, col=36)']})
        self.assertEqual(result['status'],'MATCH')
        for receipt in [{'executed':True},{'executed':True,'executed_actions':[]},
                        {'executed':True,'executed_actions':['UP'],'executed_count':True},
                        {'executed_count':-1},{'executed_count':0,'executed':True},
                        {'executed_count':1,'executed':False},{'executed_count':1,'executed_actions':[]}]:
            result,_,_=self.run_probe(receipt=receipt)
            self.assertEqual(result['status'],'ACTION_UNCONFIRMED')
        result,_,_=self.run_probe(receipt={'executed':False})
        self.assertEqual(result['status'],'REFUSED')

    def test_prediction_or_persistence_failure_prevents_execution(self):
        for callback in ['predict','record','state_key']:
            def fail(*args):
                raise ValueError('qualification failure')
            with self.assertRaisesRegex(ValueError, 'qualification failure'):
                self.run_probe(**{callback:fail})

    def test_missing_prediction_and_invalid_key_prevent_execution(self):
        with self.assertRaisesRegex(ValueError, 'Prediction is required'):
            self.run_probe(predict=lambda *args:None)
        with self.assertRaises(TypeError):
            self.run_probe(state_key=lambda s:[])
        with self.assertRaises(TypeError):
            self.run_probe(observe=None)

    def test_observation_failure_retains_acted_stage(self):
        rows, calls = [], []
        def observe():
            calls.append('observe')
            if len(calls) > 1:
                raise RuntimeError('cannot observe')
            return 0
        with self.assertRaisesRegex(RuntimeError, 'cannot observe'):
            probe('UP',observe=observe,predict=lambda s,a:1,state_key=lambda s:s,
                  execute=lambda a:{'executed_count':1},record=lambda row:rows.append(row))
        self.assertEqual(rows[-1]['status'], 'ACTED')
        self.assertIsNone(rows[-1]['matched'])


if __name__ == '__main__':
    unittest.main()
