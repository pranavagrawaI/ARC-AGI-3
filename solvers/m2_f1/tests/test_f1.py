"""F1 behaviour through real sandbox subprocesses. The source tree is produced by executing the install cell
taken from the built F1 notebook against a copy of the B1 source, exactly as the notebook does on Kaggle."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest

HERE = Path(__file__).resolve().parents[1]
B1_SRC = HERE.parent / 'm2_b1' / 'src'
NOTEBOOK = HERE / 'kernel' / 'm2-f1.ipynb'
REL = 'inference/agent/python_tool_sandbox.py'
STATE = {'current_frame': {'grid': [[0, 1]], 'ascii': 'Ww', 'shape': [1, 2], 'step': 0, 'level': 1},
         'history': [], 'valid_actions': ['UP']}


def install_cell():
    cells = json.loads(NOTEBOOK.read_text())['cells']
    found = [''.join(c['source']) for c in cells if ''.join(c['source']).startswith('# M2 F1:')]
    assert len(found) == 1
    return found[0]


def run_cell(bundle):
    exec(compile(install_cell(), 'f1-install-cell', 'exec'), {'BUNDLE_DIR': bundle})


TMP = Path(tempfile.mkdtemp(prefix='m2-f1-test-'))
TREE = TMP / 'src' / 'ARC3-Inference'
shutil.copytree(B1_SRC, TREE)
run_cell(TMP)
sys.path.insert(0, str(TREE))
package = ModuleType('inference.agent')
package.__path__ = [str(TREE / 'inference/agent')]
sys.modules['inference.agent'] = package
from inference.agent import python_tool_sandbox as sandbox  # noqa: E402


def tearDownModule():
    shutil.rmtree(TMP, ignore_errors=True)


class InstallCell(unittest.TestCase):
    def test_refuses_a_sandbox_that_is_not_b1(self):
        with self.assertRaisesRegex(RuntimeError, 'not the B1 source'):
            run_cell(TMP)  # TREE already carries the F1 sandbox

    def test_patched_module_is_the_one_under_test(self):
        self.assertIn('_f1_module', (TREE / REL).read_text())
        self.assertEqual(Path(sandbox.__file__).resolve(), (TREE / REL).resolve())


class Preload(unittest.TestCase):
    def run_code(self, code, **kwargs):
        return sandbox.run_sandboxed_python(code=code, timeout_seconds=5, initial_state=STATE,
                                            action_handler=lambda actions, **k: {}, **kwargs)

    def test_forgotten_imports_run(self):
        result = self.run_code('print(collections.Counter("aab")["a"], math.sqrt(9), json.dumps([1]), '
                               'heapq.nsmallest(1, [3, 2]), list(itertools.islice(itertools.count(), 2)))')
        self.assertFalse(result.get('error'))
        self.assertEqual(result['stdout'].strip(), '2 3.0 [1] [2] [0, 1]')

    def test_function_using_a_preloaded_module_is_retained_and_restored(self):
        first = self.run_code('def top(xs): return collections.Counter(xs).most_common(1)[0][0]',
                              kept_functions=[], retain_imports=True)
        kept = [f['source'] for f in first['keepable_functions']]
        self.assertEqual([f['name'] for f in first['keepable_functions']], ['top'])
        self.assertEqual(first['retention_rejected'], [])
        second = self.run_code('print(top("abb"))', kept_functions=kept, retain_imports=True)
        self.assertFalse(second.get('error'))
        self.assertEqual(second['stdout'].strip(), 'b')

    def test_explicit_imports_are_still_retained_without_conflict(self):
        result = self.run_code('import collections\nfrom collections import deque\n'
                               'def last(xs): return collections.deque(xs).pop() + deque(xs).pop()',
                               kept_functions=[], retain_imports=True)
        self.assertEqual(result['retention_rejected'], [])
        kept = [f['source'] for f in result['keepable_functions']]
        again = self.run_code('print(last([1, 2]))', kept_functions=kept, retain_imports=True)
        self.assertEqual(again['stdout'].strip(), '4')

    def test_snippet_can_rebind_a_preloaded_name(self):
        result = self.run_code('json = 3\nprint(json + 1)')
        self.assertFalse(result.get('error'))
        self.assertEqual(result['stdout'].strip(), '4')

    def test_allowlist_is_unchanged(self):
        self.assertIn('not allowed', str(self.run_code('import os').get('error')))
        self.assertIn('NameError', str(self.run_code('os.getcwd()').get('error')))


if __name__ == '__main__':
    unittest.main()
