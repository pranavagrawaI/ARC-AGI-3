"""m2_guard: the build touches only its anchors, and each guard fires on its fault and stays silent otherwise.
Guard code is executed from the built notebook text, not re-typed here. End-to-end fault runs (dead server,
mid-run crash, startup crash) live in analysis/f1x-hidden-zero-diagnosis-20261021/faults/."""
import contextlib, io, json, os, shutil, stat, subprocess, sys, tempfile, types, unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import build  # noqa: E402


def code(nb): return [build.source(c) for c in nb['cells'] if c['cell_type'] == 'code']


def built(name):
    path, _ = build.PARENTS[name]
    parent = json.loads(path.read_bytes())
    return parent, build.guard(parent)


def end_snippet(nb):
    cell = next(c for c in code(nb) if build.RUN_ANCHOR in c)
    start = cell.index('# M2 GUARD end')
    lines = cell[start:].splitlines()
    end = next(i for i, l in enumerate(lines) if l.startswith('    raise RuntimeError(f"M2 GUARD:'))
    return '\n'.join(lines[:end + 2])


def run(state, note, levels=0, tokens=None, history=()):
    if tokens is None:
        tokens = int(note.split('=')[1]) if note.startswith('tokens=') else 0
    return types.SimpleNamespace(state=state, solver_note=note, levels_completed=levels,
                                 final_generated_tokens=tokens, history=list(history))


class BuildTouchesOnlyAnchors(unittest.TestCase):
    def test_both_parents(self):
        for name in build.PARENTS:
            parent, nb = built(name)
            old, new = code(parent), code(nb)
            self.assertEqual(len(new), len(old) + 1, name)
            sup = [i for i, c in enumerate(new) if c.startswith('# M2 GUARD supervise')]
            self.assertEqual(len(sup), 1)
            del new[sup[0]]
            changed = [i for i, (a, b) in enumerate(zip(old, new)) if a != b]
            self.assertEqual(len(changed), 2, name)  # server cell (gpu) and run cell (end)
            for i in changed:
                undone = new[i].replace(build.GPU, build.GPU_ANCHOR).replace(build.END, build.RUN_ANCHOR)
                self.assertEqual(undone, old[i], name)
            # supervisor sits after the server cell and before the pickle cell, i.e. before the solver import
            self.assertTrue(code(nb)[sup[0] + 1].startswith(build.SUPERVISOR_BEFORE))

    def test_repeated_prelaunch_builds_are_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'm2_guard'
            shutil.copytree(HERE, folder, ignore=shutil.ignore_patterns('evidence', '__pycache__'))
            with mock.patch.object(build, 'HERE', folder), contextlib.redirect_stdout(io.StringIO()):
                build.main(['f1x', 'control'])
                before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
                build.main(['f1x', 'control'])
                after = {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            self.assertIn('fault_screen_1', json.loads((folder / 'manifest.json').read_text()))

    def test_any_build_after_launch_attempt_refuses_before_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory) / 'm2_guard'
            shutil.copytree(HERE, folder, ignore=shutil.ignore_patterns('evidence', '__pycache__'))
            with mock.patch.object(build, 'HERE', folder), contextlib.redirect_stdout(io.StringIO()):
                build.main(['f1x', 'control'])
                (folder / 'experiment/fault-screen-1/launch-attempt.json').write_text('{"test": true}\n')
                before = {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
                for names in [['f1x'], ['f1x'], ['control'], []]:
                    with self.subTest(names=names):
                        with self.assertRaisesRegex(RuntimeError, 'launch attempt'):
                            build.main(names)
                        after = {p.relative_to(folder): p.read_bytes() for p in folder.rglob('*') if p.is_file()}
                        self.assertEqual(before, after)


class EndGuard(unittest.TestCase):
    snippet = end_snippet(built('f1x')[1])

    def check(self, runs):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            exec(compile(self.snippet, 'm2-guard-end', 'exec'),
                 {'bm': types.SimpleNamespace(game_runs=runs), 'json': json, '__file__': 'm2-guard-end'})
        return json.loads(output.getvalue().removeprefix('M2 GUARD end '))

    def test_dead_stack_raises(self):
        runs = [run('gave_up', 'analyzer failed 11 times consecutively') for _ in range(108)] + \
               [run('crashed', 'error: X'), run('gave_up', 'tokens=0')]
        with self.assertRaisesRegex(RuntimeError, '109/109 served games ended on analyzer failures'):
            self.check(runs)

    def test_dead_stack_with_unserved_games_raises(self):
        # CPU T1 shape: games that never reached a stream before their cap end as tokens=0, not as failures
        runs = [run('gave_up', 'analyzer failed 41 times consecutively') for _ in range(80)] + \
               [run('gave_up', 'tokens=0') for _ in range(30)]
        with self.assertRaisesRegex(RuntimeError, 'M2 GUARD'):
            self.check(runs)

    def test_genuine_zero_play_passes(self):
        self.check([run('gave_up', 'tokens=812345') for _ in range(110)])

    def test_recorded_f1x_zero_level_games_are_played_zeros(self):
        path = HERE / 'tests/fixtures/f1x-played-zero.json'
        benchmark = json.loads(path.read_text(), object_hook=lambda row: types.SimpleNamespace(**row))
        zeros = [game for game in benchmark.game_runs if game.levels_completed == 0]
        self.assertEqual(len(zeros), 3)
        self.assertEqual([game.final_generated_tokens for game in zeros], [0, 0, 0])
        result = self.check(zeros)
        self.assertEqual(result, {'games': 3, 'served': 3, 'failed': 0, 'tokens': 233500, 'levels': 0})

    def test_history_and_final_no_move_turn_tokens_are_both_counted_once(self):
        result = self.check([run('cancelled', '', tokens=3, history=[types.SimpleNamespace(generated_tokens=7)])])
        self.assertEqual(result['tokens'], 10)

    def test_synthetic_dead_endpoint_reset_is_not_model_activity(self):
        # Real GameRun/ActionRecord shape: RESET is recorded even when the model endpoint never serves a token.
        reset = types.SimpleNamespace(action={'id': 'RESET', 'data': {}}, generated_tokens=0)
        with self.assertRaisesRegex(RuntimeError, '0 tokens, no level'):
            self.check([run('gave_up', 'analyzer failed 41 times consecutively', history=[reset])])

    def test_any_level_passes(self):
        self.check([run('gave_up', 'analyzer failed 41 times consecutively') for _ in range(109)] +
                   [run('gave_up', 'tokens=9', levels=1)])

    def test_partial_failure_passes(self):
        self.check([run('gave_up', 'analyzer failed 41 times consecutively') for _ in range(98)] +
                   [run('gave_up', 'tokens=9') for _ in range(12)])


class GpuGuard(unittest.TestCase):
    snippet = build.GPU.split("'''], env=env)", 1)[1].split('# ---- validate target')[0]

    def check(self, name):
        d = tempfile.mkdtemp()
        fake = Path(d) / 'nvidia-smi'
        fake.write_text('#!/bin/sh\n' + ''.join(f'echo "{n}"\n' for n in name.split('|')))
        fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
        old = os.environ['PATH']
        os.environ['PATH'] = f'{d}:{old}'
        try:
            exec(self.snippet, {'subprocess': subprocess})
        finally:
            os.environ['PATH'] = old

    def test_t4_raises(self):
        with self.assertRaisesRegex(RuntimeError, 'Tesla T4'):
            self.check('Tesla T4')

    def test_two_t4_raises(self):  # serving trial-1, 2026-10-21, was placed on 2x T4
        with self.assertRaisesRegex(RuntimeError, 'Tesla T4'):
            self.check('Tesla T4|Tesla T4')

    def test_mixed_raises(self):
        with self.assertRaisesRegex(RuntimeError, 'Tesla T4'):
            self.check('NVIDIA RTX PRO 6000 Blackwell Server Edition|Tesla T4')

    def test_no_gpu_raises(self):
        with self.assertRaisesRegex(RuntimeError, 'M2 GUARD'):
            self.check('')

    def test_rtx_pro_passes(self):
        self.check('NVIDIA RTX PRO 6000 Blackwell Server Edition')


if __name__ == '__main__':
    unittest.main()


class ScreenFault(unittest.TestCase):
    def cell(self):
        nb = build.fault_screen(built('f1x')[1])
        return next(c for c in code(nb) if c.startswith('# M2 GUARD screen fault'))

    def run_cell(self, true_submission):
        import threading, time
        killed = threading.Event()
        proc = types.SimpleNamespace(pid=1, kill=killed.set)
        os.environ['M2_GUARD_FAULT_KILL_S'] = '0.05'
        try:
            exec(self.cell(), {'TRUE_SUBMISSION': true_submission, 'os': os, 'time': time, 'threading': threading,
                               '_guard': {'ready': True, 'proc': proc}, '_guard_event': lambda *a, **k: None})
        finally:
            del os.environ['M2_GUARD_FAULT_KILL_S']
        return killed.wait(1.0)

    def test_screen_kills_server_once_ready(self):
        self.assertTrue(self.run_cell(False))

    def test_submission_never_injects(self):
        self.assertFalse(self.run_cell(True))
