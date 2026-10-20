"""F1x + REAP448 build: the merge is exactly F1x with REAP's serving cell at F1x's context, and nothing else.

python3 -B -m unittest solvers/m2_f1x_reap/tests/test_build.py
"""
import ast, difflib, json, re, subprocess, sys, unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import build  # noqa: E402

RUN = HERE / 'experiment/screen-1/merge/kernel'


def code(nb):
    return [''.join(c['source']) for c in nb['cells'] if c['cell_type'] == 'code']


def cell(nb, first_line):
    hits = [s for s in code(nb) if s.startswith(first_line)]
    assert len(hits) == 1, first_line
    return hits[0]


def config_value(text, key):
    m = re.search(rf"^\s*'?{key}'?\s*[:=]\s*(.+?),", text, re.M)
    return eval(m.group(1), {})  # literal arithmetic such as (116+16)*1024


class Build(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.f1x = json.loads(build.F1X_NB.read_text())
        cls.reap = json.loads(build.REAP_NB.read_text())
        cls.merge = build.merge_notebook()
        cls.screen = json.loads((RUN / 'm2-f1x-reap14-screen1.ipynb').read_text())

    def test_parents_are_the_drawn_notebooks(self):
        self.assertEqual(build.sha(build.F1X_NB.read_bytes()), build.F1X_SHA)
        self.assertEqual(build.sha(build.REAP_NB.read_bytes()), build.REAP_SHA)

    def test_written_kernel_is_the_build(self):
        self.assertEqual(code(self.screen), code(self.merge))

    def test_merge_minus_serving_change_is_f1x(self):
        launcher = build.LAUNCHER
        merge = [s for s in code(self.merge) if not s.startswith(launcher)]
        f1x = [s for s in code(self.f1x) if not s.startswith(launcher)]
        for a, b in build.CONFIG:
            merge = [s.replace(b, a) for s in merge]
        self.assertEqual(merge, f1x)
        self.assertEqual(len(code(self.merge)), len(code(self.f1x)))

    def test_merge_launcher_is_reap_launcher_at_f1x_context(self):
        merge = cell(self.merge, build.LAUNCHER)
        for a, b in build.SERVING:
            merge = merge.replace(b, a)
        self.assertEqual(merge, cell(self.reap, build.LAUNCHER))

    def test_f1x_launcher_lines_survive_except_stream_sizing(self):
        f1x = cell(self.f1x, build.LAUNCHER).splitlines()
        merge = cell(self.merge, build.LAUNCHER).splitlines()
        dropped = [l[2:] for l in difflib.ndiff(f1x, merge) if l.startswith('- ')]
        allowed = ('    MAXREQ=', '    CUDAGRAPH_MAXBS=', '    MAMBA_CACHE=', 'graph_bs = sorted(', '    CTX=')
        self.assertTrue(dropped and all(l.startswith(allowed) for l in dropped), dropped)
        self.assertIn('    CTX=(116+16+8)*1024,', cell(self.merge, build.LAUNCHER))

    def test_merge_minus_harness_is_reap(self):
        """Removing F1x's harness cells and context lines recovers REAP's code, except the screen soft end."""
        merge = [s for s in code(self.merge) if not s.startswith(('# M2 B1: install', '# M2 F1:'))]
        reap = code(self.reap)
        self.assertEqual(len(merge), len(reap))
        # REAP deleted its hook-cell override line and kept the newline before it; nothing else differs there
        differs = [i for i, (a, b) in enumerate(zip(merge, reap)) if a.rstrip() != b.rstrip()]
        names = [next(l for l in reap[i].splitlines() if l.strip())[:40] for i in differs]
        self.assertEqual(len(differs), 3, names)  # config, launcher, benchmark
        self.assertIn(build.SCREEN_END, merge[differs[2]])
        self.assertIn(build.REAP_SMOKE_END, reap[differs[2]])

    def test_stream_sizing_is_consistent(self):
        cfg = cell(self.merge, 'import json')
        launcher = cell(self.merge, build.LAUNCHER)
        streams = config_value(cfg, 'ARC3_MAX_ACTIVE_STREAMS')
        self.assertEqual(streams, build.STREAMS)
        self.assertEqual(config_value(launcher, 'MAXREQ'), streams)
        self.assertEqual(config_value(launcher, 'CUDAGRAPH_MAXBS'), streams)
        self.assertEqual(config_value(launcher, 'MAMBA_CACHE'), 6 * streams)
        window, out = config_value(cfg, 'LOCAL_ANALYZER_CONTEXT_WINDOW'), config_value(cfg, 'LOCAL_ANALYZER_MAX_OUTPUT')
        self.assertEqual((window - out, out), (116 * 1024, 16 * 1024))
        self.assertEqual(config_value(launcher, 'CTX'), window + 8 * 1024)
        self.assertIn("PRUNE_MODE = 'pruned'", launcher)

    def test_ledger_check_passes_without_exemptions(self):
        nb = RUN / 'm2-f1x-reap14-screen1.ipynb'
        r = subprocess.run([sys.executable, '-B', str(ROOT / 'incumbent/check.py'), str(nb)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertNotIn('ABSENT', r.stdout)

    def test_metadata_slug_matches_title_and_parent_sources(self):
        meta = json.loads((RUN / 'kernel-metadata.json').read_text())
        self.assertEqual(meta['id'], 'pranavagrawai/' + meta['title'].lower().replace(' ', '-'))
        self.assertEqual(meta['code_file'], meta['id'].split('/')[1] + '.ipynb')
        for parent in (build.F1X_NB, build.REAP_NB):
            pmeta = json.loads((parent.parent / 'kernel-metadata.json').read_text())
            for k in ('dataset_sources', 'model_sources', 'competition_sources', 'docker_image', 'machine_shape', 'enable_internet'):
                self.assertEqual(meta[k], pmeta[k], k)
        self.assertTrue(meta['is_private'])


class CompetitionBranches(unittest.TestCase):
    """Exercise the real cell code for the branches the screen can't reach."""

    @classmethod
    def setUpClass(cls):
        cls.merge = build.merge_notebook()

    def soft_end(self, competition):
        import datetime as dt
        tree = ast.parse(cell(self.merge, "print('Starting benchmark...')"))
        i = next(i for i, n in enumerate(tree.body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'soft_end' for t in n.targets))
        ns = dict(TRUE_SUBMISSION=competition, datetime=dt.datetime, timedelta=dt.timedelta,
                  target=SimpleNamespace(max_runtime_s=32400))
        exec(compile(ast.Module(body=tree.body[i:i + 2], type_ignores=[]), '<soft-end>', 'exec'), ns)
        return ns['soft_end']

    def test_submission_has_no_soft_end_and_screen_stops_at_60_min(self):
        import datetime as dt
        self.assertIsNone(self.soft_end(True))
        left = (self.soft_end(False) - dt.datetime.now()).total_seconds()
        self.assertAlmostEqual(left, 3600, delta=5)

    def ready_wait(self, competition, ready_after_polls):
        """Run the launcher's post-deadline READY loop with a live server that is slow to answer /health."""
        launcher = cell(self.merge, build.LAUNCHER)
        start = launcher.index('LATE_READY_LIMIT = ')
        end = launcher.index('print(f"SERVER READY at', start)
        clock = {'t': 0.0, 'polls': 0}

        class Health:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def urlopen(url, timeout):
            clock['polls'] += 1
            if clock['polls'] < ready_after_polls:
                raise OSError('not ready')
            return Health()

        def sleep(s): clock['t'] += 600  # each poll costs ten minutes: a very slow disk
        ns = dict(TRUE_SUBMISSION=competition, NOTEBOOK_START_TIME=0.0, url='http://x/health',
                  urllib=SimpleNamespace(request=SimpleNamespace(urlopen=urlopen)),
                  time=SimpleNamespace(time=lambda: clock['t'], sleep=sleep),
                  proc=SimpleNamespace(poll=lambda: None, returncode=None), show_log_tail=lambda: None)
        exec(compile(launcher[start:end], '<ready-wait>', 'exec'), ns)
        return clock

    def test_submission_keeps_waiting_for_a_live_slow_server(self):
        clock = self.ready_wait(True, ready_after_polls=6)
        self.assertGreater(clock['t'], 1300)  # past the screen limit, still waiting, then READY

    def test_screen_stops_a_slow_server_at_the_limit(self):
        with self.assertRaisesRegex(RuntimeError, 'LATE_READY_LIMIT'):
            self.ready_wait(False, ready_after_polls=6)


if __name__ == '__main__':
    unittest.main()
