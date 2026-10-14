"""Real c14 gate/session reproduction; only game/analyzer IO is substituted.

The full tool_agent module is imported from Duck commit 7652836 plus the
frozen notebook patch. The session's actual play and queue-position methods
are AST-extracted to avoid importing unavailable arcengine on this CPU host.
"""
import ast
import contextlib
import hashlib
import io
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import types
import unittest

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[2]/'analysis/model-serving-20261011/wave-1/e2-c14/kernel/wave1-e2-c14-20261011.ipynb'
DUCK = Path(os.environ.get('REFERENCE_DUCK_REPO', '/home/pranav/projects/arc-agi-3-reference/duck-harness'))


def session_type(source_root):
    from inference.agent import tool_agent
    path=Path(source_root)/'ARC3-Inference/inference/framework/solver.py'
    tree=ast.parse(path.read_text())
    original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_HarnessGameSession')
    methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in {'play','_queue_position'}]
    helper=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_analyzer_reported_tokens')
    cls=ast.ClassDef(name='Session',bases=[],keywords=[],body=methods,decorator_list=[])
    ns={'Path':Path,'time':time,'Any':object,'_priority_gate':tool_agent._priority_gate,
        '_PRIORITY_UNTRIMMED_BASE':tool_agent._PRIORITY_UNTRIMMED_BASE}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[helper,cls],type_ignores=[])),str(path),'exec'),ns)
    return ns['Session']


def play_empty_session(source_root, out, started):
    Session=session_type(source_root)
    s=Session();s.started_at=started;s.game_index=0;s.pass_index=0
    s.game=types.SimpleNamespace(game_run=types.SimpleNamespace())
    s.solver=types.SimpleNamespace(max_runtime_s_per_game=900,soft_time_remaining_seconds=lambda:None)
    s.analyzer=types.SimpleNamespace(total_tokens=0)
    s.transcript_path=Path(out)/'transcript.txt';s.analysis_html_relpath='empty.html'
    # Leave the scheduling lifecycle intact; substitute only model/game/disk IO.
    for name in ['seed_initial_history','write_runtime_state','_append_initial_viewer_event','write_viewer_payload','_play_inner']:
        setattr(s,name,lambda:None)
    s.play()
    from inference.agent import tool_agent
    g=tool_agent._PRIORITY_GATE
    return {'start':g._clock_start,'deadline':g._clock_deadline,'free_slots':g._free}


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        archive=subprocess.check_output(['git','-C',str(DUCK),'archive','7652836','ARC3-Inference'])
        subprocess.run(['tar','-x','-C',str(cls.root)],input=archive,check=True)
        cls.nb=json.loads(BASE.read_text())
        patch=''.join(cls.nb['cells'][2]['source']).split('\n',1)[1]+'\n'
        subprocess.run(['git','apply','--include=ARC3-Inference/*','-'],input=patch,text=True,cwd=cls.root,check=True)
        sys.path.insert(0,str(cls.root/'ARC3-Inference'))
        cls.env=dict(os.environ)
        # Exactly the frozen notebook's scheduler settings, no invented gate behavior.
        cell=''.join(cls.nb['cells'][4]['source'])
        node=next(n for n in ast.walk(ast.parse(cell.replace('!',' #!'))) if isinstance(n,ast.Call)
                  and isinstance(n.func,ast.Attribute) and n.func.attr=='update'
                  and ast.unparse(n.func.value)=='setup_env'
                  and n.args and isinstance(n.args[0],ast.Dict))
        cls.settings={k:str(v) for k,v in ast.literal_eval(node.args[0]).items()};os.environ.update(cls.settings)
        from inference.agent import tool_agent
        cls.agent=tool_agent
        cls.record={'base_sha256':hashlib.sha256(BASE.read_bytes()).hexdigest(),
                    'duck_commit':subprocess.check_output(['git','-C',str(DUCK),'rev-parse','7652836'],text=True).strip(),
                    'patched_source_sha256':{str(p.relative_to(cls.root)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in [cls.root/'ARC3-Inference/inference/agent/tool_agent.py',cls.root/'ARC3-Inference/inference/framework/solver.py']},
                    'scope':'Full real gate module and actual session play/queue methods; game/analyzer IO omitted; no gameplay efficacy claim'}

    @classmethod
    def tearDownClass(cls):
        os.environ.clear();os.environ.update(cls.env)
        (HERE/'evidence/diagnosis-replay.json').write_text(json.dumps(cls.record,indent=2)+'\n')
        cls.tmp.cleanup()

    def test_real_session_lifecycle_fails_when_interpreter_is_reused(self):
        self.agent._PRIORITY_GATE=None
        first=play_empty_session(self.root,self.root/'first',100)
        second=play_empty_session(self.root,self.root/'second',1500)
        self.assertEqual(first,second)
        self.assertLess(second['deadline'],1500)
        # One invariant used for both the failing original and repaired processes.
        with self.assertRaises(AssertionError):self.assertEqual(second['start'],1500)
        snapshot=self.agent.PrioritySnapshot(level=1,actions=10,tokens=10000,total_levels=6)
        stale=self.agent._PRIORITY_GATE._snapshot_priority(snapshot,1500)
        fresh=self.agent._PriorityGate(14);fresh.configure_clock(1500,2400)
        correct=fresh._snapshot_priority(snapshot,1500)
        self.assertNotEqual(stale,correct)
        self.assertEqual(second['free_slots'],14)
        self.record['before']={'first':first,'second':second,'fresh_clock_invariant':False,
                               'same_snapshot_stale_priority':stale,'same_snapshot_fresh_priority':correct}

    def test_repaired_runner_isolates_real_module_and_session_lifecycle(self):
        sources=self.root/'phase-sources.json'
        sources.write_text(json.dumps({'14':'from inference.agent import tool_agent\n', '16':'pass\n',
            '20':f"from test_repair import play_empty_session\nplay_empty_session(BUNDLE_DIR, WORKING_DIR, time.monotonic())\n"}))
        receipts=[]
        for arm in ['control','candidate']:
            out=self.root/arm;out.mkdir()
            config=out/'config.json';config.write_text(json.dumps({'phase_sources':str(sources),
                'bundle_dir':str(self.root),'working_dir':str(out),'play_s':900,
                'source_entries':[str(self.root/'ARC3-Inference')]}))
            subprocess.run([sys.executable,'-B',str(HERE/'isolated_phase.py'),str(config)],check=True,timeout=20)
            receipt=json.loads((out/'harness-isolation.json').read_text());receipts.append(receipt)
            self.assertTrue(receipt['gate_is_fresh']);self.assertTrue(receipt['clock_is_fresh_and_bounded'])
            self.assertGreaterEqual(receipt['clock_start'],receipt['child_started'])
            self.assertAlmostEqual(receipt['clock_deadline']-receipt['clock_start'],900)
        self.assertNotEqual(receipts[0]['pid'],receipts[1]['pid'])
        self.record['after']=receipts

    def test_repaired_runner_rejects_a_gate_initialized_before_play(self):
        out=self.root/'early-gate';out.mkdir()
        sources=out/'phase-sources.json'
        sources.write_text(json.dumps({'14':'from inference.agent import tool_agent\ntool_agent._priority_gate()\n',
                                       '16':'raise AssertionError("must not proceed")','20':'pass'}))
        config=out/'config.json';config.write_text(json.dumps({'phase_sources':str(sources),
            'bundle_dir':str(self.root),'working_dir':str(out),'play_s':900,
            'source_entries':[str(self.root/'ARC3-Inference')]}))
        result=subprocess.run([sys.executable,'-B',str(HERE/'isolated_phase.py'),str(config)],capture_output=True,text=True,timeout=20)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('harness process already has a scheduling gate',result.stderr)
        self.assertFalse(json.loads((out/'harness-isolation.json').read_text())['gate_is_fresh'])

    def test_containment_reaps_both_owned_process_groups(self):
        spec=importlib.util.spec_from_file_location('repaired_screen',HERE/'screen.py')
        screen=importlib.util.module_from_spec(spec);spec.loader.exec_module(screen)
        children=[]
        try:
            for key in ['proc','gameplay']:
                child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(120)'],start_new_session=True)
                children.append(child);screen.OWNED[key]=child
            screen.stop_server()
            self.assertTrue(all(p.poll() is not None for p in children))
            self.assertTrue(all(p is None for p in screen.OWNED.values()))
            screen.stop_server()  # repeated cleanup is harmless
        finally:
            for child in children:
                if child.poll() is None:child.kill();child.wait()

    def test_actual_launcher_retains_success_failure_and_missing_graph_reports(self):
        launcher=ast.parse(''.join(self.nb['cells'][12]['source']))
        fn=next(n for n in launcher.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        tree=ast.parse((HERE/'build.py').read_text())
        install=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign)
                     and any(isinstance(t,ast.Name) and t.id=='install' for t in n.targets)
                     and isinstance(n.value,ast.Constant))
        body=ast.parse(install).body[-1]  # actual injected control qualification branch
        results=[]
        for mode in ['success','failure','missing_graph']:
            out=self.root/('receipt-'+mode);out.mkdir()
            test='import sys,unittest\n'
            test+='class Case(unittest.TestCase):\n def test_probe(self):\n  self.assertTrue('+str(mode!='failure')+')\n'
            test+='if __name__=="__main__":\n'
            if mode=='success':test+=' print("REFERENCE_GRAPH_TEST_PASSED")\n'
            test+=' unittest.main(verbosity=2)\n'
            (out/'test_routing.py').write_text(test)
            ns={'subprocess':subprocess,'shlex':shlex,'ARM':'control','PYTHON':sys.executable,'KIT':out,
                'WORKING_DIR':out,'install_env':dict(os.environ),'wheel':'not-needed','json':json}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'frozen-run','exec'),ns)
            with contextlib.redirect_stdout(io.StringIO()) as old_output:
                old=ns['run']([sys.executable,str(out/'test_routing.py')],check=False)
            if mode=='success':
                self.assertIn('Ran 1 test',old.stderr)
                self.assertNotIn('Ran 1 test',old_output.getvalue())
                self.assertFalse((out/'routing-tests.stderr.txt').exists())
            with contextlib.redirect_stdout(io.StringIO()):
                if mode=='success':exec(compile(ast.Module(body=[body],type_ignores=[]),'repaired-qualification','exec'),ns)
                else:
                    with self.assertRaises(RuntimeError):exec(compile(ast.Module(body=[body],type_ignores=[]),'repaired-qualification','exec'),ns)
            report=json.loads((out/'routing-tests.json').read_text())
            self.assertIn('Ran 1 test',(out/'routing-tests.stderr.txt').read_text())
            self.assertEqual(report['returncode'],int(mode=='failure'))
            self.assertEqual(report['gpu_graph_executed'],mode=='success')
            results.append({'mode':mode,**report,'stderr_saved':True})
        self.record['launcher_receipts']=results


if __name__=='__main__':unittest.main(verbosity=2)
