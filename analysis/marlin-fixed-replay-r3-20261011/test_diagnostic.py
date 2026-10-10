import datetime as dt
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import numpy as np
import benchmark
import audit_capture
import launch
from build_notebook import build,HERE
from reference import errors,paired_summary,qualify_outputs


class DiagnosticTests(unittest.TestCase):
    def test_control_envelope_distinguishes_noise_from_cross_arm_change(self):
        first=np.ones((2,4),np.float32)
        control=np.stack([first,first+.001]*4)
        duplicate=np.stack([first+.002]*8)
        treatment=np.stack([first+.01]*8)
        stats=benchmark.repeatability_metrics(control,treatment,duplicate,first)
        self.assertAlmostEqual(stats['control_max_abs_envelope'],.002,places=6)
        self.assertTrue(stats['max_cross_arm_exceeds_control_envelope'])
        stable=np.stack([first]*8)
        self.assertFalse(benchmark.repeatability_metrics(stable,stable,stable,first)['max_cross_arm_exceeds_control_envelope'])

    def test_live_allowance_prevents_historical_quota_override(self):
        quota=types.SimpleNamespace(time_used=dt.timedelta(seconds=34386),time_reserved=dt.timedelta(),total_time_allowed=dt.timedelta(seconds=21600))
        response=types.SimpleNamespace(gpu_quota=quota,quota_refresh_time=dt.datetime(2026,10,17))
        with self.assertRaisesRegex(ValueError,'Live reported allowance'):
            launch.quota_receipt(response)
        quota.total_time_allowed=dt.timedelta(seconds=108000)
        self.assertEqual(launch.quota_receipt(response)['reported_allowed_seconds'],108000)

    def test_generated_notebook_has_no_inline_patch_argv_or_server_launch(self):
        notebook=build()
        import nbformat
        nbformat.validate(nbformat.from_dict(notebook))
        self.assertEqual(len(notebook['cells']),5)
        for i,c in enumerate(notebook['cells']):
            if c['cell_type']=='code':compile(''.join(c['source']),str(i),'exec')
        source=''.join(''.join(c['source']) for c in notebook['cells'])
        self.assertNotIn('serve_process',source)
        self.assertNotIn('precache_model',source)
        self.assertIn("KIT/'apply_patch.py'",source)
        self.assertIn("timeout=2400",source)
        meta=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
        self.assertEqual(meta['kernel_sources'],['pranavagrawai/marlin-reduction-replay-v3-20261010'])

    def fixture(self,case):
        record={'case':'test-case.pt'};first=np.ones((2,4),np.float32)
        # Both arms fail v3's ideal-reference gate, yet remain finite and identical.
        actual=first+.5;arrays=dict(baseline=actual,treatment=actual,poisoned=actual,fp32=first)
        rows=[]
        for variant in ['original','changed','restored']:
            np.savez(case/f'{variant}-outputs.npz',**arrays)
            repeated={k:np.stack([actual]*8) for k in ['baseline','treatment','independent_control']}
            np.savez(case/f'{variant}-repeats.npz',**repeated)
            np.savez(case/f'{variant}-dtype-reference.npz',dtype_reference=actual)
            rows.append({'variant':variant,**qualify_outputs(**arrays),
                         'repeatability':benchmark.repeatability_metrics(**repeated,fp32=first),
                         'dtype_reference':{arm:errors(arrays[arm],actual) for arm in ['baseline','treatment']}})
        np.savez(case/'eager-outputs.npz',baseline=actual,treatment=actual)
        summaries={}
        for mode in ['graph','eager','control-graph']:
            speed=.9 if mode!='control-graph' else 1.
            pairs=[{'pair':i,'order':['baseline','treatment'] if i%2==0 else ['treatment','baseline'],'repeats':64,'ms_per_call':{'baseline':1.,'treatment':speed}} for i in range(30)]
            (case/f'{mode}-pairs.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in pairs))
            summaries[mode]=paired_summary([[1,speed]]*30)
        result={'record':record,'actual_gemm_calls':[{'use_atomic_add':flag,'use_fp32_reduce':True} for flag in [True,True,False,False]],'inputs_before':{'x':'same'},'inputs_after':{'x':'same'},'numerical':rows,'eager_graph':[errors(actual,actual)]*2,'numerical_passed':False,'graph_timing':summaries['graph'],'eager_timing':summaries['eager'],'control_graph_timing':summaries['control-graph']}
        (case/'result.json').write_text(json.dumps(result))
        return record

    def test_rejected_numerics_keep_diagnostic_timings_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);record=self.fixture(case)
            result=audit_capture.audit_case(case,record)
            self.assertFalse(result['v3_numerical_passed'])
            self.assertGreater(result['timings']['graph']['geometric_speedup'],1.1)
            repeats=dict(np.load(case/'changed-repeats.npz'))
            repeats['baseline'][1,0,0]+=1
            np.savez(case/'changed-repeats.npz',**repeats)
            with self.assertRaisesRegex(ValueError,'Repeatability claim'):
                audit_capture.audit_case(case,record)

    def test_missing_pair_is_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);record=self.fixture(case)
            path=case/'graph-pairs.jsonl';path.write_text('\n'.join(path.read_text().splitlines()[:-1])+'\n')
            with self.assertRaisesRegex(ValueError,'Missing timing pairs'):
                audit_capture.audit_case(case,record)

    def test_finite_eager_graph_variation_is_retained_without_suppressing_timing(self):
        with tempfile.TemporaryDirectory() as tmp:
            case=Path(tmp);record=self.fixture(case)
            restored=dict(np.load(case/'restored-outputs.npz'))
            eager={'baseline':restored['baseline']+.05,'treatment':restored['treatment']}
            np.savez(case/'eager-outputs.npz',**eager)
            result=json.loads((case/'result.json').read_text())
            result['eager_graph']=[errors(restored[arm],eager[arm]) for arm in ['baseline','treatment']]
            (case/'result.json').write_text(json.dumps(result))
            derived=audit_capture.audit_case(case,record)
            self.assertGreater(derived['eager_graph'][0]['nmax'],.01)
            self.assertFalse(derived['v3_numerical_passed'])
            self.assertGreater(derived['timings']['graph']['geometric_speedup'],1.1)


if __name__=='__main__':unittest.main()
