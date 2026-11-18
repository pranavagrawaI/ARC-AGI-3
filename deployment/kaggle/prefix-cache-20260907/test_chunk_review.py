"""Independent review checks for the joint chunk/cache experiment."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


baseline = load('review_baseline', HERE / 'baseline.py')


class ExperimentReview(unittest.TestCase):
    def setUp(self):
        self.runner = load('review_runner', HERE / 'runner.py')

    def test_chunk_is_only_launch_argument_change(self):
        def command(*args, **kwargs):
            return ['sglang', 'serve', '--chunked-prefill-size', '2048',
                    '--mem-fraction-static', '0.85', '--attention-backend', 'triton',
                    '--sampling-backend', 'pytorch']
        runtime = SimpleNamespace(sglang_command=command)
        a = self.runner.engine_command(runtime, Path('/model'), 2048)
        b = self.runner.engine_command(runtime, Path('/model'), 4096)
        self.assertEqual([(x,y) for x,y in zip(a,b) if x != y], [('2048','4096')])
        self.assertEqual(len(a),len(b))
        self.assertEqual(tuple(self.runner.CHUNKS),(2048,4096,4096,2048))
        with self.assertRaises(ValueError):self.runner.engine_command(runtime,Path('/model'),8192)

    def test_built_asset_execution_matches_reviewed_sources(self):
        builder=load('review_builder',HERE/'build.py')
        files=builder.build()
        for name,data in files.items():
            self.assertEqual((HERE/name).read_bytes(),data,'Rebuild required: '+name)
        nb=json.loads(files['prefix_cache.ipynb'])
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def mapped_path(value):
                path=Path(value)
                self.assertEqual(path.parent,Path('/tmp'))
                return root/path.name
            with patch('pathlib.Path',mapped_path):
                exec(compile(''.join(nb['cells'][2]['source']),'embedded-assets','exec'),{})
            for target,source in [('actor-replay.py','runner.py'),('baseline.py','baseline.py'),('prefix-assess.py','assess.py'),('prefix-corpus.json','corpus.json')]:
                self.assertEqual((root/target).read_bytes(),(HERE/source).read_bytes())
        meta=json.loads(files['kernel-metadata.json'])
        self.assertEqual(meta['competition_sources'],['arc-prize-2026-arc-agi-3'])
        self.assertEqual(meta['machine_shape'],'NvidiaRtxPro6000')
        self.assertFalse(meta['enable_internet'])

    def test_fixed_control_retains_gregg_sampling(self):
        p=self.runner.fixed_payload([1,2,3])
        self.assertEqual(p['sampling_params'], self.runner.base.payload([1,2,3])['sampling_params'])
        self.assertEqual(p['sampling_params']['max_new_tokens'],256)
        self.assertNotIn('return_token_ids',p,'Unsupported in pinned native GenerateReqInput')

    def test_resolved_chunk_gate_keeps_graph_evidence(self):
        info={'chunked_prefill_size':4096,'mem_fraction_static':.85,
              'speculative_algorithm':None,'sampling_defaults':'model',
              'cuda_graph_config':{'prefill':{'max_bs':4096}},
              'mamba_radix_cache_strategy':'extra_buffer','mamba_track_interval':256}
        resolved=self.runner.validate_server_info(info,4096)
        self.assertEqual(resolved['cuda_graph_config'],info['cuda_graph_config'])
        with self.assertRaises(RuntimeError):self.runner.validate_server_info(info,2048)

    def test_reused_ids_still_validate_natural_output_and_identical_batch_inputs(self):
        r=self.runner
        corpus=json.loads((HERE/'corpus.json').read_text())
        names={record['name'] for pair in corpus['pairs'] for record in (pair['seed'],pair['successor'])}
        names.update(record['name'] for record in corpus['batch_records'])
        rendered={name:[1,2,3,int(name.split('-')[-1])] for name in names}
        events=[];generations=[];chats=[]
        def http(route,*args):events.append(('http',route))
        def gen(name,body):
            generations.append((name,body));events.append(('generate',name))
            return {'name':name,'meta_info':{'completion_tokens':body['sampling_params']['max_new_tokens'],
                    'prompt_tokens':len(body['input_ids']),'cached_tokens':0},
                    'cached_input_tokens':0,'server_forward_to_prefill_finished_s':1.0}
        def chat(name,record,*args):
            chats.append(name);events.append(('chat',name))
            return rendered[record['name']]
        with tempfile.TemporaryDirectory() as directory,patch.object(r,'OUT',Path(directory)),patch.object(r.base,'OUT',Path(directory)),patch.object(r.base,'http',http),patch.object(r.base,'generate',gen),patch.object(r.base,'chat',chat):
            result=r.experiments(corpus,Path('/model'),None,rendered=rendered,reverse=True)
            self.assertEqual(result,rendered)
            self.assertTrue(chats,'4096/reused-ID arms must receive natural Actor validation too')
            self.assertEqual(len(chats),9)
            self.assertEqual(len(generations),28)
            self.assertEqual(r.REQUEST_COUNT,37)
            for name,body in generations:
                if name.endswith('-successor'):
                    self.assertEqual(body['sampling_params']['max_new_tokens'],256)
                if name.endswith('-prime'):
                    ix=events.index(('generate',name))
                    self.assertEqual(events[ix+1],('generate',name.removesuffix('-prime')+'-successor'))
            for workload in ['captured','long95000']:
                bodies=[]
                for c in [1,4]:
                    bodies.append({name.split('-case-')[-1]:body for name,body in generations if name.startswith(f'batch-{workload}-{c}-')})
                self.assertEqual(bodies[0],bodies[1])

    def test_request_cap_rejects_before_transmission(self):
        r=self.runner
        with tempfile.TemporaryDirectory() as directory,patch.object(r,'OUT',Path(directory)),patch.object(r,'REQUEST_COUNT',148),patch.object(r.base,'generate') as generate:
            with self.assertRaisesRegex(RuntimeError,'cap reached'):
                r.checked_generate('extra',r.fixed_payload([1,2]))
            generate.assert_not_called()

    def test_cross_arm_render_drift_fails(self):
        r=self.runner
        corpus=json.loads((HERE/'corpus.json').read_text())
        records=[record for pair in corpus['pairs'] for record in (pair['seed'],pair['successor'])]+corpus['batch_records']
        expected={record['name']:[1,2] for record in records}
        with tempfile.TemporaryDirectory() as directory,patch.object(r,'OUT',Path(directory)),patch.object(r.base,'OUT',Path(directory)),patch.object(r.base,'http'),patch.object(r.base,'chat',return_value=[1,3]):
            with self.assertRaisesRegex(RuntimeError,'Cross-arm rendered token drift'):
                r.render_records(corpus,Path('/model'),None,expected=expected)

    def test_arm_failure_retires_engine_and_preserves_primary_error(self):
        r=self.runner
        info={'chunked_prefill_size':2048,'mem_fraction_static':.85,
              'speculative_algorithm':None,'sampling_defaults':'model'}
        for cleanup_fails in [False,True]:
            events=[]
            def lifecycle(*args,**kwargs):
                events.append('lifecycle')
                if cleanup_fails:raise RuntimeError('cleanup failure')
                return {'status':'passed'}
            def contain(*args):
                events.append('contain')
                return {'status':'contained'}
            runtime=SimpleNamespace(R8_L3B_OWNER='fixture',
                sglang_command=lambda *args,**kw:['fixture-server'],
                process_identity=lambda *args:{},wait_for_server=lambda *args:None,
                wait_for_sglang_warmup=lambda *args:None,
                qualify_lifecycle=lifecycle,emergency_contain_process_group=contain)
            with tempfile.TemporaryDirectory() as directory,patch.object(r,'OUT',Path(directory)),patch.object(r.base,'OUT',Path(directory)),patch.object(r.base,'PROCESS',None),patch.object(r.base,'http',lambda *args:json.dumps(info)),patch.object(r.base,'telemetry',lambda stop:None),patch.object(r.subprocess,'Popen',return_value=SimpleNamespace(pid=42,poll=lambda:None)),patch.object(r,'experiments',side_effect=RuntimeError('primary trial failure')):
                with self.assertRaisesRegex(RuntimeError,'primary trial failure'):
                    r.run_arm(runtime,Path('/model'),None,{},0,2048,None)
                self.assertEqual(events,['lifecycle','contain'] if cleanup_fails else ['lifecycle'])


class TransportReview(unittest.TestCase):
    def test_c4_common_ready_and_actual_client_overlap(self):
        barrier = threading.Barrier(4, timeout=3)

        def call(name):
            barrier.wait()
            return {'name': name, 'meta_info': {'completion_tokens': 256}}

        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / 'events.jsonl'
            result = baseline.finite_batch([(str(i),) for i in range(4)], 4, call, events)
            rows = [json.loads(line) for line in events.read_text().splitlines()]
        self.assertEqual(len({row['workload_ready_monotonic_s'] for row in result['requests']}), 1)
        self.assertAlmostEqual(result['aggregate_generated_tok_s'] * result['wall_s'], 1024)
        self.assertEqual(sum(row['event'] == 'REQUEST_COMPLETED' for row in rows), 4)
        for row in result['requests']:
            self.assertGreaterEqual(row['ready_to_completed_s'], row['dispatch_to_completed_s'])

    def test_failure_closes_admission(self):
        admitted = []

        def fail(name):
            admitted.append(name)
            raise RuntimeError('terminal state unknown')

        with tempfile.TemporaryDirectory() as directory:
            events = Path(directory) / 'events.jsonl'
            with self.assertRaisesRegex(RuntimeError, 'terminal state unknown'):
                baseline.finite_batch([(str(i),) for i in range(4)], 1, fail, events)
            rows = [json.loads(line) for line in events.read_text().splitlines()]
        self.assertEqual(admitted, ['0'])
        self.assertEqual(rows[-1]['event'], 'WORKLOAD_ABORTED')

    def test_timing_is_not_http_latency_and_cache_unknown_is_unknown(self):
        measured = baseline.decode_stats({'completion_tokens': 256, 'decode_throughput': 40,
                                         'forward_entry_time': 100, 'prefill_finished_time': 102.5})
        self.assertEqual(measured['server_tokenizer_decode_span_s'], 255 / 40)
        self.assertEqual(measured['server_forward_to_prefill_finished_s'], 2.5)
        self.assertIsNone(baseline.cache_stats({'prompt_tokens': 100})['new_input_tokens'])
        self.assertEqual(baseline.cache_stats({'prompt_tokens': 100, 'cached_tokens': 80})['new_input_tokens'], 20)


if __name__ == '__main__':
    unittest.main()
