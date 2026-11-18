import ast
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import marlin_capture as capture
import prepare
import reference
from build_notebook import build_notebook

HERE=Path(__file__).resolve().parent


class Tensor:
    def __init__(self, data):self.data=np.asarray(data);self.device='cuda';self.dtype=self.data.dtype
    @property
    def shape(self):return self.data.shape
    def size(self,index):return self.shape[index]
    def is_contiguous(self):return True
    def view(self,*shape):return Tensor(self.data.reshape(*shape))
    def __getitem__(self,item):return Tensor(self.data[item])
    def zero_(self):self.data.fill(0)
    def detach(self):return self
    def cpu(self):return self
    def flatten(self):return Tensor(self.data.flatten())
    def tolist(self):return self.data.tolist()
    def numel(self):return self.data.size
    def element_size(self):return self.data.itemsize
    def to(self,dtype):return self


class PreparationTests(unittest.TestCase):
    def test_inverse_known_words(self):
        # Low nibble is q[0,0], next q[8,0], then q[0,8], q[8,8],
        # then q[1,0], q[9,0], q[1,8], q[9,8]. Independent golden locations.
        words=np.full(128,np.uint32(0x88888888),dtype=np.uint32)
        words[0]=np.uint32(0xFEDCBA98)
        result=reference.unpack_marlin(words.view(np.int32),16,64)
        for (row,col),value in zip([(0,0),(8,0),(0,8),(8,8),(1,0),(9,0),(1,8),(9,8)],range(8)):
            self.assertEqual(result[row,col],value)
        self.assertEqual(np.count_nonzero(result),7)

    def test_inverse_multiple_tiles_and_warps(self):
        words=np.full((2,2,128),np.uint32(0x88888888),dtype=np.uint32)
        words[1,1,31*4+3]=np.uint32(0x77777777)
        result=reference.unpack_marlin(words.view(np.int32),32,128)
        self.assertEqual(np.count_nonzero(result),8)
        for row in [22,23,30,31]:
            for col in [119,127]:self.assertEqual(result[row,col],-1)

    def test_scale_known_columns(self):
        source=np.arange(128,dtype=np.float32).reshape(2,64)
        permuted=np.stack([source[row].reshape(8,8).T.ravel() for row in range(2)])
        np.testing.assert_array_equal(reference.unpermute_scales(permuted,256),source)

    def test_zero_weight_reference(self):
        w=np.full((8,256),np.int32(-2004318072),dtype=np.int32) # unsigned0x88888888
        result=reference.dequantize(w,np.ones((2,64)),256,64)
        np.testing.assert_array_equal(result,np.zeros((256,64)))

    def test_invalid_layout_rejected(self):
        with self.assertRaises(ValueError):reference.unpack_marlin(np.zeros(1),15,64)

    def test_numerics_reject_drift_and_nonfinite(self):
        b=reference.errors(np.ones(100)*1.001,np.ones(100));t=reference.errors(np.ones(100)*1.002,np.ones(100));drift=reference.errors(np.ones(100)*1.001,np.ones(100))
        self.assertFalse(reference.numerical_gate(b,t,drift))
        self.assertFalse(reference.numerical_gate(b,b,reference.errors(np.full(100,np.nan),np.ones(100))))
        zero=reference.errors(np.ones(100),np.ones(100))
        self.assertTrue(reference.numerical_gate(zero,zero,zero))

    def test_v4_gate_passes_v3_case1_control_parity_and_rejects_worse_treatment(self):
        # Actual v3 case-1 numbers (layer-00-replay-08); the v3 gate rejected the `changed` row.
        def e(max_abs,rms,ref):return {'finite':True,'max_abs':max_abs,'rms':rms,'reference_rms':ref,'nrms':rms/ref,'nmax':max_abs/ref}
        rows={'original':(e(.0015268325805664062,7.106398084657034e-05,.0224471637598245),e(.0015268325805664062,7.04603263601999e-05,.0224471637598245),e(.0009765625,5.2461894189145225e-05,.0224471637598245)),
              'changed':(e(.000535234808921814,5.2943288572409845e-06,.0010662972107542221),e(.000535234808921814,5.184269256109566e-06,.0010662972107542221),e(.0001220703125,3.45720060413742e-06,.0010662972107542221))}
        for name,(b,t,d) in rows.items():self.assertTrue(reference.numerical_gate(b,t,d),name)
        b,t,d=rows['changed']
        self.assertFalse(reference.numerical_gate(b,t,e(b['max_abs']*1.01,d['rms'],d['reference_rms'])))  # drift beyond control's own error
        self.assertFalse(reference.numerical_gate(b,e(b['max_abs']*1.2,t['rms'],t['reference_rms']),d))  # treatment max worse than control
        self.assertFalse(reference.numerical_gate(b,e(t['max_abs'],b['rms']*1.2,t['reference_rms']),d))  # treatment RMS worse than control
        self.assertFalse(reference.numerical_gate(b,t,e(d['max_abs'],.006*d['reference_rms'],d['reference_rms'])))  # drift RMS
        gross=e(.1,.04*.0224,.0224);self.assertFalse(reference.numerical_gate(gross,gross,e(0,0,.0224)))  # oracle/kernel disagreement

    def test_paired_gate_resolves_tie_and_gain(self):
        tie=reference.paired_summary([[1,1]]*30);gain=reference.paired_summary([[1,.9]]*30)
        self.assertEqual(tie['geometric_speedup'],1)
        self.assertGreater(gain['bootstrap_95'][0],1.1)
        self.assertEqual(gain['faster_pairs'],30)
        with self.assertRaises(ValueError):reference.paired_summary([[1,0]])

    def test_target_only_binding(self):
        class Kernel:pass
        kernels=[Kernel() for _ in range(48)]
        model=types.SimpleNamespace(named_modules=lambda:[(f'layer.{i}',types.SimpleNamespace(scheme=types.SimpleNamespace(kernel=k))) for i,k in enumerate(kernels)])
        module=types.ModuleType('sglang.srt.hardware_backend.gpu.quantization.gptq_kernels');module.GPTQMarlinMoEKernel=Kernel
        with patch.dict(sys.modules,{'sglang.srt.hardware_backend.gpu.quantization.gptq_kernels':module}):
            target=types.SimpleNamespace(is_draft_worker=False,model=model)
            draft=types.SimpleNamespace(is_draft_worker=True,model=model)
            self.assertEqual(len(capture.target_kernels(target)),48)
            self.assertEqual(capture.target_kernels(draft),[])

    def test_binding_modes_and_layer_selection(self):
        import tempfile
        kernels=[types.SimpleNamespace() for _ in range(48)]
        items=[(f'layer.{i}',None,k) for i,k in enumerate(kernels)]
        with tempfile.TemporaryDirectory() as out:
            cfg={'mode':'capture','out':out,'expected_target_layers':48,'layer_ordinals':[0,24,47]}
            with patch.object(capture,'config',return_value=cfg),patch.object(capture,'target_kernels',return_value=items):
                capture.bind_model(types.SimpleNamespace(is_draft_worker=False))
            self.assertEqual([i for i,k in enumerate(kernels) if hasattr(k,'experiment_capture_tag')],[0,24,47])
            self.assertTrue(all(k.experiment_atomic_override is None for k in kernels))
            cfg['mode']='treatment'
            with patch.object(capture,'config',return_value=cfg),patch.object(capture,'target_kernels',return_value=items):
                capture.bind_model(types.SimpleNamespace(is_draft_worker=False))
            self.assertTrue(all(k.experiment_atomic_override is False for k in kernels))

    def test_graph_key_selects_capture_buffers(self):
        cfg={'mode':'capture'};key=types.SimpleNamespace(size=10)
        runner=types.SimpleNamespace(model_runner=types.SimpleNamespace(is_draft_worker=False))
        with patch.object(capture,'config',return_value=cfg):capture.begin_capture(runner,key)
        self.assertEqual(capture._ACTIVE,(id(runner.model_runner),repr(key)))
        capture.end_capture();self.assertIsNone(capture._ACTIVE)
        runner.model_runner.is_draft_worker=True
        with patch.object(capture,'config',return_value=cfg):capture.begin_capture(runner,key)
        self.assertIsNone(capture._ACTIVE)

    def test_actual_fused_function_passes_both_reduction_flags(self):
        source=prepare.revised_sources({key:(HERE/'source'/name).read_text() for key,name in prepare.FILES.items()})['fused']
        fn=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='fused_marlin_moe');fn.decorator_list=[]
        calls=[]
        def gemm(*args,**kwargs):calls.append((args,kwargs));return args[1]
        fake=types.SimpleNamespace(Tensor=Tensor,half=np.dtype('float16'),float16=np.dtype('float16'),bfloat16=np.dtype('float32'),float8_e8m0fnu='e8m0',int=np.int32,
            cuda=types.SimpleNamespace(get_device_capability=lambda _: (12,0),get_device_properties=lambda _:types.SimpleNamespace(multi_processor_count=144)),
            empty=lambda shape,**kw:Tensor(np.zeros(shape,dtype=kw.get('dtype',np.float16))),zeros=lambda shape,**kw:Tensor(np.zeros(shape,dtype=kw.get('dtype',np.float16))),empty_like=lambda x:Tensor(np.zeros_like(x.data)))
        env={'torch':fake,'Optional':__import__('typing').Optional,'get_scalar_type':lambda *a:4,'moe_wna16_marlin_gemm':gemm,'silu_and_mul':lambda *a:None,'moe_sum_reduce':lambda *a:None,'zero_copy_context':types.SimpleNamespace(get_moe_output=lambda x:None)}
        module=types.ModuleType('sglang.srt.layers.moe.fused_moe_triton');module.moe_align_block_size=lambda *a:(Tensor(np.arange(32)),None,None)
        exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[])),'<actual-fused>','exec'),env)
        args={'hidden_states':Tensor(np.zeros((2,256),np.float16)),'w1':Tensor(np.zeros((4,16,512),np.int32)),'w2':Tensor(np.zeros((4,8,512),np.int32)),
              'w1_scale':Tensor(np.ones((4,2,256),np.float16)),'w2_scale':Tensor(np.ones((4,1,256),np.float16)),
              'gating_output':Tensor(np.zeros((2,4),np.float16)),'topk_weights':Tensor(np.ones((2,2),np.float32)),'topk_ids':Tensor(np.zeros((2,2),np.int32)),'num_bits':4}
        with patch.dict(sys.modules,{'sglang.srt.layers.moe.fused_moe_triton':module}):
            for override,expected in [(None,True),(False,False)]:
                calls.clear();env['fused_marlin_moe'](**args,use_atomic_add_override=override)
                self.assertEqual(len(calls),2)
                self.assertEqual([c[1]['use_atomic_add'] for c in calls],[expected,expected])
                self.assertTrue(all(c[1]['use_fp32_reduce'] for c in calls))
                self.assertEqual([c[1]['mul_topk_weights'] for c in calls],[False,True])
                self.assertIs(calls[0][0][13],args['topk_weights'])

    def test_patch_does_not_remove_precision_or_atomic_architecture_rule(self):
        manifest=prepare.build()
        source=(HERE/'patched'/prepare.FILES['fused']).read_text()
        self.assertEqual(source.count('use_fp32_reduce=True'),2)
        self.assertIn('get_device_capability(hidden_states.device)[0] >= 9',source)
        self.assertEqual(len(manifest),5)
        with self.assertRaises(ValueError):prepare.replace_once('xx','x','y')

    def test_poison_targets_actual_native_scratch_and_restores_factory(self):
        import benchmark
        calls=[]
        scratch=types.SimpleNamespace(numel=lambda:16,fill_=lambda value:calls.append(('fill',np.isnan(value))))
        native=types.SimpleNamespace(moe_wna16_marlin_gemm=lambda *values:calls.append(('native',values[15])))
        factory=lambda *args:native
        module=types.SimpleNamespace(_jit_moe_wna16_marlin_module=factory)
        with patch('importlib.import_module',return_value=module):
            with benchmark.poison_actual_scratch():
                module._jit_moe_wna16_marlin_module(None,False,False).moe_wna16_marlin_gemm(*([None]*15+[scratch]))
            self.assertIs(module._jit_moe_wna16_marlin_module,factory)
        self.assertEqual(calls,[('fill',True),('native',scratch)])

    def test_raw_audit_recomputes_performance_and_rejects_fabricated_verdict(self):
        import tempfile
        import audit_capture
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);records=[];summaries=[]
            for i in range(9):
                record={'case':f'case-{i}.pt'};records.append(record)
                case=root/f'case-{i}';case.mkdir()
                arrays={k:np.ones((2,4),np.float32) for k in ['baseline','treatment','poisoned','fp32']}
                numerical=[]
                for variant in ['original','changed','restored']:
                    np.savez(case/f'{variant}-outputs.npz',**arrays)
                    numerical.append({'variant':variant,**reference.qualify_outputs(**arrays)})
                np.savez(case/'eager-outputs.npz',baseline=arrays['baseline'],treatment=arrays['treatment'])
                summary=reference.paired_summary([[1,.9]]*30);summaries.append(summary)
                for mode in ['graph','eager']:
                    rows=[{'pair':j,'order':['baseline','treatment'] if j%2==0 else ['treatment','baseline'],'repeats':64,'ms_per_call':{'baseline':1,'treatment':.9}} for j in range(30)]
                    (case/f'{mode}-pairs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
                result={'record':record,'actual_gemm_calls':[{'use_atomic_add':a,'use_fp32_reduce':True} for a in [True,True,False,False]],'numerical':numerical,'numerical_passed':True,
                        'eager_graph':[reference.errors(arrays['baseline'],arrays['baseline'])]*2,'inputs_before':{'hash':'same'},'inputs_after':{'hash':'same'},'graph_timing':summary,'eager_timing':summary}
                (case/'result.json').write_text(json.dumps(result))
            claimed={'cases_completed':9,'sentinel':{'passed':True},'advance_to_serving_preparation':True,'case_graph_summaries':summaries}
            (root/'RESULT.json').write_text(json.dumps(claimed))
            self.assertTrue(audit_capture.audit_benchmark(root,records)['advance_to_serving_preparation'])
            # Alter the raw treatment interval; derived summary/advancement must fail.
            path=root/'case-0/graph-pairs.jsonl';rows=[json.loads(line) for line in path.read_text().splitlines()]
            rows[0]['ms_per_call']['treatment']=10
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            with self.assertRaisesRegex(ValueError,'summary differs'):audit_capture.audit_benchmark(root,records)

    def test_nonfinite_raw_arrays_reject_poison_qualification(self):
        arrays=[np.ones((2,4))]*3+[np.ones((2,4))]
        arrays[2]=np.full((2,4),np.nan)
        self.assertFalse(reference.qualify_outputs(*arrays)['passed'])

    def test_capture_flushes_real_selected_graph_and_reuses_weight_artifact(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);(out/'ARM-CAPTURE').write_text('armed')
            cfg={'mode':'capture','out':tmp,'selected_tags':['layer-00'],'replay_ordinals':[8,16,32],'max_capture_bytes':1024*1024}
            runner=types.SimpleNamespace(model_runner=types.SimpleNamespace(is_draft_worker=False),_replay_graph_key='actual-key',raw_num_token=2)
            batch=types.SimpleNamespace(batch_size=10,forward_mode=types.SimpleNamespace(name='TARGET_VERIFY'))
            key=(id(runner.model_runner),repr(runner._replay_graph_key))
            buf={'hidden_states':Tensor(np.ones((2,4))),'gating_output':Tensor(np.ones((2,4))),
                 'topk_weights':Tensor(np.ones((2,2))),'topk_ids':Tensor(np.array([[0,1],[1,2]],np.int32))}
            args={k:None for k in capture.WEIGHTS+capture.SCALARS};args['w1']=Tensor(np.ones((4,4)));args['w2']=Tensor(np.ones((4,4)))
            def save(value,stream):stream.write(json.dumps(value,default=lambda t:t.tolist()).encode())
            def unique(value,return_counts):
                values,counts=np.unique(value.data,return_counts=True);return Tensor(values),Tensor(counts)
            fake=types.SimpleNamespace(Tensor=Tensor,save=save,unique=unique,cuda=types.SimpleNamespace(synchronize=lambda:None))
            with patch.object(capture,'config',return_value=cfg),patch.object(capture,'_BUFFERS',{(key,'layer-00'):(buf,args)}),patch.object(capture,'_COUNTS',{}),patch.dict(sys.modules,{'torch':fake}):
                for ordinal in [8,16,32]:
                    capture._COUNTS[key]=ordinal-1;buf['hidden_states'].data[:]=ordinal
                    capture.after_execute(runner,batch)
            records=[json.loads(line) for line in (out/'cases.jsonl').read_text().splitlines()]
            self.assertEqual([r['replay_ordinal'] for r in records],[8,16,32])
            self.assertEqual(len(list(out.glob('*weights.pt'))),1)
            self.assertEqual(len({r['weights_sha256'] for r in records}),1)
            self.assertEqual(records[0]['expert_counts'],[[0,1],[1,2],[2,1]])
            self.assertEqual(json.loads((out/'layer-00-replay-32.pt').read_text())['hidden_states'],[[32]*4]*2)

    def test_original_raw_baseline_profile_still_qualifies(self):
        import audit_capture
        root=HERE.parent/'serving-headroom-v3-20261008/readout/raw/headroom'
        warmup=audit_capture.v3.verify_block(root,'WARMUP',10)
        profile=audit_capture.v3.verify_block(root,'PROFILE10',10)
        self.assertEqual(warmup['generated_tokens'],320)
        self.assertEqual(profile['generated_tokens'],40960)
        self.assertEqual(profile['accounting']['retractions'],0)
        self.assertTrue(profile['traces'])

    def test_raw_audit_rejects_missing_evidence(self):
        import audit_capture,tempfile
        with tempfile.TemporaryDirectory() as tmp:
            result=audit_capture.audit(Path(tmp))
            self.assertFalse(result['qualified'])
            self.assertIn('FileNotFoundError',result['error'])

    def test_actual_gptq_to_marlin_runner_propagates_target_override(self):
        sources=prepare.revised_sources({key:(HERE/'source'/name).read_text() for key,name in prepare.FILES.items()})
        tree=ast.parse(sources['gptq']);klass=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='GPTQMarlinMoEKernel')
        apply_fn=next(n for n in klass.body if isinstance(n,ast.FunctionDef) and n.name=='apply')
        runner_fn=next(n for n in ast.parse(sources['runner']).body if isinstance(n,ast.FunctionDef) and n.name=='fused_experts_none_to_marlin');runner_fn.decorator_list=[]
        def compile_function(fn,env):
            node=ast.fix_missing_locations(ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),fn],type_ignores=[]))
            exec(compile(node,'<actual-call-chain>','exec'),env);return env[fn.name]
        seen=[]
        def fused(**kw):seen.append(kw);return kw['hidden_states']
        fused_module=types.ModuleType('sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe');fused_module.fused_marlin_moe=fused
        standard=types.ModuleType('sglang.srt.layers.moe.token_dispatcher.standard');standard.StandardCombineInput=types.SimpleNamespace
        utils=types.ModuleType('sglang.srt.layers.quantization.marlin_utils');utils.marlin_make_workspace=lambda *a,**kw:None
        def info(**kw):
            optional={k:None for k in ['w13_qzeros','w2_qzeros','expert_map','w13_global_scale','w2_global_scale','w13_bias','w2_bias']}
            return types.SimpleNamespace(**dict(optional,global_num_experts=-1,**kw))
        env={'torch':types.SimpleNamespace(float8_e8m0fnu='e8m0',float16=np.dtype('float16'),bfloat16=np.dtype('float32'))}
        run=compile_function(runner_fn,env);apply=compile_function(apply_fn,{'MarlinMoeQuantInfo':info})
        x=Tensor(np.ones((2,4),np.float16));layer=types.SimpleNamespace(**{k:x for k in ['w13_qweight','w2_qweight','w13_scales','w2_scales','w13_g_idx','w2_g_idx','w13_g_idx_sort_indices','w2_g_idx_sort_indices']})
        dispatch=types.SimpleNamespace(hidden_states=x,topk_output=types.SimpleNamespace(topk_ids=x,topk_weights=x,router_logits=x))
        config=types.SimpleNamespace(is_gated=True,activation='silu',routed_scaling_factor=1.,gemm1_alpha=None,gemm1_clamp_limit=None,swiglu_limit=None)
        kernel=types.SimpleNamespace(quant_config=types.SimpleNamespace(weight_bits=4),is_k_full=True,runner=types.SimpleNamespace(run=lambda d,q:run(d,q,config)))
        modules={m.__name__:m for m in [fused_module,standard,utils]}
        with patch.dict(sys.modules,modules):
            apply(kernel,layer,dispatch)
            self.assertIsNone(seen[-1]['use_atomic_add_override'])
            kernel.experiment_atomic_override=False;kernel.experiment_capture_tag='layer-00'
            apply(kernel,layer,dispatch)
        self.assertIs(seen[-1]['use_atomic_add_override'],False)
        self.assertEqual(seen[-1]['experiment_capture_tag'],'layer-00')
        self.assertIs(seen[-1]['w1'],layer.w13_qweight)
        self.assertIs(seen[-1]['topk_weights'],dispatch.topk_output.topk_weights)

    def test_reference_rejects_empty_zero_points_and_noninteger_quantization(self):
        args={k:None for k in capture.WEIGHTS+capture.INPUTS+capture.SCALARS}
        args.update(num_bits=4,is_k_full=True,inplace=False,is_gated=True,activation='silu',hidden_states=Tensor(np.ones((2,4),np.float32)),
                    w1=Tensor(np.ones((4,4),np.int32)),w2=Tensor(np.ones((4,4),np.int32)),w1_scale=Tensor(np.ones((4,4),np.float32)),w2_scale=Tensor(np.ones((4,4),np.float32)))
        fake=types.SimpleNamespace(bfloat16=np.dtype('float32'),int32=np.dtype('int32'))
        with patch.dict(sys.modules,{'torch':fake}):
            capture.validate_call(args)
            args['w1_zeros']=Tensor(np.empty(0,np.int32))
            with self.assertRaisesRegex(ValueError,'w1_zeros'):capture.validate_call(args)
            args['w1_zeros']=None;args['w1_scale']=Tensor(np.ones((4,4),np.float16))
            with self.assertRaisesRegex(ValueError,'BF16 scales'):capture.validate_call(args)

    def test_hardware_and_native_source_receipts_use_the_audited_output_root(self):
        import contextlib,io,shutil,tempfile
        nb=build_notebook()
        hardware=''.join(nb['cells'][1]['source']);native=''.join(nb['cells'][5]['source'])
        self.assertIn("hardware_dir = Path('/kaggle/working/marlin-replay')",hardware)
        self.assertIn('/kaggle/working/marlin-replay/runtime-source',native)
        self.assertNotIn('/kaggle/working/headroom',''.join(''.join(c['source']) for c in nb['cells']))
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'output';out.mkdir()
            fixed=hardware.replace('/kaggle/working/marlin-replay',str(out))
            inventory=types.SimpleNamespace(returncode=0,stdout='NVIDIA RTX PRO 6000 Blackwell Workstation Edition, 97887, 580.00\n',stderr='')
            with patch('subprocess.run',return_value=inventory),contextlib.redirect_stdout(io.StringIO()):exec(fixed,{})
            self.assertTrue(json.loads((out/'hardware.json').read_text())['valid'])
            package=Path(tmp)/'sglang';shutil.copytree(HERE.parent/'serving-headroom-v3-20261008/native-source',package)
            model=Path(tmp)/'model';model.mkdir();(model/'config.json').write_text('{"cpu_fixture":true}')
            sg=types.ModuleType('sglang');sg.__file__=str(package/'__init__.py');sg.__version__='exact-source-double'
            torch=types.ModuleType('torch');torch.__version__='CPU-test-double'
            def run(args,**kw):
                self.assertEqual(args[1],'-c')
                with patch.dict(sys.modules,{'sglang':sg,'torch':torch}),contextlib.redirect_stdout(io.StringIO()):exec(args[2],{})
            namespace={'run':run,'PYTHON':'cpu-fixture','env':{},'OUT':out,'MODEL_DIR':str(model),'Path':Path,'shutil':shutil}
            exec(native.replace('/kaggle/working/marlin-replay',str(out)),namespace)
            identity=json.loads((out/'runtime-source/identity.json').read_text())['files']
            contract=json.loads((HERE.parent/'serving-headroom-v3-20261008/native-contract.json').read_text())
            for name,digest in contract['files'].items():self.assertEqual(identity[name],digest)
            self.assertTrue((out/'target-config.json').exists())

    def test_patch_cell_runs_in_a_real_subprocess_and_produces_manifest_bytes(self):
        # v2 died here: the payload went through argv and hit Linux's per-argument limit (E2BIG).
        import contextlib,hashlib,io,os,subprocess,tempfile
        nb=build_notebook();cell=''.join(nb['cells'][6]['source'])
        manifest=json.loads((HERE/'patch-manifest.json').read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp=Path(tmp);site=tmp/'site';(site/'sglang').mkdir(parents=True);(site/'sglang/__init__.py').write_text('')
            for name in manifest:
                target=site/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((HERE/'source'/name).read_bytes())
            working=tmp/'working';out=working/'marlin-replay';out.mkdir(parents=True)
            def run(cmd,check=True,env=None):
                cmd=list(map(str,cmd));result=subprocess.run(cmd,text=True,capture_output=True,env=env)
                if check and result.returncode:raise RuntimeError(result.stderr[-2000:])
                return result
            env={**os.environ,'PYTHONPATH':str(site)}
            namespace={'run':run,'PYTHON':sys.executable,'env':env,'OUT':out,'WORKING_DIR':working,'json':json}
            with contextlib.redirect_stdout(io.StringIO()):exec(cell,namespace)
            for name,digest in manifest.items():
                self.assertEqual(hashlib.sha256((site/name).read_bytes()).hexdigest(),digest['after'],name)
            self.assertEqual(namespace['env']['PYTHONPATH'],str(working/'marlin-kit'))

    def test_prepared_notebook_scope(self):
        nb=build_notebook();driver=''.join(nb['cells'][7]['source']);worker=''.join(nb['cells'][8]['source'])
        self.assertIn('PLAN = [("PROFILE10", 10)]',driver)
        self.assertIn('ARM-CAPTURE',driver);self.assertIn('< 1260',driver)
        self.assertIn('timeout=600',worker);self.assertIn('finally:',worker)
        self.assertIn('sweep.stop_process_group(proc, grace=30)',driver)
        meta=json.loads((HERE/'kernel/kernel-metadata.json').read_text())
        self.assertTrue(meta['is_private']);self.assertFalse(meta['enable_internet'])
        self.assertEqual(meta['machine_shape'],'NvidiaRtxPro6000')
        for cell in nb['cells']:
            if cell['cell_type']=='code':compile(''.join(cell['source']),'<prepared-notebook>','exec')


if __name__=='__main__':unittest.main()
