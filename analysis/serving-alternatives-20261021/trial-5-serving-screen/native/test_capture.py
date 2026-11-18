import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
import capture_runtime as cr

class CaptureTests(unittest.TestCase):
    def test_first_committed_target_and_finish_are_once_per_request(self):
        with tempfile.TemporaryDirectory() as tmp:
            log = cr.Capture(Path(tmp)/'native.jsonl', 'test-session', 'GPU-fixture', heartbeat=False)
            log.accept('r1', [11,12])
            log.observe('r1', 0, False, 'running')
            log.observe('r1', 4, False, 'running')
            log.observe('r1', 7, True, 'length')
            log.observe('r1', 7, True, 'length')
            log.close()
            rows = [json.loads(line) for line in (Path(tmp)/'native.jsonl').read_text().splitlines()]
            self.assertEqual([r['kind'] for r in rows], ['capture_start','native_accept','target_first','native_finish','capture_end'])
            self.assertEqual(rows[-2]['output_tokens'],7)
            self.assertEqual(rows[1]['prompt_tokens'],2)
            self.assertEqual([r['seq'] for r in rows], list(range(5)))


class CaptureFixture(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.path=Path(self.tmp.name)/'native.jsonl'
        self.log=cr.Capture(self.path,'test-session','GPU-fixture',heartbeat=False)
        self.old=cr._CAPTURE;cr._CAPTURE=self.log
    def tearDown(self):
        self.log.close();cr._CAPTURE=self.old;self.tmp.cleanup()
    def rows(self): return [json.loads(x) for x in self.path.read_text().splitlines()]
class DefensiveCaptureTests(CaptureFixture):
    def test_duplicate_accept_is_fatal(self):
        self.log.accept('r',[1])
        with self.assertRaises(ValueError): self.log.accept('r',[1])
    def test_decreasing_count_is_fatal(self):
        self.log.accept('r',[1]);self.log.observe('r',4,False,'running')
        with self.assertRaises(ValueError): self.log.observe('r',3,False,'running')
    def test_unknown_request_is_fatal(self):
        with self.assertRaises(KeyError): self.log.observe('unknown',1,False,'running')
    def test_no_first_output_for_waiting_abort(self):
        req=NS(request_id='r',prompt_token_ids=[1],output_token_ids=[],status=NS(name='FINISHED_ABORTED'))
        cr.vl_accept(req);cr.vl_finish(req)
        self.assertEqual([r['kind'] for r in self.rows()],['capture_start','native_accept','abort','native_finish'])
    def test_sglang_stopping_inside_speculative_block_uses_visible_length(self):
        req=NS(rid='r',origin_input_ids=[1],output_ids=[5,6,7,8],finished_len=2,finished_reason=NS())
        req.finished=lambda:True
        cr.sg_queue(req,False);cr.sg_output(req);cr.sg_output(req)
        self.assertEqual([r['output_tokens'] for r in self.rows() if r['kind']=='native_finish'],[2])
        self.assertEqual(sum(r['kind']=='target_first' for r in self.rows()),1)
    def test_retraction_and_abort_signals_are_preserved(self):
        req=NS(rid='r',origin_input_ids=[1]);cr.sg_queue(req,False);cr.sg_queue(req,True)
        cr.sg_abort(NS(rid='r',abort_all=False));cr.vl_preempt(NS(request_id='r'))
        self.assertEqual([r['kind'] for r in self.rows()][-3:],['retract','abort_signal','preempt'])

import ast
import overlay

def real_method(source, name, extra=None):
    node=next(n for n in ast.walk(ast.parse(source)) if isinstance(n,ast.FunctionDef) and n.name==name)
    node.decorator_list=[]
    namespace=dict(_capacity_trace=cr, **(extra or {}))
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),node],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<pinned-real-method>','exec'),namespace)
    return namespace[name]

class PinnedMethodTests(CaptureFixture):
    # Full upstream method bodies run without importing/installing the providers.
    def test_vllm_real_commit_method_trims_fast_mtp_before_trace(self):
        src=overlay.patch('vllm/v1/core/sched/scheduler.py',overlay.SOURCES['vllm/v1/core/sched/scheduler.py'][0].read_text())
        request_src=(overlay.VL/'vllm-request.raw.py').read_text()
        append=real_method(request_src,'append_output_token_ids')
        req=NS(request_id='v',prompt_token_ids=[10,11],_output_token_ids=[],_all_token_ids=[10,11],update_block_hashes=lambda:None)
        req.output_token_ids=req._output_token_ids
        req.append_output_token_ids=lambda token:append(req,token)
        cr.vl_accept(req)
        def check_stop(request,max_model_len):
            return len(request.output_token_ids)==2
        method=real_method(src,'_update_request_with_output',{'check_stop':check_stop})
        accepted,stopped=method(NS(max_model_len=100),req,[31,32,33,34])
        self.assertEqual(accepted,[31,32]);self.assertTrue(stopped)
        self.assertEqual(req.output_token_ids,[31,32])
        self.assertEqual([(r['kind'],r.get('output_tokens')) for r in self.rows()][-1],('target_first',2))
        req.status=NS(name='FINISHED_LENGTH_CAPPED');cr.vl_finish(req)
        self.assertEqual(self.rows()[-1]['output_tokens'],2)
    def test_sglang_full_decode_method_commits_before_finish_and_deduplicates_overlap(self):
        src=overlay.patch('sglang/srt/managers/scheduler_components/batch_result_processor.py',overlay.SOURCES['sglang/srt/managers/scheduler_components/batch_result_processor.py'][0].read_text())
        req=NS(rid='s',origin_input_ids=[1,2],output_ids=[],finished_len=None,is_retracted=False,
               return_logprob=False,return_sampling_mask=False,return_hidden_states=False,grammar=None,
               finished_reason=None,time_stats=NS(set_last_decode_finish_time=lambda:None))
        req.finished=lambda:req.finished_reason is not None
        def update(n):
            req.finished_len=2;req.finished_reason=type('FINISH_LENGTH',(),{})()
        req.update_finish_state=update
        cr.sg_queue(req,False)
        allocator=NS(free_group_begin=lambda:None,free_group_end=lambda:None)
        reporter=NS(num_generated_tokens=0,forward_ct_decode=0,update_spec_metrics=lambda *a,**k:None,report_decode_stats=lambda *a,**k:None)
        instance=NS(enable_overlap=True,enable_overlap_mlx=False,token_to_kv_pool_allocator=allocator,metrics_reporter=reporter,
                    snapshot_auxiliary_output_starts=lambda *a:[],_normalize_decode_outputs=lambda **k:(k['next_token_ids'],None),
                    _maybe_update_reasoning_tokens=lambda *a:None,output_streamer=NS(stream_output=lambda *a:None))
        # This boundary assertion is independently tied to the upstream call order:
        # capture must already contain finished target output before release-side handling.
        def finish_side(*a):
            self.assertEqual(self.rows()[-1]['kind'],'native_finish')
            self.assertEqual(self.rows()[-1]['output_tokens'],2)
        instance._handle_finish_state_updated_req=finish_side
        batch=NS(reqs=[req],spec_algorithm=NS(is_none=lambda:False),batch_size=lambda:1,return_logprob=False)
        result=NS(copy_done=None,auxiliary_host_output=None,routed_experts_output=None,indexer_topk_output=None,
                  logits_output=NS(hidden_states=None),next_token_ids=[[21,22,23,24]],can_run_cuda_graph=True,
                  num_correct_drafts=3,num_block_accept_tokens=None,num_cap_tokens=None)
        method=real_method(src,'process_batch_result_decode',{'get_observability':lambda:NS(enable_metrics=False)})
        method(instance,batch,result);method(instance,batch,result)
        self.assertEqual(req.output_ids,[21,22,23,24])
        self.assertEqual(sum(r['kind']=='target_first' for r in self.rows()),1)
        self.assertEqual(sum(r['kind']=='native_finish' for r in self.rows()),1)
    def test_source_drift_and_wrong_anchor_fail_closed(self):
        with self.assertRaises(ValueError): overlay.replace_once('changed','original','new')
    def test_all_source_overlays_compile_with_exact_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest=overlay.build(Path(tmp)/'overlay')
            self.assertEqual(len(manifest),6)
            for path in overlay.SOURCES:
                compile((Path(tmp)/'overlay'/path).read_text(),path,'exec')

if __name__ == "__main__": unittest.main()
