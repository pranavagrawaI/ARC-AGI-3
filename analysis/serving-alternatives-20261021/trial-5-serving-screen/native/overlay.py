"""Produce source-bound overlays; never mutate an installed engine in place."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SG = ROOT/'analysis/serving-headroom-v2-20261008/native-source'
VL = ROOT/'analysis/serving-alternatives-20261021/trial-3-matched-preparation/source-evidence'
SOURCES = {
 'sglang/srt/managers/scheduler.py':(SG/'srt/managers/scheduler.py','a1209592d8146328c11f64e338d3448438428260ab9c45e1702347663d917ffd'),
 'sglang/srt/managers/scheduler_components/batch_result_processor.py':(SG/'srt/managers/scheduler_components/batch_result_processor.py','8c9626605546de2bc54d5f25a7ae16caeb9ec96e273af50adf91076df543e05b'),
 'vllm/v1/core/sched/scheduler.py':(VL/'vllm-scheduler.raw.py','b956bc2d88f803edf0ba87918cafdaeb6ddb72cd5b85d1872c88fef6aa6bdfb3'),
 'vllm/v1/core/sched/async_scheduler.py':(VL/'vllm-async-scheduler.raw.py','e586a0ef3c6778be56a93e7f9bb712d4de9de6e7d7fe7e3e1d51dae83ecfc508'),
 'vllm/v1/request.py':(VL/'vllm-request.raw.py','dd7c30bf8b0894dc123a739c299c0133566f8cc075b081e45905f1c29506e69d'),
}

def replace_once(source, old, new, expected=1):
    if source.count(old) != expected:
        raise ValueError(f'patch anchor count {source.count(old)} != {expected}: {old[:70]}')
    return source.replace(old,new)

def startup(source, klass):
    tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==klass)
    init=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    first=init.body[0]
    line=first.end_lineno if isinstance(first,ast.Expr) and isinstance(first.value,ast.Constant) and isinstance(first.value.value,str) else first.lineno-1
    lines=source.splitlines(keepends=True)
    lines.insert(line,' '*first.col_offset+'_capacity_trace.capture()\n')
    return ''.join(lines)

def patch(path, source):
    if path.endswith(('/async_scheduler.py','/request.py')):
        return source
    # Insert after future imports, which must remain at the start.
    lines=source.splitlines(keepends=True)
    tree=ast.parse(source)
    future=[n.end_lineno for n in tree.body if isinstance(n,ast.ImportFrom) and n.module=='__future__']
    line=max(future) if future else next(n.lineno-1 for n in tree.body if isinstance(n,(ast.Import,ast.ImportFrom)))
    lines.insert(line,'import capture_runtime as _capacity_trace\n')
    source=''.join(lines)
    if path.endswith('/batch_result_processor.py'):
        # Prefill and decode both update finish state after committing verified output.
        source=replace_once(source,'                    req.output_ids.append(next_token_id)\n\n                    self._maybe_update_reasoning_tokens(req, next_token_id)\n\n                    req.update_finish_state()\n','                    req.output_ids.append(next_token_id)\n\n                    self._maybe_update_reasoning_tokens(req, next_token_id)\n\n                    req.update_finish_state()\n                    _capacity_trace.sg_output(req)\n')
        source=replace_once(source,'            req.update_finish_state(new_accept_len)\n','            req.update_finish_state(new_accept_len)\n            _capacity_trace.sg_output(req)\n')
    elif path.startswith('sglang/'):
        source=startup(source,'Scheduler')
        source=replace_once(source,'    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):\n','    def _add_request_to_queue(self, req: Req, is_retracted: bool = False):\n        _capacity_trace.sg_queue(req, is_retracted)\n')
        source=replace_once(source,'    def abort_request(self, recv_req: AbortReq):\n','    def abort_request(self, recv_req: AbortReq):\n        _capacity_trace.sg_abort(recv_req)\n')
        source=replace_once(source,'    def pause_generation(self, recv_req: PauseGenerationReqInput):\n','    def pause_generation(self, recv_req: PauseGenerationReqInput):\n        _capacity_trace.capture().emit(\"pause_signal\", mode=recv_req.mode)\n')
    else:
        source=startup(source,'Scheduler')
        source=replace_once(source,'            self.requests[request.request_id] = request\n','            self.requests[request.request_id] = request\n            _capacity_trace.vl_accept(request)\n')
        source=replace_once(source,'        return new_token_ids, stopped\n','        _capacity_trace.vl_output(request)\n        return new_token_ids, stopped\n')
        anchor='        assert request.is_finished()\n\n        if self.aux_output_connector is not None:'
        source=replace_once(source,anchor,'        assert request.is_finished()\n        _capacity_trace.vl_finish(request)\n\n        if self.aux_output_connector is not None:')
        source=replace_once(source,'        if self.aux_output_connector is not None:\n            self.aux_output_connector.request_finished(request)\n        self._free_request_blocks(request)\n','        _capacity_trace.vl_preempt(request)\n        if self.aux_output_connector is not None:\n            self.aux_output_connector.request_finished(request)\n        self._free_request_blocks(request)\n')
    compile(source,path,'exec')
    return source

def build(destination):
    destination=Path(destination)
    destination.mkdir(parents=True,exist_ok=False)
    manifest={}
    for path,(original,expected) in SOURCES.items():
        data=original.read_bytes()
        if hashlib.sha256(data).hexdigest()!=expected:
            raise ValueError(f'source identity changed: {original}')
        result=patch(path,data.decode()).encode()
        out=destination/path;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(result)
        manifest[path]=dict(source_sha256=expected,overlay_sha256=hashlib.sha256(result).hexdigest(),source=str(original))
    shutil.copyfile(HERE/'capture_runtime.py',destination/'capture_runtime.py')
    manifest['capture_runtime.py']={'overlay_sha256':hashlib.sha256((destination/'capture_runtime.py').read_bytes()).hexdigest()}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('destination');args=parser.parse_args()
    print(json.dumps(build(args.destination),indent=2))
