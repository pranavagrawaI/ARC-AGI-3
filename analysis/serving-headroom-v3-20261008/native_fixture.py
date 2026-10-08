"""Execute hash-bound pinned methods, replacing GPU/framework boundaries only.

This qualifies counter/admission/profiler control flow on CPU, not CUDA execution.
"""
import ast
import dataclasses
import gc
import gzip
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from types import SimpleNamespace as NS

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'native-source'


def methods(relative, cls, names, namespace):
    path = SOURCE / relative
    manifest = json.loads((HERE / 'native-contract.json').read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['files'][relative]
    tree = ast.parse(path.read_text())
    node = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == cls)
    found = [x for x in node.body if isinstance(x, ast.FunctionDef) and x.name in names]
    assert len(found) == len(names)
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    module = ast.fix_missing_locations(ast.Module(body=[future, *found], type_ignores=[]))
    exec(compile(module, str(path), 'exec'), namespace)
    return type(cls, (), {name: namespace[name] for name in names})


class Counter:
    def __init__(self):
        self.values = {}
    def labels(self, **labels):
        mode = labels['mode']
        return NS(inc=lambda delta: self.values.__setitem__(mode, self.values.get(mode, 0) + delta))
    def scrape(self):
        return '\n'.join(f'sglang:realtime_tokens_total{{mode="{mode}"}} {value}' for mode, value in self.values.items())


def decoder(n):
    collector_cls = methods('srt/observability/metrics_collector.py', 'SchedulerMetricsCollector',
                            ['increment_realtime_tokens'], {})
    collector = collector_cls()
    collector.realtime_tokens_total = Counter()
    collector.labels = {}
    reporter_cls = methods('srt/managers/scheduler_components/metrics_reporter.py', 'SchedulerMetricsReporter',
                          ['report_decode_stats'], {})
    reporter = reporter_cls()
    reporter.current_scheduler_metrics_enabled = True
    reporter.metrics_collector = collector
    reporter.enable_mfu_metrics = False
    reporter.scheduler_status_logger = None
    reporter.forward_ct_decode = 1
    reporter.decode_log_interval = 40  # Exercise the real every-iteration path, then its periodic early return.
    reporter.scheduler = NS(running_batch=NS(batch_size=lambda: n, dp_cooperation_info=None))
    return reporter, collector


def admission(slots):
    schedule = NS(max_running_requests=16, max_mamba_cache_size=slots, disable_overlap_schedule=False)
    namespace = dict(get_schedule=lambda: schedule, get_memory=lambda: NS(disable_radix_cache=False),
                     envs=NS(SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=NS(get=lambda: False)),
                     mamba_extra_buffer_enabled=lambda: True, mamba_extra_buffer_lazy_enabled=lambda: False,
                     logger=logging.getLogger('native-admission'),
                     MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO=3, MAMBA_CACHE_BASE_RATIO_DROP_ON_SKIP=1,
                     MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP=2,
                     MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP_LAZY=1,
                     MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP=1,
                     MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_BUFFER=1)
    cls = methods('srt/mem_cache/kv_cache_configurator.py', 'KVCacheConfigurator',
                  ['_calculate_mamba_ratio', 'resolve_max_num_reqs'], namespace)
    obj = cls()
    obj.ps = NS(attn_dp_size=1)
    obj.model_config = NS(context_len=139264)
    obj.mambaish_config = object()
    return obj._calculate_mamba_ratio(), obj.resolve_max_num_reqs(1000000)


class FakeTorchProfile:
    def __init__(self, **kwargs):
        self.started = self.stopped = False
    def start(self):
        self.started = True
    def stop(self):
        self.stopped = True
    def export_chrome_trace(self, path):
        assert self.started and self.stopped
        with gzip.open(path, 'wt') as stream:
            json.dump({'traceEvents': [{'cat': 'kernel', 'ph': 'X', 'name': 'fixture_only', 'dur': 1, 'ts': 0}]}, stream)


def profiler():
    namespace = dict(Path=Path, os=os, time=time, gc=gc,
                     logger=logging.getLogger('native-profile'), _is_npu=False,
                     envs=NS(SGLANG_PROFILE_V2=NS(get=lambda: False)),
                     current_platform=NS(is_out_of_tree=lambda: False),
                     torch=NS(profiler=NS(ProfilerActivity=NS(CPU='CPU', CUDA='CUDA'), profile=FakeTorchProfile),
                              distributed=NS(barrier=lambda group: None)),
                     ProfileReqOutput=lambda **kw: NS(**kw),
                     ProfileReqType=NS(START_PROFILE='start'), set_detailed_annotations_enabled=lambda flag: None)
    names = ['__post_init__', '_init_profile', '_start_profile', '_stop_profile', '_profile',
             '_profile_batch_predicate', '_apply_detailed_annotations', '_merge_profile_traces']
    cls = methods('srt/managers/scheduler_components/profiler_manager.py', 'SchedulerProfilerManager', names, namespace)
    manager = cls()
    manager.ps = NS(tp_rank=0, dp_size=1, pp_size=1, moe_ep_size=1)
    manager.dp_tp_cpu_group = None
    manager.forward = 0
    manager.get_forward_ct = lambda: manager.forward
    manager.__post_init__()
    return manager


def start_profile(manager, body):
    req = NS(**dict(body, req_type='start', start_step=None, merge_profiles=False,
                    profile_prefix='', profile_stages=None))
    ret = manager._profile(req)
    if not ret.success:
        raise RuntimeError(ret.message)
    start = manager.forward
    for step in range(start + 1, start + body['num_steps'] + 2):
        manager.forward = step
        manager._profile_batch_predicate(NS())
    assert manager.profile_in_progress is False
    return b'Start profiling.\n'
