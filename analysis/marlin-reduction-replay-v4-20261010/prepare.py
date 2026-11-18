"""Build a hash-bound optional patch from the exact qualified wheel, locally."""
import difflib
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FILES = {
    'fused': 'sglang/srt/layers/moe/fused_moe_triton/fused_marlin_moe.py',
    'runner': 'sglang/srt/layers/moe/moe_runner/marlin.py',
    'gptq': 'sglang/srt/hardware_backend/gpu/quantization/gptq_kernels.py',
    'model': 'sglang/srt/model_executor/model_runner.py',
    'graph': 'sglang/srt/model_executor/runner/decode_cuda_graph_runner.py',
}


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Expected exactly one source anchor: {old[:80]!r}')
    return source.replace(old, new, 1)


def revised_sources(originals):
    out = dict(originals)
    out['fused'] = replace_once(out['fused'], '    is_gated: bool = True,\n',
        '    is_gated: bool = True,\n    use_atomic_add_override: Optional[bool] = None,\n    experiment_capture_tag: Optional[str] = None,\n')
    out['fused'] = replace_once(out['fused'], '    M, K = hidden_states.shape\n',
        '    if experiment_capture_tag is not None:\n        from marlin_capture import snapshot\n        snapshot(experiment_capture_tag, locals())\n\n    M, K = hidden_states.shape\n')
    out['fused'] = replace_once(out['fused'], '    intermediate_cache1 = moe_wna16_marlin_gemm(\n',
        '    if use_atomic_add_override is not None:\n        use_atomic_add = use_atomic_add_override\n\n    intermediate_cache1 = moe_wna16_marlin_gemm(\n')
    out['runner'] = replace_once(out['runner'], '    weight_bits: int\n',
        '    weight_bits: int\n    use_atomic_add_override: Optional[bool] = None\n    experiment_capture_tag: Optional[str] = None\n')
    out['runner'] = replace_once(out['runner'], '        is_gated=runner_config.is_gated,\n',
        '        is_gated=runner_config.is_gated,\n        use_atomic_add_override=quant_info.use_atomic_add_override,\n        experiment_capture_tag=quant_info.experiment_capture_tag,\n')
    out['gptq'] = replace_once(out['gptq'], '            weight_bits=self.quant_config.weight_bits,\n',
        '            weight_bits=self.quant_config.weight_bits,\n            use_atomic_add_override=getattr(self, "experiment_atomic_override", None),\n            experiment_capture_tag=getattr(self, "experiment_capture_tag", None),\n')
    out['model'] = replace_once(out['model'], '        self.load_model()\n',
        '        self.load_model()\n        from marlin_capture import bind_model\n        bind_model(self)\n')
    out['graph'] = replace_once(out['graph'], '                out = forward(\n',
        '                from marlin_capture import begin_capture\n                begin_capture(self, shape_key)\n                out = forward(\n')
    out['graph'] = replace_once(out['graph'], '                return out\n',
        '                from marlin_capture import end_capture\n                end_capture()\n                return out\n')
    out['graph'] = replace_once(out['graph'], '            output = self.backend.replay(self._replay_graph_key, forward_batch)\n',
        '            output = self.backend.replay(self._replay_graph_key, forward_batch)\n            from marlin_capture import after_execute\n            after_execute(self, forward_batch)\n')
    return out


def build():
    identity = json.loads((HERE / 'source-identity.json').read_text())
    originals = {key: (HERE / 'source' / name).read_text() for key, name in FILES.items()}
    for key, name in FILES.items():
        assert hashlib.sha256(originals[key].encode()).hexdigest() == identity['members'][name]
    revised = revised_sources(originals)
    manifest, diff = {}, []
    for key, name in FILES.items():
        path = HERE / 'patched' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(revised[key])
        compile(revised[key], name, 'exec')
        manifest[name] = {'before': identity['members'][name], 'after': hashlib.sha256(revised[key].encode()).hexdigest()}
        diff.extend(difflib.unified_diff(originals[key].splitlines(True), revised[key].splitlines(True), 'a/'+name, 'b/'+name, n=0))
    (HERE / 'candidate.patch').write_text(''.join(diff))
    (HERE / 'patch-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return manifest


if __name__ == '__main__':
    print(json.dumps(build(), indent=2))
