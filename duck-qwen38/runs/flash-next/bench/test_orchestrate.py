"""python3 -B bench/test_orchestrate.py — run the built notebook's speed-screen cells against a stub setup + mock server.

Proves: the inputs cell round-trips the sessions, every config restarts with the intended tuning env, the prune
guard and log evidence parse, S3's base/k decision logic runs, and speed-screen.json is complete.
"""
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mock_server  # noqa: E402

PORT = 18235
STUB = textwrap.dedent(f'''
    import json, os, sys, time
    from pathlib import Path
    sys.path.insert(0, {str(HERE)!r})
    import mock_server
    WORKING_DIR = Path(os.environ["TAAF_KAGGLE_WORKING_DIR"])
    SERVER_LOG = WORKING_DIR / "vllm-openai-server.log"
    SERVER_READY_TIMEOUT = 5
    STARTS = []
    _srv = mock_server.__dict__.setdefault('SHARED', [None])
    def resolve_model_dir(): return Path("/model")
    def resolve_vllm_tuning():
        return {{k: os.environ.get(k) for k in ("TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION", "TAAF_VLLM_ENABLE_PREFIX_CACHING",
                                               "TAAF_VLLM_MTP_TOKENS", "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE")}}
    def runtime_environment(*, deep_preload_validation, tuning): return dict(os.environ), {{}}
    def stop_owned_server():
        if _srv[0]: _srv[0].shutdown(); _srv[0].server_close(); _srv[0] = None
        return {{"stopped": True}}
    def start_server(model_dir, env, *, tuning):
        stop_owned_server()
        STARTS.append(tuning)
        prefix = tuning["TAAF_VLLM_ENABLE_PREFIX_CACHING"] == "1"
        if prefix and os.environ.get("STUB_FAIL_PREFIX"):
            SERVER_LOG.write_text("NotImplementedError: QSA prefix caching\\n")
            raise RuntimeError("vLLM exited before readiness.")
        _srv[0], _ = mock_server.serve({PORT}, slots=4, latency=0.02, prefix=prefix,
                                       corrupt_prefix=bool(os.environ.get("STUB_CORRUPT_PREFIX")))
        share = tuning["TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION"] == "1"
        SERVER_LOG.write_text(
            "FLASHNEXT_PRUNE pid=1 load_weights totals {{'dropped': 1}} blocks=48 keep=448\\n"
            "Model loading took 73.9 GiB\\nGPU KV cache size: 252,000 tokens\\n"
            + "INFO non-default args: {{'enable_prefix_caching': " + str(prefix) + ", 'speculative_config': "
            + "{{'method': 'mtp', 'num_speculative_tokens': " + tuning["TAAF_VLLM_MTP_TOKENS"]
            + (", 'index_share_for_mtp_iteration': True" if share else "") + "}}}}\\n"
            + ("WARNING Mamba cache mode is set to 'align' for X by default\\n" if prefix else ""))
        return {{"pid": 2}}
    def wait_for_server(identity, deadline): return {{"ready_seconds": 0.1}}
''')


def main(fail_prefix: bool = False, corrupt_prefix: bool = False) -> None:
    os.environ.pop('STUB_FAIL_PREFIX', None)
    os.environ.pop('STUB_CORRUPT_PREFIX', None)
    if corrupt_prefix:
        os.environ['STUB_CORRUPT_PREFIX'] = '1'
    if corrupt_prefix:
        d = res["decisions"]
        assert d["s2_equal"] is False and d["s3_base"] == "S1" and d["logprob_S2cached_vs_S1"]["mean_abs"] >= 0.4, d
        from assess_screen import assess
        assert any("diverge" in f for f in assess(res)["rows"]["S2"]["integrity_failures"])
        print("PASS orchestrate/prefix-corrupt", json.dumps({k: d[k] for k in ("s2_equal", "s3_base")}))
        return
    if fail_prefix:
        os.environ['STUB_FAIL_PREFIX'] = '1'
    nb = json.loads((HERE / "kernel/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb").read_text())
    cells = ["".join(c["source"]) for c in nb["cells"][-2:]]
    assert "SPEED SCREEN inputs" in cells[0] and "SPEED SCREEN: stacked" in cells[1]
    orch = cells[1].replace("WARMUP_S, MEASURE_S, WINDOWS = 180.0, 600.0, 5", "WARMUP_S, MEASURE_S, WINDOWS = 8.0, 4.0, 2")
    orch2 = orch.replace("windows=WINDOWS)", "windows=WINDOWS, poll_s=0.3, stagger_s=0.5)")
    assert orch != cells[1] and orch2 != orch
    orch = orch2
    with tempfile.TemporaryDirectory() as tmp:
        work, bundle = Path(tmp) / "working", Path(tmp) / "bundle"
        work.mkdir(); bundle.mkdir()
        (bundle / "serving_setup.py").write_text(STUB)
        profile = {"TAAF_VLLM_MTP_TOKENS": "3", "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "48",
                   "TAAF_VLLM_ENABLE_PREFIX_CACHING": "0"}
        os.environ.update(profile)
        env_extra = {"TAAF_KAGGLE_WORKING_DIR": str(work), "TAAF_KAGGLE_BUNDLE_DIR": str(bundle)}
        ns = {"WORKING_DIR": work, "BUNDLE_DIR": bundle, "PUBLIC25_VLLM_PROFILE_ENV": profile,
              "_command_env": lambda: {**os.environ, **env_extra}}
        exec(cells[0], ns)
        # S0 is the server the notebook's setup cell started: stand it up with the P profile first
        os.environ.update(env_extra)
        import importlib.util
        spec = importlib.util.spec_from_file_location("s0", bundle / "serving_setup.py")
        s0 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s0)
        s0.start_server(None, None, tuning=s0.resolve_vllm_tuning())
        ns["__name__"] = "orchestrate_test"
        sys.modules.pop("speedbench", None)
        try:
            exec(compile(orch.replace('run_config("S0", {}, restart=False)',
                                      'import speedbench as _sb; _sb.BASE = "http://127.0.0.1:%d"\n'
                                      'run_config("S0", {}, restart=False)' % PORT), "orchestrate", "exec"), ns)
        finally:
            s0.stop_owned_server()
        res = json.loads((work / "speed-screen.json").read_text())
        starts = ns["setup"].STARTS
    if corrupt_prefix:
        d = res["decisions"]
        assert d["s2_equal"] is False and d["s3_base"] == "S1" and d["logprob_S2cached_vs_S1"]["mean_abs"] >= 0.4, d
        from assess_screen import assess
        assert any("diverge" in f for f in assess(res)["rows"]["S2"]["integrity_failures"])
        print("PASS orchestrate/prefix-corrupt", json.dumps({k: d[k] for k in ("s2_equal", "s3_base")}))
        return
    if fail_prefix:
        assert not res["configs"]["S2"]["ok"] and "exited before readiness" in res["configs"]["S2"]["error"]
        assert res["configs"]["S2"]["log"]["error_lines"] if "log" in res["configs"]["S2"] else True
        assert res["decisions"]["s3_base"] == "S1" and res["configs"]["S3"]["ok"], res["decisions"]
        assert starts[-1]["TAAF_VLLM_ENABLE_PREFIX_CACHING"] == "0" and starts[-1]["TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION"] == "1"
        assert (work_log := res["configs"]["S3"]["log"]) and not work_log["prefix_caching_in_args"]
        print("PASS orchestrate/prefix-fails", json.dumps(res["decisions"]))
        return
    for name in ("S0", "S1", "S2", "S3"):
        ev = res["configs"][name]
        assert ev["ok"], (name, ev.get("error"), ev.get("traceback"))
        assert ev["load"]["errors_after_warmup"] == 0 and len(ev["logprobs"]) == 32 and all("top" in r for r in ev["logprobs"])
        assert len(ev["load"]["windows"]) == 2 and all("error" not in w for w in ev["load"]["windows"])
    ev = {n: res["configs"][n]["log"] for n in res["configs"]}
    assert not ev["S0"]["index_share_in_args"] and ev["S1"]["index_share_in_args"] and ev["S2"]["prefix_caching_in_args"]
    assert ev["S2"]["mamba_cache_mode"] == ["align"] and not ev["S1"]["prefix_caching_in_args"]
    # the notebook's restarts (S0 was started by setup): S1 index share only; S2 adds prefix;
    # S3 = S2 + k=4 (mock acceptance pos2 = 0.45) + capture 60
    assert len(starts) == 3, starts
    assert [s["TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION"] for s in starts] == ["1", "1", "1"], starts
    assert [s["TAAF_VLLM_ENABLE_PREFIX_CACHING"] for s in starts] == ["0", "1", "1"], starts
    assert starts[2]["TAAF_VLLM_MTP_TOKENS"] == "4" and starts[2]["TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"] == "60"
    assert starts[0]["TAAF_VLLM_MTP_TOKENS"] == "3" and starts[0]["TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"] == "48"
    d = res["decisions"]
    assert d["s3_base"] == "S2" and d["s3_mtp_tokens"] == 4 and d["s2_equal"], d
    assert res["teardown"] == {"stopped": True}
    from assess_screen import assess
    rows = assess(res)["rows"]
    flag_fails = {n: [f for f in r["integrity_failures"] if "measurement windows" not in f] for n, r in rows.items()}
    assert not any(flag_fails.values()), flag_fails
    print("PASS orchestrate", json.dumps(d))


if __name__ == "__main__":
    main()
    main(fail_prefix=True)
    main(corrupt_prefix=True)
