# SPEED SCREEN: stacked serving configs on P, one server restart per config. Runs after the ARM P guard cell.
# Inputs defined by earlier notebook cells: WORKING_DIR, BUNDLE_DIR, PUBLIC25_VLLM_PROFILE_ENV, _command_env.
import importlib.util
import json
import os
import re
import shutil
import sys
import time
import traceback

BENCH_DIR = WORKING_DIR / "bench"
sys.path.insert(0, str(BENCH_DIR))
import speedbench as sb  # noqa: E402  (written by the previous cell)

WARMUP_S, MEASURE_S, WINDOWS = 180.0, 600.0, 5
DATA = json.loads((BENCH_DIR / "sessions.json").read_text())
SESSIONS, TOOLS = DATA["sessions"], sb.python_tool(DATA["tool_description"])
TUNING_KEYS = ("TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION", "TAAF_VLLM_ENABLE_PREFIX_CACHING", "TAAF_VLLM_MTP_TOKENS",
               "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE")

os.environ.update(_command_env())                         # TAAF_KAGGLE_* for the serving setup module
_spec = importlib.util.spec_from_file_location("taaf_vllm_serving_setup", BUNDLE_DIR / "serving_setup.py")
setup = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(setup)
MODEL_DIR = setup.resolve_model_dir()


def log_evidence(text: str) -> dict:
    args = re.findall(r"non-default args: (\{.*)", text)
    return {
        "non_default_args": args[-1][:4000] if args else None,
        "prune_load_weights": re.findall(r"FLASHNEXT_PRUNE pid=\d+ load_weights totals (\{[^}]*\}) blocks=(\d+) keep=(\d+)", text),
        "model_gib": [float(x) for x in re.findall(r"Model loading took ([\d.]+) GiB", text)],
        "kv_tokens": re.findall(r"GPU KV cache size: ([\d,]+) tokens", text),
        "max_concurrency": re.findall(r"Maximum concurrency for [\d,]+ tokens per request: ([\d.]+)x", text),
        "mamba_cache_mode": re.findall(r"[Mm]amba cache mode is set to '(\w+)'", text)
                            + re.findall(r"Prefix caching in Mamba cache '(\w+)'", text),
        "index_share_in_args": bool(re.search(r"index_share_for_mtp_iteration['\"]?\s*:\s*[Tt]rue", text)),
        "prefix_caching_in_args": bool(re.search(r"enable_prefix_caching['\"]?\s*:\s*[Tt]rue", text)),
        "num_speculative_tokens": re.findall(r"num_speculative_tokens['\"]?\s*:\s*(\d+)", text)[-1:],
        "block_size": re.findall(r"[Ss]etting attention block size to (\d+)", text)[-1:],
        "error_lines": [l[:300] for l in text.splitlines() if re.search(r"Traceback|\bERROR\b|NotImplementedError", l)][:20],
    }


def set_profile(overrides: dict) -> None:
    for key in TUNING_KEYS:
        if key in PUBLIC25_VLLM_PROFILE_ENV:
            os.environ[key] = PUBLIC25_VLLM_PROFILE_ENV[key]
        else:
            os.environ.pop(key, None)
    os.environ.update(overrides)


def restart_server(overrides: dict) -> dict:
    set_profile(overrides)
    tuning = setup.resolve_vllm_tuning()
    env, _check = setup.runtime_environment(deep_preload_validation=False, tuning=tuning)
    for attempt in range(3):
        try:
            identity = setup.start_server(MODEL_DIR, env, tuning=tuning)    # stops the owned server first
            break
        except RuntimeError as exc:
            # start_server can read /proc/<pid>/cmdline before exec (fork/setsid race); it terminates that child.
            if "argv did not match" not in str(exc) or attempt == 2:
                raise
            print(f"SPEED_SCREEN argv race on start, retrying: {exc}", flush=True)
            time.sleep(5)
    readiness = setup.wait_for_server(identity, time.monotonic() + setup.SERVER_READY_TIMEOUT + 60)
    return {"tuning": tuning, "ready_seconds": readiness["ready_seconds"], "pid": identity.get("pid")}


def run_config(name: str, overrides: dict, *, restart: bool) -> dict:
    ev = {"name": name, "overrides": overrides, "started": time.time(), "ok": False}
    print(f"SPEED_SCREEN {name} start overrides={overrides}", flush=True)
    try:
        if restart:
            ev["restart"] = restart_server(overrides)
        ev["ready_wall_s"] = time.time() - ev["started"]
        log = setup.SERVER_LOG.read_text(errors="replace")
        ev["log"] = log_evidence(log)
        lw, mem = ev["log"]["prune_load_weights"], ev["log"]["model_gib"]
        assert lw and all(b == "48" and k == "448" for _, b, k in lw), "pruning plugin did not run in the worker"
        assert mem and max(mem) < 76.0, f"model memory {mem} GiB: pruning did not free VRAM"
        ev["logprobs"] = sb.logprob_check(SESSIONS)
        load = sb.run_load(SESSIONS, TOOLS, warmup_s=WARMUP_S, measure_s=MEASURE_S, windows=WINDOWS)
        (BENCH_DIR / f"raw-{name}.json").write_text(json.dumps({"records": load.pop("records"),
                                                                "samples": load.pop("samples")}))
        ev["load"] = load
        ev["ok"] = True
        t = load["total"]
        print(f"SPEED_SCREEN {name} done " + json.dumps({key: t.get(key) for key in (
            "gen_tok_s", "requests_s", "running_mean", "kv_usage_mean", "prefix_hit_rate", "accept_rate_by_pos",
            "prompt_tokens_mean", "gpu_mib_max", "error")} | {"errors": load["errors_after_warmup"],
                                                             "req_per_turn": load["requests_per_turn"]},
            default=str), flush=True)
    except BaseException as exc:                               # record and move on to the next config
        ev["error"] = f"{type(exc).__name__}: {exc}"[:4000]
        ev["traceback"] = traceback.format_exc()[-6000:]
        print(f"SPEED_SCREEN {name} FAILED {ev['error'][:800]}", flush=True)
    finally:
        if setup.SERVER_LOG.exists():
            shutil.copy(setup.SERVER_LOG, BENCH_DIR / f"server-{name}.log")
        ev["ended"] = time.time()
        RESULTS["configs"][name] = ev
        (WORKING_DIR / "speed-screen.json").write_text(json.dumps(RESULTS, indent=1, default=str))
    return ev


RESULTS = {"plan": "analysis/kv-lever-scope-20260930/NEXT-PLAN.md", "warmup_s": WARMUP_S, "measure_s": MEASURE_S,
           "windows": WINDOWS, "turn_budget": sb.TURN_BUDGET, "max_turn_requests": sb.MAX_TURN_REQUESTS,
           "profile_P": PUBLIC25_VLLM_PROFILE_ENV, "configs": {}, "decisions": {}}

S1 = {"TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION": "1"}
S2 = {**S1, "TAAF_VLLM_ENABLE_PREFIX_CACHING": "1"}
run_config("S0", {}, restart=False)
run_config("S1", S1, restart=True)
run_config("S2", S2, restart=True)

# S3 = stack base + MTP length. Base is S2 if it ran and its prefix-cached next-token distributions agree with S1
# within the noise floor (S1's own repeat and S0-vs-S1, neither cached); else S1 if it ran; else S0.
c = RESULTS["configs"]
lp = lambda n: c[n].get("logprobs") or []
floors = [x for x in (sb.compare_logprobs(lp("S1"), lp("S1"), 0, 1), sb.compare_logprobs(lp("S0"), lp("S1"), 0, 0)) if x]
s2_cached = sb.compare_logprobs(lp("S1"), lp("S2"), 0, 1) if c["S2"]["ok"] and c["S1"]["ok"] else None
floor_m = max((x["top1_mismatch"] for x in floors), default=0)
floor_abs = max((x["mean_abs"] for x in floors), default=0.0)
s2_equal = bool(floors and s2_cached and s2_cached["top1_mismatch"] <= floor_m + 1
                and s2_cached["mean_abs"] <= 3 * floor_abs + 0.02)
base_name = "S2" if (c["S2"]["ok"] and s2_equal) else ("S1" if c["S1"]["ok"] else "S0")
base = {"S0": {}, "S1": S1, "S2": S2}[base_name]
acc = (c[base_name].get("load") or {}).get("total", {}).get("accept_rate_by_pos", {})
p2 = acc.get("accepted_pos2")
k = 2 if (p2 is not None and p2 < 0.35) else 4
S3 = {**base, "TAAF_VLLM_MTP_TOKENS": str(k)}
if k == 4:
    S3["TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"] = "60"                 # 12 seqs x (1 + 4) tokens
RESULTS["decisions"] = {"logprob_floors": floors, "logprob_S2cached_vs_S1": s2_cached, "s2_equal": s2_equal,
                        "s3_base": base_name, "base_pos2_acceptance": p2, "s3_mtp_tokens": k}
print("SPEED_SCREEN decisions", json.dumps(RESULTS["decisions"]), flush=True)
run_config("S3", S3, restart=True)

try:
    RESULTS["teardown"] = setup.stop_owned_server()
except BaseException as exc:
    RESULTS["teardown"] = {"error": f"{type(exc).__name__}: {exc}"}
(WORKING_DIR / "speed-screen.json").write_text(json.dumps(RESULTS, indent=1, default=str))
print("SPEED_SCREEN complete", {n: (e["ok"], e.get("error", "")[:120]) for n, e in RESULTS["configs"].items()}, flush=True)
