"""python3 -B bench/test_assess_screen.py — the rule must pass a real gain and reject the ways a screen can mislead."""
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from assess_screen import assess  # noqa: E402

S1 = {"TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION": "1"}
S2 = {**S1, "TAAF_VLLM_ENABLE_PREFIX_CACHING": "1"}


def cfg(overrides, gens, reqs=None, *, share=None, prefix=None, k="3", errors=0):
    ch_share = overrides.get("TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION") == "1"
    ch_prefix = overrides.get("TAAF_VLLM_ENABLE_PREFIX_CACHING") == "1"
    reqs = reqs or [g / 1000 for g in gens]
    return {"ok": True, "overrides": overrides,
            "log": {"index_share_in_args": ch_share if share is None else share,
                    "prefix_caching_in_args": ch_prefix if prefix is None else prefix,
                    "num_speculative_tokens": [overrides.get("TAAF_VLLM_MTP_TOKENS", k)]},
            "load": {"errors_after_warmup": errors, "measure_start": 0, "ended": 600,
                     "total": {"prompt_tokens_mean": 20500}, "stats": {"tokenize_ok": 100},
                     "windows": [{"gen_tok_s": g, "requests_s": r, "client_completed": 100} for g, r in zip(gens, reqs)]}}


BASE = {"decisions": {"s2_equal": True, "logprob_S2cached_vs_S1": None, "logprob_floors": []},
        "configs": {"S0": cfg({}, [400, 410, 390, 405, 395]),
                    "S1": cfg(S1, [404, 412, 398, 401, 400]),
                    "S2": cfg(S2, [470, 480, 460, 475, 465]),
                    "S3": cfg({**S2, "TAAF_VLLM_MTP_TOKENS": "4"}, [478, 470, 482, 466, 474])}}


def test_real_gain_passes_and_prefers_fewer_changes():
    d = assess(BASE)["decision"]
    # S3 is within 3% of... S2 (subset) -> choose S2
    assert d["status"] == "PASS" and d["p_plus"] == "S2" and d["p_plus_overrides"] == S2, d


def test_flag_not_live_fails_integrity():
    r = copy.deepcopy(BASE)
    r["configs"]["S2"]["log"]["prefix_caching_in_args"] = False
    r["configs"]["S3"]["log"]["prefix_caching_in_args"] = False
    out = assess(r)
    assert out["rows"]["S2"]["integrity_failures"] and out["decision"]["status"] == "FAIL", out["decision"]


def test_logprob_divergence_blocks_prefix_configs():
    r = copy.deepcopy(BASE)
    r["decisions"]["s2_equal"] = False
    d = assess(r)["decision"]
    assert d["status"] == "FAIL" and d["best_by_gen"] in ("S0", "S1"), d


def test_small_gain_fails():
    r = copy.deepcopy(BASE)
    for name in ("S2", "S3"):
        r["configs"][name]["load"]["windows"] = [dict(w, gen_tok_s=w["gen_tok_s"] * 0.9) for w in r["configs"][name]["load"]["windows"]]
    d = assess(r)["decision"]
    assert d["status"] == "FAIL", d


def test_gen_up_but_requests_flat_fails():
    r = copy.deepcopy(BASE)
    for name in ("S2", "S3"):
        r["configs"][name]["load"]["windows"] = [dict(w, requests_s=0.4) for w in r["configs"][name]["load"]["windows"]]
    assert assess(r)["decision"]["status"] == "FAIL"


def test_errors_fail_integrity_and_s0_invalid_is_invalid():
    r = copy.deepcopy(BASE)
    r["configs"]["S2"]["load"]["errors_after_warmup"] = 50
    assert assess(r)["rows"]["S2"]["integrity_failures"]
    r["configs"]["S0"]["ok"] = False
    assert assess(r)["decision"]["status"] == "INVALID"


def test_crashed_config_is_skipped():
    r = copy.deepcopy(BASE)
    r["configs"]["S2"] = {"ok": False, "error": "RuntimeError: vLLM exited", "overrides": S2}
    r["configs"]["S3"] = cfg({**S1, "TAAF_VLLM_MTP_TOKENS": "4"}, [460, 470, 455, 465, 462])
    d = assess(r)["decision"]
    assert d["status"] == "PASS" and d["p_plus"] == "S3", d


def test_short_prompts_or_tokenize_fallback_invalidate():
    r = copy.deepcopy(BASE)
    r["configs"]["S1"]["load"]["total"]["prompt_tokens_mean"] = 6238      # v1's collapse
    assert assess(r)["decision"]["status"] == "INVALID"
    r = copy.deepcopy(BASE)
    r["configs"]["S2"]["load"]["stats"] = {"tokenize_ok": 90, "tokenize_fallback": 10}
    assert assess(r)["decision"]["status"] == "INVALID"


def test_v1_output_is_invalid():
    import json
    v1 = Path(__file__).resolve().parent / "out/speed-screen.json"
    if v1.exists():
        assert assess(json.loads(v1.read_text()))["decision"]["status"] == "INVALID"


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
