"""python3 -B bench/test_speedbench.py — exercises speedbench against mock_server with the real sessions."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mock_server  # noqa: E402
import speedbench as sb  # noqa: E402

PORT = 18234
sb.BASE = f"http://127.0.0.1:{PORT}"
data = json.loads((HERE / "sessions.json").read_text())
sessions = data["sessions"]
tools = sb.python_tool(data["tool_description"])


def test_drop_oldest_block_keeps_user_first():
    hist = [{"role": "user"}, {"role": "assistant"}, {"role": "tool"}, {"role": "user"}, {"role": "assistant"}]
    assert sb.drop_oldest_block(hist) and hist[0]["role"] == "user" and len(hist) == 2
    assert not sb.drop_oldest_block(hist) and len(hist) == 2


def test_parse_metrics_sums_labels_and_positions():
    text = ('vllm:request_success_total{finished_reason="stop"} 5.0\n'
            'vllm:request_success_total{finished_reason="length"} 2.0\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_total{engine="0",position="2"} 7.0\n'
            'vllm:spec_decode_num_accepted_tokens_per_pos_created{engine="0",position="2"} 1.7e9\n'
            'vllm:generation_tokens_total{engine="0"} 1.5e+06\n# HELP x\n')
    m = sb.parse_metrics(text)
    assert m == {"request_success_total": 7.0, "accepted_pos2": 7.0, "vllm:generation_tokens_total": 1.5e6}, m


def run_mock(**kw):
    srv, st = mock_server.serve(PORT, slots=4, latency=0.05, **kw)
    try:
        lp = sb.logprob_check(sessions)
        res = sb.run_load(sessions, tools, warmup_s=8, measure_s=6, windows=3, poll_s=0.5, stagger_s=1)
    finally:
        srv.shutdown(); srv.server_close()
    return lp, res, st


def test_load_and_logprobs_against_mock():
    lp, res, st = run_mock(prefix=True)
    assert len(lp) == 2 * sb.LOGPROB_GAMES * sb.LOGPROB_CUTS and all("top" in r for r in lp), lp[:1]
    same = sb.compare_logprobs(lp, lp, 0, 1)
    assert same == {"n": sb.LOGPROB_GAMES * sb.LOGPROB_CUTS, "top1_mismatch": 0, "mean_abs": 0.0, "max_abs": 0.0}, same
    ok = [r for r in res["records"] if r.get("status") == 200]
    assert len({r["game"] for r in ok}) == 25, "every session issued requests"
    assert res["errors_after_warmup"] == 0, [r for r in res["records"] if r.get("status") != 200][:3]
    assert res["stats"].get("tokenize_ok", 0) > 0 and not res["stats"].get("tokenize_fallback")
    turn_prompts = [r["prompt_tokens"] for r in ok if r["turn_start"]]
    # growth + trim hold turn-start prompts near the budget: P's live mean was 21.3K
    # (Duck's one-block trim granularity can drop a single large turn, so bound the mean, not the minimum)
    assert sum(turn_prompts) / len(turn_prompts) > 0.75 * sb.TURN_BUDGET and max(turn_prompts) <= sb.TURN_BUDGET + 600, \
        (min(turn_prompts), sum(turn_prompts) / len(turn_prompts), max(turn_prompts))
    assert 1.0 < res["requests_per_turn"] <= sb.MAX_TURN_REQUESTS, res["requests_per_turn"]
    for w in res["windows"]:
        assert "error" not in w, w
        assert w["gen_tok_s"] > 0 and abs(w["accept_rate_by_pos"]["accepted_pos2"] - 18 / 40) < 1e-9
        assert w["running_mean"] is not None and w["client_completed"] > 0
    seeds = {r["seed"] for r in st.requests if r["temperature"] > 0}
    assert len(seeds) > 25, "per-request seeds vary"
    print("load ok:", {k: round(v, 2) if isinstance(v, float) else v for k, v in res["total"].items()
                       if k in ("gen_tok_s", "requests_s", "client_completed", "prompt_tokens_mean")},
          "req/turn", round(res["requests_per_turn"], 2), "retries", res["stats"].get("max_tokens_retries", 0))


def test_tokenize_undercount_is_absorbed_without_shrinking_history():
    _, res, _ = run_mock(undercount=2500)
    ok = [r for r in res["records"] if r.get("status") == 200]
    assert res["stats"].get("max_tokens_retries", 0) > 0, "the overflow path was exercised"
    assert res["errors_after_warmup"] == 0 and not [r for r in res["records"] if r.get("status") == 400]
    assert sum(r["prompt_tokens"] for r in ok) / len(ok) > 0.6 * sb.TURN_BUDGET


def test_corrupted_prefix_cache_is_detected():
    srv, st = mock_server.serve(PORT, slots=4, latency=0.0, prefix=False)
    try:
        clean = sb.logprob_check(sessions)
    finally:
        srv.shutdown(); srv.server_close()
    srv, st = mock_server.serve(PORT, slots=4, latency=0.0, prefix=True, corrupt_prefix=True)
    try:
        bad = sb.logprob_check(sessions)
    finally:
        srv.shutdown(); srv.server_close()
    d = sb.compare_logprobs(clean, bad, 0, 1)
    assert d["mean_abs"] >= 0.4, d


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("PASS", name)
