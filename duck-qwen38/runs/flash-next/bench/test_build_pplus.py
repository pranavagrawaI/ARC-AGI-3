"""python3 -B bench/test_build_pplus.py — P+ differs from P only in its profile env + flag guard, and the guard bites."""
import ast
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_pplus import P1, NB, build  # noqa: E402

OV = {"TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION": "1", "TAAF_VLLM_ENABLE_PREFIX_CACHING": "1",
      "TAAF_VLLM_MTP_TOKENS": "4", "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "60"}
LIVE = ("INFO non-default args: {'enable_prefix_caching': True, 'speculative_config': {'method': 'mtp', "
        "'num_speculative_tokens': 4, 'index_share_for_mtp_iteration': True}}\n")


def guard_tail(nb) -> str:
    src = "".join(nb["cells"][10]["source"])
    return src[src.index("# P+ guard"):]


def test_only_profile_and_guard_change():
    p, q = json.loads((P1 / NB).read_text())["cells"], build(OV)["cells"]
    assert len(p) == len(q)
    assert [i for i in range(len(p)) if p[i] != q[i]] == [3, 10]
    c3p, c3q = "".join(p[3]["source"]), "".join(q[3]["source"])
    env = ast.literal_eval(re.search(r"PUBLIC25_VLLM_PROFILE_ENV = (\{.*?\n\})", c3q, re.S)[1])
    base = ast.literal_eval(re.search(r"PUBLIC25_VLLM_PROFILE_ENV = (\{.*?\n\})", c3p, re.S)[1])
    assert env == {**base, **OV}, env
    strip = lambda s: re.sub(r"PUBLIC25_VLLM_PROFILE_(ENV = \{.*?\n\}|NAME = '[^']*')", "", s, flags=re.S)
    assert strip(c3p) == strip(c3q), "nothing else in cell 3 changed"
    assert "".join(q[10]["source"]).startswith("".join(p[10]["source"]).rstrip())


def test_guard_passes_live_and_rejects_missing_flags():
    tail = guard_tail(build(OV))
    exec(tail, {"_re": re, "_log": LIVE})
    for broken in (LIVE.replace("'enable_prefix_caching': True", "'enable_prefix_caching': False"),
                   LIVE.replace(", 'index_share_for_mtp_iteration': True", ""),
                   LIVE.replace("'num_speculative_tokens': 4", "'num_speculative_tokens': 3"), "no args here\n"):
        try:
            exec(tail, {"_re": re, "_log": broken})
        except AssertionError:
            continue
        raise AssertionError(f"guard accepted {broken!r}")


def test_single_flag_build():
    nb = build({"TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION": "1"})
    exec(guard_tail(nb), {"_re": re, "_log": "non-default args: {'speculative_config': {'index_share_for_mtp_iteration': True}}"})


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("PASS", n)
