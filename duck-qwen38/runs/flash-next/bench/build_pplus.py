"""Build the P+ validation notebooks = arm P's notebook with the screen's winning serving overrides, nothing else.

  python3 bench/build_pplus.py bench/out-v2/screen-assessment.json     # -> bench/pplus1/, bench/pplus2/

Changes vs prune/p1 (all in cell 3's PUBLIC25_VLLM_PROFILE_ENV, plus a liveness assert in the guard cell):
  - each TAAF_VLLM_* override from the assessment's p_plus_overrides (replaced if present, inserted if not);
  - profile name suffixed with the overrides;
  - guard: the server's non-default args must show each intended flag before any game is played.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / "prune/p1"
NB = "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb"
ALLOWED = {"TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION", "TAAF_VLLM_ENABLE_PREFIX_CACHING", "TAAF_VLLM_MTP_TOKENS",
           "TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE"}


def liveness_guard(overrides: dict) -> str:
    checks = []
    if overrides.get("TAAF_VLLM_MTP_INDEX_SHARE_FOR_ITERATION") == "1":
        checks.append(r'"index_share": bool(_re.search(r"index_share_for_mtp_iteration[\'\"]?\s*:\s*[Tt]rue", _args))')
    if overrides.get("TAAF_VLLM_ENABLE_PREFIX_CACHING") == "1":
        checks.append(r'"prefix": bool(_re.search(r"enable_prefix_caching[\'\"]?\s*:\s*[Tt]rue", _args))')
    if "TAAF_VLLM_MTP_TOKENS" in overrides:
        k = overrides["TAAF_VLLM_MTP_TOKENS"]
        checks.append(r'"mtp": bool(_re.search(r"num_speculative_tokens[\'\"]?\s*:\s*' + k + r'\b", _args))')
    return (
        "# P+ guard: every intended serving flag must be live in the server's non-default args.\n"
        "_args = (_re.findall(r\"non-default args: (\\{.*)\", _log) or [''])[-1]\n"
        "_live = {" + ", ".join(checks) + "}\n"
        "print('P+ FLAGS', _live, flush=True)\n"
        "assert _live and all(_live.values()), f'P+ flags not live: {_live}'\n"
    )


def build(overrides: dict) -> dict:
    assert overrides and set(overrides) <= ALLOWED, overrides
    nb = json.loads((P1 / NB).read_text())
    c3 = "".join(nb["cells"][3]["source"])
    m = re.search(r"PUBLIC25_VLLM_PROFILE_ENV = \{\n(.*?)\n\}", c3, re.S)
    lines = m[1].split("\n")
    env = {re.match(r'\s*"(\w+)": "([^"]*)"', l)[1]: re.match(r'\s*"(\w+)": "([^"]*)"', l)[2] for l in lines}
    env.update(overrides)
    block = "PUBLIC25_VLLM_PROFILE_ENV = {\n" + ",\n".join(f'    "{k}": "{v}"' for k, v in sorted(env.items())) + "\n}"
    c3 = c3[:m.start()] + block + c3[m.end():]
    name_old = "PUBLIC25_VLLM_PROFILE_NAME = 'kv12-bf16-mtp3-c12-cg48-reap448'"
    assert c3.count(name_old) == 1
    tag = "-".join(f"{k.removeprefix('TAAF_VLLM_').lower()}{v}" for k, v in sorted(overrides.items()))
    c3 = c3.replace(name_old, f"PUBLIC25_VLLM_PROFILE_NAME = 'kv12-bf16-mtp3-c12-cg48-reap448-pplus-{tag}'")
    nb["cells"][3]["source"] = c3.splitlines(keepends=True)
    guard = "".join(nb["cells"][10]["source"])
    assert guard.rstrip().endswith('print("ARM P LIVE", flush=True)')
    nb["cells"][10]["source"] = (guard.rstrip() + "\n" + liveness_guard(overrides)).splitlines(keepends=True)
    return nb


def main() -> None:
    assessment = json.loads(Path(sys.argv[1]).read_text())
    decision = assessment["decision"]
    assert decision["status"] == "PASS", f"screen did not pass: {decision}"
    nb = build(decision["p_plus_overrides"])
    for arm in ("pplus1", "pplus2"):
        out = HERE / arm
        out.mkdir(exist_ok=True)
        (out / NB).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        meta = json.loads((P1 / "kernel-metadata.json").read_text())
        meta["id"], meta["title"] = f"pranavagrawai/duck-flash-next-{arm}", f"duck-flash-next-{arm}"
        (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(out, meta["id"])


if __name__ == "__main__":
    main()
