"""Build arm P notebooks = KF + load-time REAP-k448 expert pruning, with the freed VRAM given to the KV cache.

  python3 prune/build_p.py            # probe (1,200 s/game) -> prune/p-probe/
  python3 prune/build_p.py --full     # two full passes (7,920 s/game) -> prune/p1/, prune/p2/

Changes vs KF (everything else byte-identical):
  cell 3  profile kv5-c8-cg32 -> kv12-c12-cg48; writes the flashnext_prune plugin + mask to /tmp and puts it
          on PYTHONPATH with FLASHNEXT_PRUNE_MASK, before serving setup starts vLLM.
  new cell after setup: guard — the server log must show the plugin's load_weights line and "Model loading
          took" < 76 GiB, else the run stops before any game is played.
  cell 13 (probe only) 1,200 s per game.
"""
import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLASH = HERE.parent
KF_NB = FLASH / "kf2/kernel/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb"
KF_META = FLASH / "kf2/kernel/kernel-metadata.json"
PLUGIN = (HERE / "plugin/flashnext_prune/__init__.py").read_text()
EPOINTS = (HERE / "plugin/flashnext_prune-0.1.dist-info/entry_points.txt").read_text()
MASK = (HERE / "k448_mask_layers.json").read_text()

OLD_PROFILE = {"'kv5-bf16-mtp3-c8-cg32'": "'kv12-bf16-mtp3-c12-cg48-reap448'",
               '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "5368709120"': '"TAAF_VLLM_KV_CACHE_MEMORY_BYTES": "12884901888"',
               '"TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "32"': '"TAAF_VLLM_MAX_CUDAGRAPH_CAPTURE_SIZE": "48"',
               '"TAAF_VLLM_MAX_NUM_SEQS": "8"': '"TAAF_VLLM_MAX_NUM_SEQS": "12"'}

INSTALL = f'''
# ARM P: load-time REAP-k448 pruning (vLLM general plugin), installed before serving setup starts vLLM.
_PRUNE_DIR = Path("/tmp/flashnext_prune_pkg")
(_PRUNE_DIR / "flashnext_prune").mkdir(parents=True, exist_ok=True)
(_PRUNE_DIR / "flashnext_prune-0.1.dist-info").mkdir(parents=True, exist_ok=True)
(_PRUNE_DIR / "flashnext_prune" / "__init__.py").write_text({PLUGIN!r})
(_PRUNE_DIR / "flashnext_prune-0.1.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\\nName: flashnext-prune\\nVersion: 0.1\\n")
(_PRUNE_DIR / "flashnext_prune-0.1.dist-info" / "entry_points.txt").write_text({EPOINTS!r})
(_PRUNE_DIR / "k448_mask_layers.json").write_text({MASK!r})
os.environ["FLASHNEXT_PRUNE_MASK"] = str(_PRUNE_DIR / "k448_mask_layers.json")
os.environ["PYTHONPATH"] = os.pathsep.join([str(_PRUNE_DIR)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
print("ARM P plugin installed", _PRUNE_DIR, flush=True)
'''

GUARD = '''# ARM P guard: pruning must be live in the served model before any game is played.
import re as _re
_log = (WORKING_DIR / "vllm-openai-server.log").read_text(errors="replace")
_lw = _re.findall(r"FLASHNEXT_PRUNE pid=\\d+ load_weights totals (\\{[^}]*\\}) blocks=(\\d+) keep=(\\d+)", _log)
_mem = [float(x) for x in _re.findall(r"Model loading took ([\\d.]+) GiB", _log)]
_kv = _re.findall(r"GPU KV cache size: ([\\d,]+) tokens", _log)
print("ARM P GUARD", {"load_weights": _lw, "model_gib": _mem, "kv_tokens": _kv}, flush=True)
assert _lw and all(b == "48" and k == "448" for _, b, k in _lw), "pruning plugin did not run in the worker"
assert _mem and max(_mem) < 76.0, f"model memory {_mem} GiB: pruning did not free VRAM"
print("ARM P LIVE", flush=True)
'''


def build(cap_s: float | None) -> dict:
    nb = json.loads(KF_NB.read_text())
    c3 = "".join(nb["cells"][3]["source"])
    for old, new in OLD_PROFILE.items():
        assert c3.count(old) == 1, old
        c3 = c3.replace(old, new)
    anchor = '# Pin arc_agi\'s cached level_reset_only'
    assert c3.count(anchor) == 1
    c3 = c3.replace(anchor, INSTALL.strip() + "\n" + anchor)
    nb["cells"][3]["source"] = c3.splitlines(keepends=True)
    assert "_source_path_entries" in "".join(nb["cells"][9]["source"])      # setup cell: vLLM is up after it
    nb["cells"].insert(10, {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                            "source": GUARD.splitlines(keepends=True)})
    if cap_s is not None:
        c = "".join(nb["cells"][14]["source"])
        assert c.count("bm.solver.max_runtime_s_per_game = 7920.0") == 1
        nb["cells"][14]["source"] = c.replace("bm.solver.max_runtime_s_per_game = 7920.0",
                                              f"bm.solver.max_runtime_s_per_game = {cap_s}  # ARM P probe").splitlines(keepends=True)
    return nb


def write(nb: dict, arm: str) -> None:
    out = HERE / arm
    out.mkdir(exist_ok=True)
    (out / KF_NB.name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    meta = json.loads(KF_META.read_text())
    meta["id"], meta["title"] = f"pranavagrawai/duck-flash-next-{arm}", f"duck-flash-next-{arm}"
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(out, meta["id"])


if __name__ == "__main__":
    if "--full" in sys.argv:
        nb = build(None)
        write(nb, "p1"); write(nb, "p2")
    else:
        write(build(1200.0), "p-probe")
