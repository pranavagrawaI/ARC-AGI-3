"""Build arm G (graft) = KF notebook + Scott Le Grand's image block, run as two passes (g1, g2).

  python3 g/build_g.py

Image block = agentfix F1 (history images kept for the newest 2 messages, flat estimator, 6144 margin),
F13 (animation frames in the sandbox), F19 (contact sheet), ARM P (12x board image), plus F9 timing
(observation only; gives real prompt tokens per request). Every other agentfix switch is OFF, so KF's own
memory, yield, game-over and level-up behaviour is untouched. The 16K context line is NOT grafted:
Scott's run served at 32768 and 863/1045 requests exceeded 16384, so it never applied there either.
Source: analysis/scottlegrand-sheetu12b-20260929/taaf-flashnext-sheetu12b-0922.ipynb cell 10.
"""
import copy
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLASH = HERE.parent
SCOTT = Path.home() / "projects/arc-agi-3/analysis/scottlegrand-sheetu12b-20260929/taaf-flashnext-sheetu12b-0922.ipynb"
KF_NB = FLASH / "kf2/kernel/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb"
KF_META = FLASH / "kf2/kernel/kernel-metadata.json"

SWITCHES = {
    "AGENTFIX_IMAGES": "1", "AGENTFIX_ANIM": "1", "AGENTFIX_SHEET": "1", "AGENTFIX_TIMING": "1",
    "AGENTFIX_ANIM_SUMMARY": "0", "AGENTFIX_MEMORY": "0", "AGENTFIX_DEDUP": "0", "AGENTFIX_LEDGER": "0",
    "AGENTFIX_LOOP": "0", "AGENTFIX_ACTION7": "0", "AGENTFIX_STALL": "0", "AGENTFIX_RESULT": "0",
    "AGENTFIX_NOIMPACT": "0", "ARMP_UPSCALE": "12",
}
DEFAULTS_END = "# cell defaults set BEFORE the module sources: switches read at exec time see them\n"

GUARD = '''
# ===== arm G guard: the image block must be LIVE and nothing else may be =====
assert _rep.get('F1_images') and _rep.get('F9_timing'), _rep
assert _rep.get('F13_anim') is True and 'anim_error' not in _rep, _rep
assert _rep.get('F19_sheet') is True and 'sheet_unavailable' not in _rep, _rep
for _k in ('F2_result', 'F3_memory', 'F4_loop', 'F5_noimpact', 'F6_action7', 'F7_ledger', 'F10_stall', 'F11_dedup', 'F13_summary'):
    assert not _rep.get(_k), (_k, _rep)
assert os.environ['MULTIMODAL_UPSCALE'] == '12' and _vc.current_grid_image_upscale() == 12
print('ARM G LIVE: images+anim+sheet+upscale12, all other agentfix switches off', flush=True)
'''


def graft_cell() -> str:
    scott = json.loads(SCOTT.read_text())
    src = "".join(scott["cells"][10]["source"])
    head, sep, body = src.partition(DEFAULTS_END)
    assert sep and head.count("_os.environ.setdefault(") == 11, "Scott's switch block moved"
    switches = "".join(f"_os.environ[{k!r}] = {v!r}\n" for k, v in SWITCHES.items())
    header = ("# ARM G: Scott Le Grand's image block (taaf-flashnext-sheetu12b-0922 cell 10, verbatim below the switches)\n"
              "# grafted onto KF. Switches are FORCED, not defaulted.\nimport os as _os\n")
    assert "os.environ.setdefault('ARMP_UPSCALE', '12')" in body
    return header + switches + DEFAULTS_END + body + GUARD


def main() -> None:
    nb = json.loads(KF_NB.read_text())
    assert "AGENTFIX" not in json.dumps(nb), "KF notebook already carries agentfix"
    cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": graft_cell().splitlines(keepends=True)}
    # after cell 9 (source import + serving setup), before cell 10 (load the benchmark): Scott's position
    assert "_source_path_entries" in "".join(nb["cells"][9]["source"])
    assert "Load the benchmark" in "".join(nb["cells"][10]["source"])
    nb["cells"].insert(10, cell)
    meta = json.loads(KF_META.read_text())
    for arm in ("g1", "g2"):
        out = HERE / arm
        out.mkdir(exist_ok=True)
        (out / KF_NB.name).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
        m = copy.deepcopy(meta)
        m["id"] = f"pranavagrawai/duck-flash-next-{arm}"
        m["title"] = f"duck-flash-next-{arm}"
        (out / "kernel-metadata.json").write_text(json.dumps(m, indent=2) + "\n")
        print(out, m["id"], m["dataset_sources"])


if __name__ == "__main__":
    main()
