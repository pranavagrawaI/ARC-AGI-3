"""Build the speed-screen notebook = arm P's notebook through its guard cell + the serving benchmark.

  python3 bench/build_sessions.py && python3 bench/build_bench.py     # -> bench/kernel/

Cells 0-10 of prune/p1 are kept byte-identical: P's serving profile (cell 3), the pruning plugin, setup (the S0
server), and the ARM P guard. The game-playing cells (11-18) are replaced by:
  - a cell that writes speedbench.py and the parsed P sessions (gzip+base64) into /kaggle/working/bench;
  - bench/orchestrate.py (S0..S3, one restart per config, results in /kaggle/working/speed-screen.json).
"""
import base64
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P1 = HERE.parent / "prune/p1"
NB = "duck-qwen3-8-flash-next-nvfp4-mtp.ipynb"
SLUG = "duck-flash-next-speed-screen"


def code(src: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def main() -> None:
    nb = json.loads((P1 / NB).read_text())
    assert "ARM P GUARD" in "".join(nb["cells"][10]["source"])
    assert "_source_path_entries" in "".join(nb["cells"][9]["source"])
    sessions = (HERE / "sessions.json").read_bytes()
    blob = base64.b64encode(gzip.compress(sessions, 9)).decode()
    bench_src = (HERE / "speedbench.py").read_text()
    write_cell = (
        "# SPEED SCREEN inputs: benchmark driver + P1's 25 parsed prompt snapshots.\n"
        "import base64, gzip, hashlib\n"
        "_BENCH = WORKING_DIR / 'bench'\n"
        "_BENCH.mkdir(parents=True, exist_ok=True)\n"
        f"(_BENCH / 'speedbench.py').write_text({bench_src!r})\n"
        f"_blob = gzip.decompress(base64.b64decode({blob!r}))\n"
        f"assert hashlib.sha256(_blob).hexdigest() == {hashlib.sha256(sessions).hexdigest()!r}\n"
        "(_BENCH / 'sessions.json').write_bytes(_blob)\n"
        "print('SPEED_SCREEN inputs written', len(_blob), flush=True)\n"
    )
    md = {"cell_type": "markdown", "metadata": {},
          "source": ["## Speed screen\n", "\n",
                     "Serving benchmark on arm P: replay P1's 25 Duck sessions at sustained demand against stacked "
                     "configs S0 (P), S1 (+MTP index sharing), S2 (+prefix caching, align), S3 (+MTP length). "
                     "No games are played. See NEXT-PLAN.md."]}
    nb["cells"] = nb["cells"][:11] + [md, code(write_cell), code((HERE / "orchestrate.py").read_text())]
    out = HERE / "kernel"
    out.mkdir(exist_ok=True)
    (out / NB).write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    meta = json.loads((P1 / "kernel-metadata.json").read_text())
    meta["id"], meta["title"] = f"pranavagrawai/{SLUG}", SLUG
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(out, meta["id"], f"{(out / NB).stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
