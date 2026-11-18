"""Offline qualification of arm G: exec the grafted cell against KF's real source tree.

  cd runs/flash-next && PYTHONPATH=kf/source-dataset/src/ARC3-Inference:kf/source-dataset/src/tufa-arc-agi-framework/src \
    ~/projects/arc-agi-3-reference/duck-critic-adapter/ARC3-Inference/.venv/bin/python -B g/qualify_g.py
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.update(MULTIMODAL_UPSCALE="4", MULTIMODAL_CONTEXT="current_grid")   # what KF's serving setup persists
nb = json.loads((HERE / "g1/duck-qwen3-8-flash-next-nvfp4-mtp.ipynb").read_text())
cell = "".join(nb["cells"][10]["source"])
ns = {"__name__": "__graft__"}
exec(compile(cell, "graft-cell", "exec"), ns)
rep = ns["_rep"]
print(json.dumps({k: v for k, v in rep.items() if not isinstance(v, dict)}, default=str))
