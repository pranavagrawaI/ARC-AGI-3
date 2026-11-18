"""Reuse the cn04 trace indexer on four s5i5 L2 attempts."""
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
path = ROOT/'analysis/cn04-b1-inspection-20261012/index_records.py'
spec = importlib.util.spec_from_file_location('trace_indexer', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.main(sources={
    'b1-r1':ROOT/'solvers/m2_b1/experiment/pair-1/b1-r1/output',
    'b1-r2':ROOT/'solvers/m2_b1/experiment/pair-1/b1-r2/output',
    'control-pc':ROOT/'solvers/m2_evidence/experiment/pair-1/control/output',
    'evidence-pt':ROOT/'solvers/m2_evidence/experiment/pair-1/treatment/output',
}, game_id='s5i5-18d95033', output_dir=OUT)
