"""Pinned current Actor imports. No credential loading and no network."""

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = Path("/") if "OT_RUNTIME" in os.environ else HERE.parents[1]
RUNTIME = (
    Path(os.environ["OT_RUNTIME"])
    if "OT_RUNTIME" in os.environ
    else ROOT
    / "deployment/kaggle/submission-20260917-actor-critic-competition-v1/runtime"
)
sys.path[:0] = [str(RUNTIME / "chassis"), str(RUNTIME)]
