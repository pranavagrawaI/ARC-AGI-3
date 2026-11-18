import os
import runpy
from pathlib import Path

os.environ["EXPERIMENT_WORKFLOW"] = "1"
runpy.run_path(str(Path(__file__).with_name("headless.py")), run_name="__main__")
