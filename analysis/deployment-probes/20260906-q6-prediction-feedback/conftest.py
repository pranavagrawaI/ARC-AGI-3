import sys
import os
os.environ["LITELLM_LOCAL_MODEL_COST_MAP"]="True"
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE/'extracted'),str(HERE/'extracted/chassis'),str(ROOT/'deployment/kaggle/tests'),str(ROOT/'deployment/kaggle')]
