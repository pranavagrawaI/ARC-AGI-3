from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from hybrid.worker import main
main()
