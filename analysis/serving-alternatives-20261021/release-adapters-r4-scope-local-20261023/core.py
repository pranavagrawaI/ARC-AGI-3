"""Load only the accepted immutable local core and its exact dependencies."""
import hashlib
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
CONTRACT=json.loads((HERE/'ACCEPTED-CORE.json').read_text())
ROOT=Path(CONTRACT['directory'])
for name,expected in CONTRACT['sources'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:raise ValueError('accepted core drift: '+name)
sys.path.insert(0,str(ROOT))
import controller_shutdown as shutdown
import resources
import probe_ipc
for module in (resources,probe_ipc):
    if Path(module.__file__).resolve().parent!=ROOT:raise ValueError('wrong actual core import')

if Path(shutdown.__file__).resolve()!=HERE/'controller_shutdown.py':raise ValueError('wrong controller shutdown import')
