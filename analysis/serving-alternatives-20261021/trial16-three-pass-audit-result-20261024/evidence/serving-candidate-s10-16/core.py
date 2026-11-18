"""Packet-relative loader: exact accepted runtime/dependency bytes before import."""
import hashlib,json,sys
from pathlib import Path
HERE=ROOT=Path(__file__).resolve().parent
pins=json.loads((HERE/'runtime-pins.json').read_text())
expected={'capture.py','command_child.py','no_processes.py','artifacts.py','controller_shutdown.py','adapters.py','runtime_scope.py','registration.py','resources.py','probe_ipc.py'}
if set(pins)!=expected:raise ValueError('release runtime dependency set drift')
for name,digest in pins.items():
    if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest:raise ValueError('release runtime source drift '+name)
sys.path.insert(0,str(HERE))
import controller_shutdown as shutdown
import resources,probe_ipc
for module in (shutdown,resources,probe_ipc):
    if Path(module.__file__).resolve().parent!=HERE:raise ValueError('wrong actual release runtime import')
