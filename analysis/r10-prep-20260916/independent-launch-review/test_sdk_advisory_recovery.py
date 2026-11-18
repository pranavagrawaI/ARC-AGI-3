"""Offline exact advisory replay through the frozen gate; no provider call."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'r10a-verification'))
from test_launch_control import world,gate,invoke

def test_known_sdk_advisory_retains_identity_and_never_repushes(world):
    advisory=json.loads((Path(__file__).parent/'sdk-mapping.json').read_text())['stdout_advisory']
    world.responses[:]=[(0,'[{"resource":"GPU","remaining":"26.39h"}]'),(0,'Not found'),(0,advisory+world.success)]
    assert invoke(world)==2
    assert (world.run/'CONSUMED.json').is_file()
    record=gate.load(world.run/'launch.json')
    assert record['target']==gate.TARGET and record['version']==7
    assert invoke(world)==2
    assert sum('push' in argv for argv in world.calls)==1
    assert invoke(world,'monitor')==0
