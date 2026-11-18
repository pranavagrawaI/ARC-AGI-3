"""CPU-only replay of actual frozen guard against captured GPU table."""
import importlib.util,json,os,sys,tempfile
from pathlib import Path
from unittest.mock import patch
P=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(P/'sources'))
spec=importlib.util.spec_from_file_location('frozen_trial6_prepare',P/'sources/prepare.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
raw=json.loads((P/'remote-readback/output/serving-candidate-feas6/output/actual-nvidia-smi.json').read_text())['raw']
query=(P/'remote-readback/output/serving-candidate-feas6/output/hardware.stdout').read_text()
with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,SCREEN_OUTPUT=directory),patch.object(module.subprocess,'check_output',return_value=raw) as called:
    try:module.host_guard()
    except ValueError as error:
        assert str(error)=='actual GPU identity mismatch';message=str(error)
    else:raise AssertionError('frozen guard unexpectedly passed captured table')
    assert called.call_count==1
    assert not (Path(directory)/'host-headroom.json').exists()
assert query.split(',')[0]=='NVIDIA RTX PRO 6000 Blackwell Server Edition'
result=dict(status='FAILURE_REPRODUCED_CPU_ONLY',frozen_guard_error=message,
            query_reports_exact_expected_name=True,table_has_full_expected_name=False,
            truncated_table_fragment='NVIDIA RTX PRO 6000 Blac...',
            external_calls=0,GPU_calls=0,host_headroom_evaluated=False,
            minimal_correction='Use exact one-row CSV query name and memory for identity; retain table for raw evidence/CUDA capability only.')
if __name__=='__main__':print(json.dumps(result,indent=2))
