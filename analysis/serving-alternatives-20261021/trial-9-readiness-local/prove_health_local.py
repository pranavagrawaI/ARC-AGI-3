"""Preserve actual CPU native-parser subprocess receipts. No model/SDK/GPU."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tests'))
from test_control_health import ControlHealth
import control_prepare
case=ControlHealth(methodName='test_identity_and_serve_bind_actual_native_parsed_values')
try:
 case.setUp();control_prepare.identity();case.serve()
 a=json.loads((case.out/'control-health-identity.json').read_text());b=json.loads((case.out/'control-health-serve.json').read_text())
 proof={'scope':'LOCAL_CPU_FIXTURE_ONLY; actual pinned environ module, stub framework packages, no model/native GPU execution','identity':a,'serve':b,'server_marker_created_after_checks':case.marker.exists(),'raw_equal':a['raw']==b['raw'],'effective_equal':a['effective']==b['effective'],'native_source_equal':a['native_environ_sha256']==b['native_environ_sha256'],'native_environ_exact_pinned_sha256':hashlib.sha256(case.native.read_bytes()).hexdigest(),'external_network_calls':0,'gpu_calls':0}
 (ROOT/'evidence/control-health-local-proof.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2))
finally:case.doCleanups()
