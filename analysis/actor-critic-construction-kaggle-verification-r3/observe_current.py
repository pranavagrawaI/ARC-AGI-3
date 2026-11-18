"""Read-only post-admission observer; current status is explicitly supplemental."""
import importlib.util
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('gate', HERE / 'launch_control.py')
assert spec and spec.loader
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
CURRENT = '''import json,sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
api=KaggleApi();api.authenticate()
r=ApiGetKernelSessionStatusRequest();r.user_name,r.kernel_slug=sys.argv[1].split('/')
with api.build_kaggle_client() as c:
 response=c.kernels.kernels_api_client.get_kernel_session_status(r)
print(json.dumps({'target':sys.argv[1],'provider_status':response.status.name,'failure_message':response.failure_message,'version_scope':'CURRENT_IDENTITY_ONLY'}))
'''

def main():
    run = gate.RUN_DIR
    launch = gate.launch_identity(run)
    # Do not loop on the known unsupported exact-version endpoint.
    gate.command(run, 'r3-exact-status', [gate.SDK_PYTHON, '-c', gate.STATUS_PROGRAM, gate.TARGET, str(launch['version'])], timeout=60)
    deadline = time.monotonic() + 7200
    while time.monotonic() < deadline:
        path, receipt = gate.command(run, 'r3-current-status', [gate.SDK_PYTHON, '-c', CURRENT, gate.TARGET], timeout=60)
        try:
            state = json.loads(gate.output(receipt))
            assert state['target'] == gate.TARGET and state['version_scope'] == 'CURRENT_IDENTITY_ONLY'
            if state['provider_status'] in {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}:
                result = {'status': state['provider_status'], 'target': gate.TARGET, 'requested_launch_version': launch['version'], 'version_scope': 'CURRENT_IDENTITY_ONLY', 'receipt': path.name}
                gate.exclusive(run / 'R3-CURRENT-TERMINAL.json', result)
                print(json.dumps(result), flush=True)
                return 0
        except (ValueError, KeyError, AssertionError):
            pass
        time.sleep(30)
    print(json.dumps({'status': 'LOCAL_OBSERVER_DEADLINE', 'remote_shutdown_established': False}), flush=True)
    return 3

if __name__ == '__main__':
    raise SystemExit(main())
