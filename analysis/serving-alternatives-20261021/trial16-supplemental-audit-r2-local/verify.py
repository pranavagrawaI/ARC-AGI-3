"""Exclusive-destination, offline output-finalization qualification only."""
import argparse,hashlib,io,json,socket,sys,time,unittest
from pathlib import Path
HERE=Path(__file__).resolve().parent
APPROVED=HERE.parent/'trial16-missing-raw-local'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def validate_pins():
    pins=json.loads((HERE/'SOURCE-PINS.json').read_bytes())
    for name,expected in pins['files'].items():assert sha(HERE/name)==expected,name
    for name,expected in pins['references'].items():assert sha(name)==expected,name
    approved=json.loads((APPROVED/'SOURCE-PINS.json').read_bytes())
    for name,expected in approved['files'].items():assert sha(APPROVED/name)==expected,name
    for name,expected in approved['references'].items():assert sha(name)==expected,name
    root=HERE.parent/'trial-16-candidate-s10-r4-operational-local';freeze=json.loads((root/'freeze.json').read_bytes())
    for name,expected in freeze['files'].items():assert sha(root/name)==expected,name
    for name,expected in freeze['references'].items():assert sha(name)==expected,name
    assert freeze['bindings']==13118

def verify(destination):
    destination=Path(destination);destination.mkdir(exist_ok=False);started=time.monotonic();validate_pins()
    before={str(p):sha(p) for p in (HERE/'audit.py',HERE/'publication.py',APPROVED/'SOURCE-PINS.json',APPROVED/'audit.py')}
    calls=[]
    def blocked(*args,**kwargs):calls.append('network');raise AssertionError('offline boundary fixture only')
    saved=socket.socket.connect,socket.socket.connect_ex,socket.create_connection
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
    output=io.StringIO()
    try:result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.discover(str(HERE/'tests')))
    finally:socket.socket.connect,socket.socket.connect_ex,socket.create_connection=saved
    (destination/'TESTS.log').write_text(output.getvalue());validate_pins()
    stable=before=={name:sha(Path(name)) for name in before}
    report=dict(pass_=result.wasSuccessful() and stable and not calls,tests=result.testsRun,seconds=time.monotonic()-started,
        network_attempts=len(calls),source_stable=stable,approved_collector_unchanged=True,original_bindings=13118,
        source_pins_sha256=sha(HERE/'SOURCE-PINS.json'),approved_source_pins_sha256=sha(APPROVED/'SOURCE-PINS.json'),
        audit_source_sha256=sha(HERE/'audit.py'),publication_sha256=sha(HERE/'publication.py'),
        RED_sha256=sha(HERE/'qualification/RED.log'),GREEN_log_sha256=sha(destination/'TESTS.log'),
        fixture_scope='OUTPUT_FINALIZATION_ONLY_NO_MOCKED_OR_EXECUTED_FULL_SCIENTIFIC_PASS',
        collection_executed_by_this_verifier=False,full_audit_executed=False,runtime_baseline_rerun=False,GPU_work=False,
        inherited64of65='retained',old_original_TESTS_RISK='UNAVAILABLE_NOT_FABRICATED',
        restored_owner_proof_sha256=sha(APPROVED/'qualification/PROOF.json'),destination_honored=str(destination))
    (destination/'PROOF.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['pass_'] else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('destination',type=Path);args=parser.parse_args();sys.exit(verify(args.destination))
