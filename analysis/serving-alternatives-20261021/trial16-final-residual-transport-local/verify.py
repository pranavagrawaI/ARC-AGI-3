"""Exclusive-output CPU boundary qualification; no collection/audit execution."""
import argparse,io,json,socket,sys,time,unittest
from pathlib import Path
import settings,ledger

def verify(destination):
    destination=Path(destination);destination.mkdir(exist_ok=False);started=time.monotonic();settings.verify_local()
    calls=[]
    def blocked(*args,**kwargs):calls.append('network');raise AssertionError('offline transport-only qualification')
    original=socket.socket.connect,socket.socket.connect_ex,socket.create_connection
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
    output=io.StringIO()
    try:
        result=unittest.TextTestRunner(stream=output,verbosity=2).run(unittest.defaultTestLoader.discover(str(settings.HERE/'tests')))
        plan=ledger.prepare()
    finally:socket.socket.connect,socket.socket.connect_ex,socket.create_connection=original
    (destination/'TESTS.log').write_text(output.getvalue());settings.verify_local()
    proof=dict(pass_=result.wasSuccessful() and not calls,tests=result.testsRun,seconds=time.monotonic()-started,network_attempts=len(calls),
        original_bindings=13118,retained_files=len(plan['retained_files']),residual_files=len(plan['remaining']),conditional_waivers=len(plan['waivers']),
        first_complete=False,second_complete=False,first_after=False,second_after=False,
        first_collection_sha256=settings.FIRST_SHA,second_collection_sha256=settings.SECOND_SHA,
        source_pins_sha256=settings.sha(settings.HERE/'SOURCE-PINS.json'),manifest_sha256=settings.sha(settings.HERE/'FINAL-MANIFEST.json'),
        tests_log_sha256=settings.sha(destination/'TESTS.log'),RED_sha256=settings.sha(settings.HERE/'RED.log'),
        bounds=settings.POLICY,full_scientific_audit_executed=False,collection_executed=False,remote_authorized=False,
        approved_sources_unchanged=True,runtime_baseline_rerun=False,GPU_work=False,accepted_rate=None,ratio=None,adoption=False,candidate_negative=False,
        fixture_scope='transport retry boundary / exact retained three-pass ledger / unchanged checker source identity; no simulated full scientific pass',destination=str(destination.resolve()))
    (destination/'PROOF.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof,indent=2));return 0 if proof['pass_'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);a=p.parse_args();sys.exit(verify(a.destination))
