"""Fresh-output, CPU-only residual seam qualification. No remote/scientific execution."""
import argparse,io,json,socket,sys,time,unittest
from pathlib import Path
import context,bridge

def verify(destination):
    destination=Path(destination);destination.mkdir(exist_ok=False);started=time.monotonic();context.verify_local()
    calls=[]
    def blocked(*args,**kwargs):calls.append('network');raise AssertionError('offline residual qualification')
    originals=socket.socket.connect,socket.socket.connect_ex,socket.create_connection
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
    output=io.StringIO()
    try:
        suite=unittest.defaultTestLoader.discover(str(context.HERE/'tests'))
        result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
        plan=bridge.prepare()
    finally:socket.socket.connect,socket.socket.connect_ex,socket.create_connection=originals
    (destination/'TESTS.log').write_text(output.getvalue());context.verify_local()
    report=dict(pass_=result.wasSuccessful() and not calls,tests=result.testsRun,elapsed_seconds=time.monotonic()-started,
        network_attempts=len(calls),original_bindings=13118,first_complete=False,first_after_available=False,
        first_files=len(plan['first_files']),first_attempts=len(plan['first_attempts']),conditional_waivers=len(plan['waivers']),residual_files=len(plan['remaining']),
        expected_combined_successful_files=1586,expected_combined_attempts=1587,repeat_connection_name=context.REPEAT,
        source_pins_sha256=context.sha(context.HERE/'SOURCE-PINS.json'),manifest_sha256=context.sha(context.HERE/'RESIDUAL-MANIFEST.json'),
        first_collection_sha256=context.FIRST_SHA,root_local_authority_sha256=context.REVIEW_SHA,
        GREEN_log_sha256=context.sha(destination/'TESTS.log'),RED_log_sha256=context.sha(context.HERE/'RED.log'),
        full_scientific_audit_executed=False,remote_collection_executed=False,remote_authorized=False,
        approved_collector_and_audit_r2_unchanged=True,frozen_bindings_unchanged=True,runtime_baseline_rerun=False,
        accepted_rate=None,ratio=None,candidate_negative=False,adoption=False,
        fixture_scope='residual filtering/limits/error stop, actual retained files/waivers, explicit two-pass ledger; no mocked full scientific pass',destination=str(destination.resolve()))
    (destination/'PROOF.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['pass_'] else 1
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('destination',type=Path);args=p.parse_args();sys.exit(verify(args.destination))
