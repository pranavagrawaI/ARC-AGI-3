"""Focused network-zero qualification; does not rerun runtime61/65 or science."""
import ast,hashlib,io,json,socket,sys,time,trace,types,unittest
from pathlib import Path
from binding import HERE,ROOT,OVERLAY

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def executable_lines(code):
    lines={line for _,_,line in code.co_lines() if line is not None}
    for item in code.co_consts:
        if isinstance(item,types.CodeType):lines|=executable_lines(item)
    return lines

def risk_report(counts):
    report=[]
    for name in ('collector.py','remote.py','audit.py','run.py'):
        path=HERE/name;text=path.read_text();tree=ast.parse(text);executable=executable_lines(compile(text,str(path),'exec'))
        for node in ast.walk(tree):
            if not isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):continue
            statements={line for line in executable if node.lineno<=line<=node.end_lineno}
            covered={line for line in statements if counts.get((str(path),line),0)>0}
            cc=1+sum(isinstance(x,(ast.If,ast.IfExp,ast.For,ast.While,ast.ExceptHandler,ast.comprehension)) for x in ast.walk(node))
            cc+=sum(max(0,len(x.values)-1) for x in ast.walk(node) if isinstance(x,ast.BoolOp))
            fraction=len(covered)/len(statements) if statements else 1
            report.append(dict(file=name,function=node.name,CC=cc,line_coverage=fraction,CRAP=cc**2*(1-fraction)**3+cc))
    return dict(contract='Explicit local AST branch CC + CPython executable-line tracing; review aid, no threshold/CI/dependency change and no comparability claim to radon.',functions=report,
        limits='Supplemental full-scientific execution cannot be qualified against absent raw files. Focused tests cover input/filter/budget/failure/origin boundaries; original scientific checker is unchanged.')

def verify():
    import run,build_manifest
    start=time.monotonic();run.verify_local();pins=json.loads((HERE/'SOURCE-PINS.json').read_bytes())
    before={name:sha(HERE/name) for name in pins['files']}
    assert json.loads((HERE/'MISSING-MANIFEST.json').read_bytes())==build_manifest.build()
    network=[]
    def blocked(*args,**kwargs):network.append('NETWORK_ATTEMPT');raise AssertionError('offline-only qualification')
    original=(socket.socket.connect,socket.socket.connect_ex,socket.create_connection)
    socket.socket.connect=blocked;socket.socket.connect_ex=blocked;socket.create_connection=blocked
    stream=io.StringIO();tracer=trace.Trace(count=True,trace=False)
    try:
        suite=unittest.defaultTestLoader.discover(str(HERE/'tests'))
        result=tracer.runfunc(unittest.TextTestRunner(stream=stream,verbosity=2).run,suite)
    finally:socket.socket.connect,socket.socket.connect_ex,socket.create_connection=original
    (HERE/'qualification/TESTS.log').write_text(stream.getvalue())
    (HERE/'qualification/RISK.json').write_text(json.dumps(risk_report(tracer.results().counts),indent=2)+'\n')
    run.verify_local();after={name:sha(HERE/name) for name in pins['files']}
    report=dict(pass_=result.wasSuccessful() and before==after and not network,tests=result.testsRun,seconds=time.monotonic()-start,
        network_attempts=len(network),GPU_work=False,original_bindings=13118,source_stable=before==after,
        source_pins_sha256=sha(HERE/'SOURCE-PINS.json'),manifest_sha256=sha(HERE/'MISSING-MANIFEST.json'),
        exact_names=1596,conditional_bodies=10,concurrency=1,remote_authorized=False,original_overlay_status='INVALID_INCONCLUSIVE',
        runtime_baseline_rerun=False,inherited_baseline='64/65 conservative identity abort retained',
        red_green_evidence={name:sha(HERE/'qualification'/name) for name in ('red-filter.log','green-filter.log','red-actual-result.log','green-actual-result.log')},
        full_supplemental_raw_qualification='NOT_RUN_MISSING_INPUTS; remains guarded until exact remote approval and complete collection')
    (HERE/'qualification/PROOF.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2));return 0 if report['pass_'] else 1
if __name__=='__main__':sys.exit(verify())
