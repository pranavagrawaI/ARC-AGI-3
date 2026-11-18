"""Capture the actual one-shot collection/audit CLI outcome; no synthetic success receipt."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run(script,arguments,receipt_directory):
 path=(HERE/script).resolve()
 if path not in ((HERE/'run_required.py').resolve(),(HERE/'audit_required.py').resolve()):raise ValueError('only reviewed collection/offline audit CLI')
 destination=Path(receipt_directory);destination.mkdir(exist_ok=False);started=time.monotonic()
 command=[sys.executable,'-u',str(path),*map(str,arguments)]
 with (destination/'stdout').open('xb') as stdout,(destination/'stderr').open('xb') as stderr:
  completed=subprocess.run(command,stdout=stdout,stderr=stderr)
 ended=time.monotonic();output=(destination/'stdout').read_bytes();errors=(destination/'stderr').read_bytes()
 report=dict(command=command,script_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),start=started,end=ended,
   exit_code=completed.returncode,stdout_bytes=len(output),stdout_sha256=hashlib.sha256(output).hexdigest(),stderr_bytes=len(errors),stderr_sha256=hashlib.sha256(errors).hexdigest(),returned=None)
 if script=='run_required.py':report['output_destination']=str(Path(arguments[-1]).resolve())
 try:
  if len(output)>64*1024**2 or errors:raise ValueError('CLI bounded stdout/empty stderr required')
  report['returned']=json.loads(output)
 except Exception as error:report['capture_error']=type(error).__name__
 with (destination/'EXIT.json').open('x') as stream:json.dump(report,stream,indent=2);stream.flush();os.fsync(stream.fileno())
 return report

if __name__=='__main__':
 if len(sys.argv)<4:raise SystemExit('owned_cli.py FRESH_RECEIPT_DIRECTORY run_required.py|audit_required.py ARGS...')
 value=run(sys.argv[2],sys.argv[3:],sys.argv[1]);print(json.dumps(value));raise SystemExit(0 if value['exit_code']==0 and not value.get('capture_error') else 1)
