"""Offline upfront plan publication; never lists or downloads remote files."""
import hashlib,json,sys
from pathlib import Path
from run_required import prepare,verify_freeze

def main(primary,terminal,destination):
 destination=Path(destination);destination.mkdir(exist_ok=False)
 try:
  freeze=verify_freeze();raw,bundle,values,plan=prepare(primary,terminal)
  value=dict(freeze_sha256=freeze,primary_sha256=hashlib.sha256(raw).hexdigest(),manifest_sha256=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest(),plan=plan,remote_authorized=False)
  (destination/'REQUIRED-PLAN.json').write_text(json.dumps(value,indent=2)+'\n');return 0
 except Exception as error:
  value=dict(complete_plan=False,remote_authorized=False,error_type=type(error).__name__,missing_name=error.args[0] if isinstance(error,KeyError) else None,reason=str(error),no_automatic_fetch=True)
  (destination/'REQUIRED-GAPS.json').write_text(json.dumps(value,indent=2)+'\n');return 1
if __name__=='__main__':
 if len(sys.argv)!=4:raise SystemExit('plan_required.py PRIMARY.json ABS_TERMINAL_READBACK FRESH_PLAN_DIRECTORY')
 raise SystemExit(main(*sys.argv[1:]))
