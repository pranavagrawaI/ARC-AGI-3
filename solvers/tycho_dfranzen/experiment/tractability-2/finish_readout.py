"""After the read-only collector stops, audit and write outcome/trace indexes; never launch."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--watch-unit',required=True);p.add_argument('--hours',type=float,default=12);args=p.parse_args();run=args.run
end=time.monotonic()+args.hours*3600
while time.monotonic()<end:
 receipt=run/'collection-receipt.json'
 active=subprocess.run(['systemctl','--user','is-active',args.watch_unit],capture_output=True,text=True).stdout.strip()=='active'
 if receipt.exists() and json.loads(receipt.read_text()).get('critical_complete'):break
 if not active:break
 time.sleep(30)
if not (run/'collection-receipt.json').exists():
 (run/'READOUT-PENDING.md').write_text('No completed collection receipt. Inspect status.json, watch-error.json and watch-ended.json. No outcome inferred and no retry launched.\n');raise SystemExit(1)
process=subprocess.run([sys.executable,str(Path(__file__).with_name('audit_run.py')),str(run)],capture_output=True,text=True)
(run/'audit.log').write_text(process.stdout+process.stderr)
if process.returncode:
 (run/'READOUT-ERROR.md').write_text('Audit failed; inspect audit.log. No validity or efficacy inference, no retry.\n');raise SystemExit(process.returncode)
audit=json.loads((run/'AUDIT.json').read_text());records=[]
for c in audit['case_details']:
 root=run/'output'/Path(c['path']).parent
 responses=[]
 for path in sorted((root/'responses').glob('*.json')):
  r=json.loads(path.read_text());responses.append({'path':str(path.relative_to(run)),'reasoning':r.get('reasoning') or '', 'tool_calls':r.get('tool_calls',[])})
 tools=[json.loads(line) for line in (root/'tool-trace.jsonl').read_text().splitlines()] if (root/'tool-trace.jsonl').exists() else []
 # Keep exact reasoning and trace references outside the compact audit/report.
 records.append({'case':c['path'],'packet':c['packet'],'arm':c['arm'],'responses':responses,
                 'tool_errors':[r for r in tools if r.get('error_stage') or r.get('error')],
                 'tool_trace_path':str((root/'tool-trace.jsonl').relative_to(run)),
                 'action_rationale':(c['action'] or {}).get('rationale')})
(run/'REASONING-INDEX.json').write_text(json.dumps(records,indent=2)+'\n')
text=['# Repaired tractability probe: automated outcome audit','',
      'This report reconciles resource admission and engine outcomes. Mechanistic interpretation of model reasoning still requires reading REASONING-INDEX.json and the raw tool traces. No model rerun or promotion is performed by this process.','',
      f"Audit valid: {audit['valid']}; retained cases: {audit['cases']}; collected files hash-checked: {audit['collected_files_hash_checked']}.",'',
      '| Arm | Stalled actions | Stalled levels completed | Final admission failures | Easy control solves |',
      '|---|---:|---:|---:|---:|']
for arm in ('baseline','terminal','time','both'):
 s=audit['stalls'].get(arm,{});ct=audit['controls'].get(arm,{})
 text.append(f"| {arm} | {s.get('actions',0)}/{s.get('cases',0)} | {s.get('levels_completed',0)} | {s.get('final_admission_rejections',0)} | {ct.get('level_completions',0)}/{ct.get('cases',0)} |")
text+=['','Audit failures: '+(('; '.join(audit['failures'])) or 'none'),'',
       'Interpretation: a verified lack of final admission failures establishes the corrected resource path, not useful solving. Actions without completed stalled levels demonstrate commitment only. A failure to emit an action must be traced to final provider receipts, deadline/refusal or tool validation. Historical totals are descriptive because all arms now reserve final tokens; compare contemporaneous arms. First committed batches and two replicas do not decide general solvability.','',
       'Evidence: AUDIT.json, REASONING-INDEX.json, summary.json, output/tractability-output and frozen EXPERIMENT-CARD.md. Notebook duration is not exact billed GPU allocation; preserve the full cap until an unconfounded receipt exists.']
(run/'AUTOMATIC-RESULT.md').write_text('\n'.join(text)+'\n')
print('Outcome audit and reasoning index written; no launch/retry/promotion')
