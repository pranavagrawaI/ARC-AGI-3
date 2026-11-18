"""Local provenance preparation only. No network, auth, or model calls."""
import hashlib, json, shutil, sys
from pathlib import Path
P=Path(__file__).resolve().parent
R=P.parent/'r10-prep-20260916'
S=R/'r10a-package'
E=P/'engine';E.mkdir(exist_ok=True)
names=['constructor.py','constructor_contract.py','request_contract.py','real_trace.py','integration_boundary.py','containment_base.py','qualified_runner.py','inspection_runner.py','accelerator.py','evaluate_visible.py','prefix.py','observation_abi.py','local_wire.py','host.py','persist.py']
orig={}
for name in names:
 raw=(S/name).read_bytes();(E/name).write_bytes(raw);orig[name]={'source':str((S/name).resolve()),'sha256':hashlib.sha256(raw).hexdigest()}
(P/'SOURCE-BINDINGS.json').write_text(json.dumps(orig,indent=2)+'\n')
(P/'rulings.md').write_bytes(Path('/tmp/or-constructor-pair-rulings.md').read_bytes())
path=R/'exact-collection-20260916/raw/model-runtime/solver-workspaces/arm-B/cd82-fb555c5d-b7azqbdf/ipc/transitions.jsonl'
raw=path.read_bytes();first=raw.splitlines(keepends=True)[0]
(P/'prefix.jsonl').write_bytes(first)
sys.path.insert(0,str(E));from prefix import project_transitions
case=project_transitions(P/'prefix.jsonl')
wirepath=R/'exact-collection-20260916/raw/model-runtime/solver-workspaces/raw/raw-requests/074.wire.json'
wire=json.loads(wirepath.read_text())
assert case.frontier==1 and case.visible[0]['action']=='RIGHT'
assert case.compact_view()==json.loads(wire['messages'][1]['content'])['visible_case']
(P/'PREFIX-BINDING.json').write_text(json.dumps({'source':str(path.resolve()),'source_sha256':hashlib.sha256(raw).hexdigest(),'selected':'first line only, step 1 RIGHT','prefix_sha256':hashlib.sha256(first).hexdigest(),'comparison_wire':str(wirepath.resolve()),'comparison_wire_sha256':hashlib.sha256(wirepath.read_bytes()).hexdigest(),'original_visible_packet_equal':True},indent=2)+'\n')
# Explicit small shared protocol adaptation, never source-format repair.
p=E/'constructor_contract.py';s=p.read_text()
s=s.replace('NOT billed cost for this local GPU run.', 'NOT billed cost for this hosted run. Actual endpoint exposure is reserved separately.')
s=s.replace('Missing/invalid usage or returned reasoning is fatal.', 'Missing/invalid usage is fatal. Reasoning policy is controlled by the API; returned reasoning is preserved separately from tool arguments.')
# Replace stale Kaggle relay/game envelope with shared hosted transport envelope.
a=s.index('        "package_envelope": {');b=s.index('\n    }',a)
s=s[:a]+'''        "package_envelope": {
            "constructor_hard_seconds": 245, "constructor_soft_seconds": 240,
            "constructor_requests": 6, "pair_dollars": 2,
            "completion_tokens_per_request_including_reasoning": 4096,
            "transport": "OpenRouter hosted FP8; reasoning policy controlled by API",
            "http_post_seconds": "remaining local soft deadline; no retries",
        },
        "scope": "Requested combined output cap and observed validity threshold; provider hard stop not guaranteed. Local deadline does not establish remote cancellation. No gameplay or suffix assessment.",''' + s[b:]
p.write_text(s)
p=E/'constructor.py';s=p.read_text();s=s.replace('from request_contract import validate_request','from request_contract import validate_request\nfrom protocol import assistant_message')
s=s.replace("'seconds_remaining': round(seconds, 3)","'seconds_remaining': 240 if result.model_requests == 0 else round(seconds, 3)")
a=s.index("        if (usage.get('completion_tokens_details')");b=s.index('        choices =',a);s=s[:a]+s[b:]
a=s.index("        if any(assistant.get(k)");b=s.index('        messages.append(outbound)',a)
s=s[:a]+"        outbound = assistant_message(assistant)\n"+s[b:]
p.write_text(s)
# Keep existing request validation except allow protocol reasoning continuation.
p=E/'request_contract.py';s=p.read_text().replace('from local_wire import project_messages','from protocol import project_messages');p.write_text(s)
(P/'ORIGINAL-FIRST-MESSAGES.json').write_text(json.dumps(wire['messages'][:3],indent=2)+'\n')
import difflib
(P/'engine.diff').write_text(''.join(''.join(difflib.unified_diff((S/n).read_text().splitlines(True),(E/n).read_text().splitlines(True),fromfile='r10a/'+n,tofile='or-pair/'+n)) for n in names))
print('Bound step 1 RIGHT matches original constructor-visible packet; copied',len(names),'engine files.')
