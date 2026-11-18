"""Read-only local candidate/provenance verification; never accesses auth/network."""
import hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'engine')]
from run_pair import verify_inventory
from prefix import project_transitions
from constructor import construction_messages
from transport import wire, digest

def main():
    inventory=verify_inventory()
    bindings=json.loads((HERE/'SOURCE-BINDINGS.json').read_text())
    for b in bindings.values():assert digest(Path(b['source']).read_bytes())==b['sha256']
    prefix=json.loads((HERE/'PREFIX-BINDING.json').read_text())
    assert digest((HERE/'prefix.jsonl').read_bytes())==prefix['prefix_sha256']
    case=project_transitions(HERE/'prefix.jsonl')
    original=json.loads((HERE/'ORIGINAL-FIRST-MESSAGES.json').read_text())
    messages=construction_messages(case,None)
    assert messages[1]==original[1]
    assert case.frontier==1 and case.visible[0]['action']=='RIGHT'
    assert not (HERE/'live').exists(), 'Consumed state present; do not rerun'
    print(json.dumps({'candidate':'VERIFIED_LOCAL_PREPARATION','inventory_sha256':inventory,'donor_sources_unchanged':len(bindings),'prefix':'step 1 RIGHT, exact visible packet','paid_calls':0,'approvals':'external gates still required'},indent=2))
if __name__=='__main__':main()
