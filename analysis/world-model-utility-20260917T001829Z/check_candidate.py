"""Read-only candidate verification; no credential load, remote call or admission."""
from common import HERE,read,hash_file
from run_utility import verify_inventory
import json
sha=verify_inventory()
d=read(HERE/'discovery/RECEIPT.json');e=d['selected']
assert d['status']==200 and d['sha256']==hash_file(HERE/'discovery/endpoints.raw.json')
assert e['tag']=='reka/fp8' and e['max_completion_tokens']==131072 and e['context_length']==262144
assert e['pricing']['prompt']=='0.000000214' and e['pricing']['completion']=='0.00000255'
assert {'reasoning','tools','tool_choice','max_tokens'}<=set(e['supported_parameters'])
assert not (HERE/'live').exists(),'partial/consumed state; no retry'
print(json.dumps({'status':'LOCAL_PREPARATION_VERIFIED_NOT_RELEASED','inventory_sha256':sha,'live_absent':True,'paid_calls':0,'credentials_accessed':False}))
