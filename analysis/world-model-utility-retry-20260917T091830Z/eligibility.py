"""Read-only full-horizon arithmetic and imported-source eligibility, without credentials."""
from decimal import Decimal
from common import HERE,CAPS,CONTEXT_RESERVATION,read,hash_file
from run_utility import validate_donor
from transport import RATES,MAX_PROVIDER_COMPLETION

def eligibility():
 f=validate_donor(HERE/'donor/prefix.jsonl')
 per_request=CONTEXT_RESERVATION*RATES['prompt']+1024*RATES['completion']
 nominal=per_request*240
 exposure=CONTEXT_RESERVATION*RATES['prompt']+MAX_PROVIDER_COMPLETION*RATES['completion']
 assert nominal+exposure<=Decimal('2')
 assert all(c['prompt']>=120*CONTEXT_RESERVATION and c['completion']==120*1024 and Decimal(c['allocation'])>=120*per_request for c in CAPS.values())
 return {'status':'ELIGIBLE_BOUNDED_CONTEXT_NOT_GUARANTEED_COMPLETION','new_pair_cap_usd':'2','per_request_reserved_prompt_tokens':CONTEXT_RESERVATION,'per_actor_prompt_tokens':120*CONTEXT_RESERVATION,'per_actor_completion_tokens':122880,'nominal_pair_max_usd':str(nominal),'full_provider_inflight_reservation_usd':str(exposure),'conservative_nominal_plus_extra_reservation_usd':str(nominal+exposure),'requests_admissible_at_max_usage':240,'freeze_sha256':hash_file(HERE/'donor/FROZEN.json'),'source_digest':f['source_digest'],'prefix_sha256':f['prefix_sha256'],'old_known_cost_usd':'0.170458010','old_unknown_exposure_usd':'0.345637232','old_exposure_separate':True,'limitation':'Budget permits all120 requests per actor for context<=20000 serialized UTF8 bytes+4096 and valid reported rates/usage. Context overflow, time, format, provider failures or official terminal can stop earlier. No claim that every future grid/prediction fits.'}
if __name__=='__main__':
 import json
 print(json.dumps(eligibility(),indent=2))
