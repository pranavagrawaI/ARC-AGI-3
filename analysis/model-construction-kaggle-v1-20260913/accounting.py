"""Receipt accounting: absent or incomplete evidence is not zero spend."""
from decimal import Decimal, InvalidOperation

def receipts(responses, attempted):
    validated = [x for x in responses if isinstance(x.get('usage'), dict) and all(type(x['usage'].get(k)) is int and x['usage'][k] >= 0 for k in ('prompt_tokens','completion_tokens'))]
    valid = (bool(responses) or attempted == 0) and len(validated) == len(responses)
    complete = valid and len(responses) == attempted
    known = {k: sum(x['usage'][k] for x in validated) if validated or attempted == 0 else None for k in ('prompt_tokens','completion_tokens')}
    cost = None
    if valid:
        try:
            costs = [Decimal(str(x['usage']['cost'])) for x in responses]
            if all(c.is_finite() and c >= 0 for c in costs): cost = sum(costs, Decimal(0))
        except (KeyError, InvalidOperation, ValueError): pass
    return {'complete': complete, 'valid_receipts': valid, **{k: v if complete else None for k,v in known.items()},
            'known_successful_usage': known, 'cost': cost if complete else None, 'known_successful_cost': cost}
