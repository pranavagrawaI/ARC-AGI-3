"""Exact-byte local reference construction. No empirical count offset."""
import hashlib,json

def digest(raw):return hashlib.sha256(raw).hexdigest()
def check_prepared(raw,record):
    if digest(raw)!=record['prepared_body_sha256']:
        raise ValueError('exact prepared HTTP body bytes/order mismatch')
    return json.loads(raw)
