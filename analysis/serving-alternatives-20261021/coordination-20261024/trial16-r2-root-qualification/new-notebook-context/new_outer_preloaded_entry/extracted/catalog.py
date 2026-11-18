"""Accepted exact-HTTP-byte local reference; no empirical count adjustment."""
import json
from pathlib import Path
from corpus_base import canonical,digest,image_observations,payload_for,verify_corpus,wire_hash
HERE=Path(__file__).resolve().parent
CATALOG_SHA='edecabdd3418d2b0305ce44e0958459dee71cbdd836a437f60ee80ca51a9005e'

def load(path=HERE/'catalog.json'):
    raw=Path(path).read_bytes()
    if digest(raw)!=CATALOG_SHA:raise ValueError('exact-wire catalog identity')
    return json.loads(raw)

def checked_body(raw,record):
    if digest(raw)!=record['prepared_body_sha256']:
        raise ValueError('exact prepared/outbound HTTP body bytes/order mismatch')
    body=json.loads(raw)
    if wire_hash(body)!=record['wire_sha'] or image_observations(body)!=record['images']:
        raise ValueError('exact body corpus/source PNG mismatch')
    return body
