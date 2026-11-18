"""Bounded HTTP error evidence. Never serialize request/auth/cookie headers."""
import hashlib
import re

SAFE_HEADERS = {'content-type', 'content-length', 'date', 'retry-after', 'x-request-id', 'x-kaggle-request-id'}
BODY_LIMIT = 65536


def redact(text):
    text = re.sub(r'(?i)(https?://[^\s\"<>?]+)\?[^\s\"<>]*', r'\1?[REDACTED]', text)
    text = re.sub(r'(?i)\b(Bearer|Basic)\s+[A-Za-z0-9_.+/=-]+', r'\1 [REDACTED]', text)
    return re.sub(r'(?i)([\"\x27]?(?:authorization|api[_-]?key|access[_-]?token|token|password|secret)[\"\x27]?\s*[:=]\s*)([\"\x27])?([^\s\",;\x27}]+)',
                  r'\1\2[REDACTED]', text)


def exception_record(error):
    record = dict(error_type=type(error).__name__, message=redact(str(error)), retry_allowed=False)
    response = getattr(error, 'response', None)
    if response is None:
        record['http_response'] = 'UNAVAILABLE'
        return record
    body = response.content
    bounded = body[:BODY_LIMIT]
    record.update(status=int(response.status_code),
                  headers={k: redact(str(v)) for k, v in response.headers.items() if k.lower() in SAFE_HEADERS},
                  body=redact(bounded.decode('utf-8', errors='replace')),
                  body_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest(),
                  body_truncated=len(body)>BODY_LIMIT, body_limit_bytes=BODY_LIMIT,
                  body_policy='bounded UTF-8; credential/query redaction; SHA256 over original complete buffered response')
    return record
