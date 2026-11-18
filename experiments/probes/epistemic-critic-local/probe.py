"""Fresh-context critic screen. Standard library only; no gameplay or retries."""
import argparse
import hashlib
import ipaddress
import json
from pathlib import Path
import time
import urllib.parse
import urllib.request

HERE = Path(__file__).resolve().parent
PROMPT = '''You are an independent epistemic critic of a game-playing Actor.
You have a fresh context. Treat supplied transitions as observations and Actor prose
as claims. Assess whether the proposed commitment is warranted at its stated scope.
You need not solve the game. Distinguish prediction agreement from objective progress,
search limits from impossibility, and model failure from failure of its premises.
Uncertainty may remain while a bounded probe proceeds. Do not invent risks or insist
on a complete theory. You may allow, challenge, or report insufficient evidence.
Give at most one material challenge. Do not assume there must be a defect.
For a challenge cite evidence, explain the unsupported inference, and suggest a
concrete bounded revision or distinguishing observation. Do not claim the alternative
is true. For allow, preserve the Actor's justified uncertainty and commitment.
Return only JSON with keys verdict (allow|challenge|insufficient_evidence),
evidence_refs (list of supplied reference strings), issue (string), and
next_step (string). No tools, actions, or code execution are available.'''


def requests(model):
    packets = json.loads((HERE / 'cases.json').read_text())
    return [(p['id'], {'model': model, 'messages': [
        {'role': 'system', 'content': PROMPT},
        {'role': 'user', 'content': json.dumps({k: p[k] for k in ('evidence', 'actor_claim', 'actor_proposal')})}],
        'temperature': 0, 'max_tokens': 1024, 'stream': False,
        'chat_template_kwargs': {'enable_thinking': False}}) for p in packets]


def validate_reply(content, refs):
    try:
        obj = json.loads(content)
    except (TypeError, ValueError):
        return ['response is not a JSON object']
    if not isinstance(obj, dict):
        return ['response is not a JSON object']
    errors = []
    if obj.get('verdict') not in ('allow', 'challenge', 'insufficient_evidence'):
        errors.append('invalid verdict')
    cited = obj.get('evidence_refs')
    if not isinstance(cited, list) or any(not isinstance(x, str) or x not in refs for x in cited):
        errors.append('unknown or malformed evidence references')
    for k in ('issue', 'next_step'):
        if not isinstance(obj.get(k), str):
            errors.append(f'{k} must be a string')
    if obj.get('verdict') == 'challenge' and (not cited or not obj.get('issue') or not obj.get('next_step')):
        errors.append('challenge requires evidence, issue and next_step')
    return errors


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('redirects disabled for local-only probe')


def local_endpoint(url):
    p = urllib.parse.urlsplit(url)
    try:
        loopback = ipaddress.ip_address(p.hostname or '').is_loopback
    except ValueError:
        loopback = False
    if p.scheme != 'http' or not loopback or p.username or p.password or p.query or p.fragment:
        raise ValueError('Use a literal loopback HTTP URL such as http://127.0.0.1:8000/v1')
    return url.rstrip('/')


def run(out, model, endpoint=None):
    # Exclusive directory creation preserves failed attempts; choose a new ID to retry.
    if endpoint:
        endpoint = local_endpoint(endpoint)
    out.mkdir(parents=True, exist_ok=False)
    prepared = requests(model)
    for id, body in prepared:
        (out / f'{id}.request.json').write_text(json.dumps(body, indent=2) + '\n')
    manifest = {'status': 'prepared', 'model_requested': model, 'endpoint': endpoint,
                'cases': len(prepared), 'call_cap': 8, 'timeout_seconds_per_call': 30,
                'retries': 0, 'game_actions': 0, 'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
                'cases_sha256': hashlib.sha256((HERE / 'cases.json').read_bytes()).hexdigest(),
                'scope': 'diagnostic screen only; no Actor continuation or gameplay efficacy'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not endpoint:
        return manifest
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    try:
        with opener.open(endpoint + '/models', timeout=5) as r:
            models = json.load(r)
        (out / 'models.json').write_text(json.dumps(models, indent=2) + '\n')
        if model not in [m['id'] for m in models['data']]:
            raise ValueError('requested model absent from local server listing')
        manifest['status'] = 'completed'
        for id, body in prepared:
            start = time.monotonic()
            req = urllib.request.Request(endpoint + '/chat/completions', json.dumps(body).encode(),
                                         {'Content-Type': 'application/json'})
            with opener.open(req, timeout=30) as r:
                raw = r.read().decode()
            (out / f'{id}.response.json').write_text(raw)
            response = json.loads(raw)
            message = response['choices'][0]['message']
            refs = {e['ref'] for e in json.loads(body['messages'][1]['content'])['evidence']}
            check = {'elapsed_seconds': time.monotonic() - start,
                     'format_errors': validate_reply(message.get('content'), refs),
                     'semantic_grade': 'pending human review against rubric.json',
                     'usage': response.get('usage'), 'returned_model': response.get('model')}
            (out / f'{id}.check.json').write_text(json.dumps(check, indent=2) + '\n')
            if message.get('reasoning_content') or response['choices'][0].get('finish_reason') != 'stop':
                raise ValueError('reasoning output or unfinished completion; run invalid')
    except Exception as e:
        manifest.update(status='aborted', error=f'{type(e).__name__}: {e}')
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', required=True, help='Exact local served model ID; never substitutes another model')
    parser.add_argument('--endpoint', help='Omit to prepare offline; loopback HTTP only')
    args = parser.parse_args()
    result = run(args.output, args.model, args.endpoint)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result['status'] == 'aborted' else 0)
