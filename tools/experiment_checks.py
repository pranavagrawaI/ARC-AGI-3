"""Offline evidence checks. No model calls, helper execution, or experiment mutation.

Exit 0 = scoped claim supported, 1 = contradicted/invalid, 2 = unresolved.
These checks never promote a model or turn a probe's verdict.json into evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import stat


def count(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def strict_json(text):
    def invalid_constant(value):
        raise ValueError(f'non-finite JSON number: {value}')
    return json.loads(text, parse_constant=invalid_constant)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Evidence:
    def __init__(self):
        self.files = {}
        self.missing = []

    def read(self, path):
        path = Path(path)
        try:
            if not path.is_file():
                raise ValueError('missing or non-regular evidence file')
            data = path.read_bytes()
            self.files[str(path)] = hashlib.sha256(data).hexdigest()
            return data.decode()
        except (OSError, UnicodeError, ValueError) as exc:
            self.missing.append(f'{path}: {exc}')
            return None

    def obj(self, path):
        text = self.read(path)
        if text is None:
            return None
        try:
            value = strict_json(text)
            if not isinstance(value, dict):
                raise ValueError('expected JSON object')
            return value
        except ValueError as exc:
            self.missing.append(f'{path}: {exc}')
            return None

    def rows(self, path):
        text = self.read(path)
        rows = []
        for line, raw in enumerate((text or '').split('\n'), 1):
            if not raw.strip():
                continue
            try:
                value = strict_json(raw)
                if not isinstance(value, dict):
                    raise ValueError('expected JSON object')
                rows.append((f'{path}:{line}', value))
            except ValueError as exc:
                self.missing.append(f'{path}:{line}: {exc}')
        return rows

    def report(self, claim, failures, pending=(), **observations):
        unresolved = list(pending) + self.missing
        return dict(claim=claim, status='invalid' if failures else
                    'unresolved' if unresolved else 'supported',
                    failures=failures, unresolved=unresolved,
                    observations=observations, evidence_sha256=self.files)


def flash(raw):
    """Necessary checks for the historical Flash-Next card, not a general card parser."""
    raw = Path(raw)
    ev = Evidence()
    card = ev.read(raw / 'EXPERIMENT-CARD.md')
    failures = []
    # Card interpretation remains explicit. Never silently apply these rules to another card.
    invalid_clause = ('Invalid if: Wrong model/runtime/GPU identity, missing request/response '
                      'accounting, timeout, OOM, failed cleanup, or missing terminal stream/usage.')
    pending = ['Model/runtime/GPU identity requires comparison with the frozen provenance pins.',
               'Structured-response semantics and all execution caps require review against the card.']
    if card is None or invalid_clause not in card:
        return ev.report('Flash-Next operational validity', [],
                         ['Unsupported card; no automatic interpretation of changed rules.'])
    log = ev.read(raw / 'vllm-openai-server.log')
    oom_hits = []
    # Progress bars contain CR characters; only LF advances an editor/rg line number.
    for line, text in enumerate((log or '').split('\n'), 1):
        if re.search(r'allocation failed with OOM|CUDA out of memory|OutOfMemoryError', text, re.I):
            oom_hits.append(dict(source=f'{raw / "vllm-openai-server.log"}:{line}', text=text))
    if oom_hits:
        failures.append(dict(condition='OOM', clause=invalid_clause, evidence=oom_hits))
    if log is not None and not log.strip():
        pending.append('Server log is empty; OOM absence cannot be checked.')
    cleanup = ev.obj(raw / 'cleanup.json')
    if cleanup is not None:
        required = {'stopped', 'survivors', 'port_open', 'identity_matches'}
        if not required <= cleanup.keys():
            pending.append('Cleanup evidence lacks required fields.')
        elif (cleanup['stopped'] is not True or cleanup['survivors'] != []
              or cleanup['port_open'] is not False or cleanup['identity_matches'] is not True):
            failures.append(dict(condition='failed cleanup', source=str(raw / 'cleanup.json')))
    results = ev.rows(raw / 'results.jsonl')
    names = [row.get('name') for _, row in results if isinstance(row.get('name'), str)]
    expected = {'text', 'tool', 'json', 'actor-case-0', 'actor-case-1',
                'actor-case-5', 'actor-case-12', 'actor-case-13'}
    if len(names) != len(results) or len(names) != len(set(names)):
        pending.append('Results index has missing or duplicate response names.')
    indexed = {}
    for source, row in results:
        name = row.get('name')
        if not isinstance(name, str) or name not in expected:
            pending.append(f'{source}: missing/unknown response name')
            continue
        indexed[name] = (source, row)
    # A failed gate prevents a results.jsonl row, but its issued response still exists.
    # Inventory raw artifacts before judging accounting or planned coverage.
    issued = sorted(name for name in expected if name in indexed or any(
        (raw / f'{name}{suffix}').exists() for suffix in ['.request.json', '.response.json', '.sse']))
    for path in sorted(raw.glob('*.request.json')):
        if path.name.removesuffix('.request.json') not in expected:
            pending.append(f'{path}: unexpected issued request')
    if set(issued) != expected:
        pending.append(f'Planned request coverage incomplete ({len(issued)}/8 issued); '
                       'unissued cases are not lost response accounting.')
    response_count = 0
    for name in issued:
        response_path = raw / f'{name}.response.json'
        row = ev.obj(response_path)
        if row is None:
            row = indexed.get(name, ('', {}))[1]
        else:
            response_count += 1
        source = str(response_path)
        if name in indexed:
            index_source, index_row = indexed[name]
            fields = ['name', 'usage', 'finish_reason', 'done', 'content', 'reasoning', 'calls', 'request_sha256']
            if any(row.get(key) != index_row.get(key) for key in fields):
                failures.append(dict(condition='response/result accounting mismatch', source=index_source))
        request_path = raw / f'{name}.request.json'
        request_text = ev.read(request_path)
        if request_text is not None and ev.files[str(request_path)] != row.get('request_sha256'):
            failures.append(dict(condition='request accounting hash mismatch', source=source))
        stream = ev.read(raw / f'{name}.sse')
        if stream is not None:
            done = False
            usage = None
            finish = None
            for line, text in enumerate(stream.splitlines(), 1):
                if not text.startswith('data: '):
                    continue
                data = text[6:].strip()
                if data == '[DONE]':
                    done = True
                    break
                try:
                    event = strict_json(data)
                    usage = event.get('usage') or usage
                    for choice in event.get('choices', []):
                        finish = choice.get('finish_reason') or finish
                except (ValueError, AttributeError, TypeError):
                    pending.append(f'{name}.sse:{line}: malformed stream event')
            if (not done or not isinstance(finish, str) or not finish
                    or not isinstance(usage, dict) or not count(usage.get('prompt_tokens'))
                    or not count(usage.get('completion_tokens'))):
                failures.append(dict(condition='missing terminal stream/usage',
                                     source=str(raw / f'{name}.sse')))
            elif usage != row.get('usage') or finish != row.get('finish_reason'):
                failures.append(dict(condition='stream/result accounting mismatch', source=source))
    for failure_path in sorted(raw.glob('*.failure.json')) + [raw / 'failure.json']:
        if failure_path.exists():
            failure = ev.read(failure_path)
            if re.search(r'TimeoutError|timed out', failure or '', re.I):
                failures.append(dict(condition='timeout', source=str(failure_path)))
            else:
                pending.append(f'{failure_path}: classify recorded failure against the card.')
    provenance = ev.obj(raw / 'vllm-setup-provenance.json') or {}
    identity = ev.obj(raw / 'vllm-server-identity.json') or {}
    ready = identity.get('ready_epoch')
    if not number(ready):
        pending.append('Missing/invalid readiness epoch; serving samples cannot be classified.')
        ready = None
    runtime = provenance.get('runtime')
    if not isinstance(runtime, dict):
        pending.append('Missing/invalid runtime provenance object.')
        runtime = {}
    samples = []
    for source, row in ev.rows(raw / 'resources.jsonl'):
        try:
            if not number(row.get('epoch')):
                raise ValueError('missing/invalid sample epoch')
            for gpu in row['gpu'].strip().splitlines():
                _, used, total, _ = gpu.rsplit(',', 3)
                free = int(total.strip().split()[0]) - int(used.strip().split()[0])
                samples.append(dict(source=source, epoch=row['epoch'], free_mib=free))
        except (KeyError, ValueError, AttributeError, IndexError):
            pending.append(f'{source}: unusable GPU sample')
    serving = [s for s in samples if ready is not None and s['epoch'] >= ready]
    readiness = next((v['ready_seconds'] for v in provenance.values()
                      if isinstance(v, dict) and 'ready_seconds' in v), None)
    return ev.report('Flash-Next operational validity', failures, pending,
                     issued_request_count=len(issued), response_count=response_count,
                     indexed_results_count=len(results), unissued_cases=sorted(expected - set(issued)), oom=oom_hits,
                     setup_seconds=provenance.get('setup_elapsed_seconds'),
                     readiness_seconds=readiness,
                     extraction_seconds=runtime.get('extract_seconds'),
                     all_run_sampled_min_free_mib=min((s['free_mib'] for s in samples), default=None),
                     post_readiness_samples=serving,
                     instantaneous_serving_minimum='unmeasured')


def batch_gate(proposal, feedback):
    """Return no eligibility when any prerequisite is absent; missing data stays unknown."""
    if feedback.get('batch_allowed') is False or proposal.get('status') not in (None, 'committed'):
        return False
    if (feedback.get('batch_allowed') is not True or proposal.get('status') != 'committed'
            or not proposal.get('model_sha256') or not feedback.get('model_sha256')):
        return None
    return proposal['model_sha256'] == feedback['model_sha256']


def batching(ipc):
    ipc = Path(ipc)
    ev = Evidence()
    feedback = ev.rows(ipc / 'model_feedback.jsonl')
    predictions = ev.rows(ipc / 'model_predictions.jsonl')
    actions = ev.rows(ipc / 'actions.jsonl')
    stops = ev.rows(ipc / 'batch_stops.jsonl')
    pending = []
    if not feedback:
        pending.append('No feedback observations; absence does not establish ineligibility.')
    if any(type(r.get('batch_allowed')) is not bool for _, r in feedback):
        pending.append('Feedback rows with missing/non-boolean permission.')
    valid_actions = []
    for source, row in actions:
        if (not isinstance(row.get('actions'), list) or not row['actions']
                or not count(row.get('turn')) or not count(row.get('step'))):
            pending.append(f'{source}: invalid submission actions/turn/step')
        else:
            valid_actions.append((source, row))
    valid_predictions = []
    for source, row in predictions:
        if (not isinstance(row.get('actions'), list) or not count(row.get('turn'))
                or not count(row.get('step'))):
            pending.append(f'{source}: invalid prediction actions/turn/step')
        else:
            valid_predictions.append((source, row))
    for source, row in feedback:
        # These are valid runtime records when no executable helper/commit exists.
        if row.get('status') in ('unmeasured', 'unavailable') and row.get('batch_allowed') is False:
            continue
        if (not count(row.get('turn')) or not count(row.get('start_step'))
                or not isinstance(row.get('model_sha256'), str) or not row['model_sha256']):
            pending.append(f'{source}: missing/invalid feedback turn/step/helper hash')
    true_rows = [dict(source=s, turn=r.get('turn'), start_step=r.get('start_step'),
                      helper_sha256=r.get('model_sha256')) for s, r in feedback
                 if r.get('batch_allowed') is True]
    # Files lack a common sequence number: never use a later check to license an earlier choice.
    ledger = []
    for source, action in valid_actions:
        matches = [(s, p) for s, p in valid_predictions if p.get('turn') == action.get('turn')
                   and p.get('step') == action.get('step') and p.get('actions') == action.get('actions')]
        ledger.append(dict(source=source, turn=action.get('turn'), step=action.get('step'),
                           actions=action.get('actions'),
                           committed_predictions=[dict(source=s, status=p.get('status'),
                                                       helper_sha256=p.get('model_sha256')) for s, p in matches],
                           eligibility_at_submission='unresolved: no atomic gate snapshot',
                           stops=[dict(r, source=s) for s, r in stops
                                  if r.get('turn') == action.get('turn')]))
    return ev.report('Recorded batching permission inventory', [], pending,
                     feedback_rows=len(feedback), permission_true=true_rows,
                     distinct_helper_versions=len({r['model_sha256'] for _, r in feedback
                                                   if isinstance(r.get('model_sha256'), str) and r['model_sha256']}),
                     batch_submissions=sum(len(r['actions']) > 1 for _, r in valid_actions),
                     malformed_submission_rows=len(actions) - len(valid_actions),
                     declined_opportunities=None,
                     opportunity_limitation='Permission rows are not a count of executable opportunities. '
                     'Historical logs lack an atomic proposal/feedback/hash snapshot at each decision.',
                     submissions=ledger)


def distribution(values):
    ordered = sorted(values)
    if not ordered:
        return None
    n = len(ordered)
    return dict(count=n, minimum=ordered[0], median=(ordered[(n-1)//2]+ordered[n//2])/2,
                p90=ordered[math.ceil(n*.9)-1], maximum=ordered[-1], p90_method='nearest rank')


def coverage(raw, calls, ceiling, output_tokens):
    ev = Evidence()
    fixtures = []
    for source, row in ev.rows(Path(raw) / 'results.jsonl'):
        if not isinstance(row.get('name'), str):
            ev.missing.append(f'{source}: missing/invalid fixture name')
            continue
        if not row['name'].startswith('actor-'):
            continue
        usage = row.get('usage')
        if not isinstance(usage, dict) or not count(usage.get('prompt_tokens')):
            ev.missing.append(f'{source}: missing/invalid fixture prompt token count')
        else:
            fixtures.append((source, usage['prompt_tokens']))
    live = []
    for path in calls:
        for source, row in ev.rows(path):
            if row.get('request_purpose') != 'actor':
                continue
            tokens = row.get('prompt_tokens')
            if not count(tokens):
                ev.missing.append(f'{source}: missing/invalid Actor prompt token count')
            else:
                live.append((source, tokens))
    pending = ['Counts are provider-reported. Retokenize exact requests with the candidate tokenizer '
               'and chat template before claiming candidate context compatibility.']
    if not fixtures or not live:
        pending.append('Both fixture and live Actor counts are required.')
    tail = [dict(source=s, prompt_tokens=n) for s, n in live if n + output_tokens > ceiling]
    failures = []
    if fixtures and live and max(n for _, n in fixtures) < max(n for _, n in live):
        failures.append(dict(condition='fixtures do not reach the reported live maximum',
                             fixture_max=max(n for _, n in fixtures), live_max=max(n for _, n in live)))
    return ev.report('Coverage of the reported live prompt distribution', failures, pending,
                     fixtures=distribution([n for _, n in fixtures]),
                     live=distribution([n for _, n in live]), ceiling=ceiling,
                     output_reservation=output_tokens,
                     prompt_only_exceeds=sum(n > ceiling for _, n in live),
                     prompt_plus_output_exceeds=tail,
                     proposed_selection='Retain exact requests at median, p90, maximum and both sides '
                     'of the candidate context boundary; preserve request hashes and tokenizer/template identity.')


def preserve(source, destination):
    """Copy evidence into an exclusively created directory; retain failures for inspection."""
    source, destination = Path(source).resolve(), Path(destination).absolute()
    if not source.is_dir():
        raise ValueError('source must be an existing directory')
    if destination.resolve().is_relative_to(source):
        raise ValueError('destination must be outside the source')
    files = sorted(source.rglob('*'))
    if any(not (stat.S_ISREG(p.lstat().st_mode) or stat.S_ISDIR(p.lstat().st_mode)) for p in files):
        raise ValueError('only regular files and directories are accepted as preserved evidence')
    if (source / 'EVIDENCE-MANIFEST.json').exists():
        raise ValueError('source already contains the reserved manifest name')
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {}
    for path in files:
        target = destination / path.relative_to(source)
        if path.is_dir():
            target.mkdir()
        elif path.is_file():
            before = sha(path)
            with path.open('rb') as src, target.open('xb') as dst:
                shutil.copyfileobj(src, dst)
            if sha(target) != before or sha(path) != before:
                raise ValueError(f'source changed during preservation: {path}')
            manifest[str(path.relative_to(source))] = before
    with (destination / 'EVIDENCE-MANIFEST.json').open('x') as output:
        json.dump(dict(source=str(source), files=manifest), output, indent=2)
        output.write('\n')
    return dict(status='supported', claim='Evidence snapshot copied without overwriting',
                destination=str(destination), files=len(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New JSON report; refuses an existing file')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('flash').add_argument('raw', type=Path)
    sub.add_parser('batching').add_argument('ipc', type=Path)
    cov = sub.add_parser('coverage')
    cov.add_argument('raw', type=Path)
    cov.add_argument('calls', type=Path, nargs='+')
    cov.add_argument('--ceiling', type=int, required=True)
    cov.add_argument('--output-tokens', type=int, required=True)
    copy = sub.add_parser('preserve')
    copy.add_argument('source', type=Path)
    copy.add_argument('destination', type=Path)
    args = parser.parse_args()
    if args.command == 'flash':
        result = flash(args.raw)
    elif args.command == 'batching':
        result = batching(args.ipc)
    elif args.command == 'coverage':
        if args.ceiling <= 0 or args.output_tokens < 0:
            parser.error('ceiling must be positive and output-tokens nonnegative')
        result = coverage(args.raw, args.calls, args.ceiling, args.output_tokens)
    else:
        result = preserve(args.source, args.destination)
    if args.output:
        with args.output.open('x') as output:
            json.dump(result, output, indent=2)
            output.write('\n')
    else:
        print(json.dumps(result, indent=2))
    return {'supported': 0, 'invalid': 1, 'unresolved': 2}[result['status']]


if __name__ == '__main__':
    raise SystemExit(main())
