#!/usr/bin/env python3
"""Local sealed evidence, not provider output. Writes only stdout; never launches."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PACKAGE = ROOT / 'deployment/kaggle/actor-critic-construction-kaggle-r6'
sys.path.insert(0, str(PACKAGE))
sys.path.insert(0, str(HERE / 'tests'))

from test_constructor_contract import CASE, VALID, assert_outbound_contract  # noqa: E402
from test_runtime import WORK  # noqa: E402
from host import bound_constructor, ConstructionLimits  # noqa: E402
from evaluate_visible import evaluate_visible_abi  # noqa: E402
from prefix import ActorPrefixCase  # noqa: E402
from integration_boundary import SealedBoundary  # noqa: E402


def collect():
    case = ActorPrefixCase((CASE,), 1)
    outbound = []

    def transport(body):
        assert_outbound_contract(body)
        outbound.append(body)
        source = VALID if len(outbound) == 2 else 'import copy\n' + VALID
        args = {'work_item': WORK, 'candidate_source': source}
        return {'usage': {'prompt_tokens': 100, 'completion_tokens': 30},
                'choices': [{'message': {'role': 'assistant', 'content': None,
                    'tool_calls': [{'id': str(len(outbound)), 'type': 'function',
                        'function': {'name': 'submit_candidate', 'arguments': json.dumps(args)}}]}}]}

    with bound_constructor() as constructor:
        result = constructor.run_construction(case, transport=transport,
            limits=ConstructionLimits(wall_seconds=45, max_model_requests=2))
    assert result.status == 'CANDIDATE_READY'
    raw = ROOT / 'analysis/actor-critic-construction-kaggle-20260914T195518Z/runtime-evidence-exact-request/model-runtime/solver-workspaces/raw'
    log = raw / 'logical-requests.jsonl'
    requests = [json.loads(line) for line in log.read_text().splitlines()]
    packet = next(json.loads(m['content'])['visible_case'] for r in requests if r['call'] == 7 for m in r['request']['messages'] if m['role'] == 'user' and 'visible_case' in json.loads(m['content']))
    row = dict(packet['visible_transition_summaries'][0], before=packet['initial_observation'], after=packet['current_observation'])
    historical = []
    inputs = [log]
    for call, error in [(7, "No module named 'copy'"), (8, 'commitments and declared_dependencies have bad types')]:
        path = raw / 'raw-http' / f'{call:02}.body'
        inputs.append(path)
        response = json.loads(path.read_text())
        source = json.loads(response['choices'][0]['message']['tool_calls'][0]['function']['arguments'])['candidate_source']
        evaluation = evaluate_visible_abi(source, ActorPrefixCase((row,), 1))
        assert error in evaluation['visible']['error']
        historical.append({'call': call, 'source_sha256': hashlib.sha256(source.encode()).hexdigest(), 'evaluation': evaluation})
    probe = '''def initialize(observation):
    result={}
    for name in ('json','collections','re','copy','numpy','math','random'):
        try:
            __import__(name); result[name]='available'
        except ImportError:
            result[name]='unavailable'
    return result
def predict(state, action): return []
def assimilate(state, observation, verdict): return state
'''
    return {
        'scope': 'LOCAL SEALED R5 ABI REPLAY; scripted transport, no provider or remote second verdict',
        'outbound_requests': outbound,
        'scripted_status': result.status,
        'synthetic_valid_evaluation': result.evaluation,
        'historical_r4_sources_replayed_locally': historical,
        'sealed_import_observation': SealedBoundary().invoke(probe, 'initialize', [{}]),
        'input_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
    }


if __name__ == '__main__':
    print(json.dumps(collect(), indent=2, sort_keys=True))
