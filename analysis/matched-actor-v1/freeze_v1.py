"""Regenerate the disabled manifest after reviewed package changes."""

import json
from pathlib import Path

from tycho_nooa.matched_actor import CODEACT_SYSTEM_PROMPT, NATIVE_SYSTEM_PROMPT
from tycho_nooa.matched_actor_manifest import sha256_file, sha256_text

repo = Path(__file__).resolve().parents[2]
base = Path(__file__).parent
corpus = json.loads((base / 'corpus.json').read_text())
def bind(path):
    target = repo / path
    return {'path': path, 'sha256': sha256_file(target)}
manifest = {
  'schema_version': 1,
  'experiment_id': 'matched-actor-v1',
  'execution_authorized': False,
  'base_commit': '205eb05',
  'corpus': bind('analysis/matched-actor-v1/corpus.json'),
  'corpus_sha256': corpus['corpus_sha256'],
  'controller': bind('solvers/tycho_nooa/src/tycho_nooa/matched_actor.py'),
  'evaluator': bind('solvers/tycho_nooa/src/tycho_nooa/cognitive_parity.py'),
  'runner': bind('solvers/tycho_nooa/src/tycho_nooa/matched_actor_experiment.py'),
  'audit': bind('solvers/tycho_nooa/src/tycho_nooa/matched_actor_audit.py'),
  'audit_driver': bind('analysis/matched-actor-v1/run_scripted_audit.py'),
  'audit_summary': bind('analysis/matched-actor-v1/scripted-audit/summary.json'),
  'shared_computation': bind('analysis/computational-surface-v1/manifest.json'),
  'protocol': bind('analysis/matched-actor-v1/PROTOCOL.md'),
  'experiment_card': bind('analysis/matched-actor-v1/EXPERIMENT-CARD.md'),
  'native_prompt_sha256': sha256_text(NATIVE_SYSTEM_PROMPT),
  'codeact_prompt_sha256': sha256_text(CODEACT_SYSTEM_PROMPT),
  'model_route': {'model': None, 'reported_model': None, 'provider': None, 'reasoning_effort': None, 'temperature': None},
  'caps': {'sessions': 24, 'calls_per_session': 8, 'calls_total': 192, 'output_tokens_per_call': 4096, 'wall_seconds_per_session': 120.0, 'max_cost_usd': 3.0, 'reserve_per_call_usd': 0.02},
  'primary_metrics': ['factored task correctness', 'provider calls', 'input/output tokens', 'dollar cost', 'latency', 'Python cells', 'evidence disclosures', 'invalid and repair interactions', 'milestone call indices'],
  'invalidity_conditions': ['committed input drift', 'unequal evidence, computation, schema, or authority', 'answer-key disclosure', 'missing trace', 'provider route drift', 'canonical or ARC mutation'],
  'stop_conditions': ['24 sessions complete', '192 calls', '$3 observed or reserved', '120 seconds in one session', 'integrity or provider failure'],
  'result_map': ['Native correctness and efficiency -> select Native Actor', 'CodeAct composition gain at tolerable cost -> isolate mechanism', 'equal correctness with higher CodeAct cost -> select Native Actor', 'seam-local failure -> replicate before redesign'],
  'output_root': 'analysis/matched-actor-v1/runs',
}
(base / 'experiment.json').write_text(json.dumps(manifest, indent=2, sort_keys=True) + '\n')
