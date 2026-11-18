"""Score final committed action prefixes without modifying downloaded evidence.

Use the exact candidate chassis that generated the run; this result is post-run
analysis and must never be counted as feedback delivered to the Actor.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile


def replay(session: Path, chassis: Path) -> dict:
    source = chassis/'prediction_feedback.py'
    guard_source = chassis/'solver_agent.py'
    spec = importlib.util.spec_from_file_location('postrun_prediction_feedback', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Load the frozen pure-computation checker without instantiating an Actor or
    # importing a model provider. Keep the checker identical to this candidate.
    tree = ast.parse(guard_source.read_text())
    names = {'_HELPER_IMPORT_ALLOW','_HELPER_BANNED_NAMES','_HELPER_BANNED_DUNDERS'}
    subset = [node for node in tree.body if (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in names for t in node.targets)
    ) or isinstance(node, ast.FunctionDef) and node.name == '_check_helper_ast']
    guard = {}
    exec(compile(ast.Module(body=subset, type_ignores=[]), '<frozen-helper-guard>', 'exec'), guard)
    original = session/'ipc'
    with tempfile.TemporaryDirectory(prefix='q6-final-replay-') as temporary:
        ipc = Path(temporary)
        for name in ('model_predictions.jsonl','transitions.jsonl'):
            if (original/name).exists():
                shutil.copyfile(original/name, ipc/name)
        helpers = ipc/'helpers'
        helpers.mkdir()
        result = module.PredictionFeedback(ipc, helpers, guard['_check_helper_ast']).observe()
    return {
        'schema':'q6-final-feedback-replay-v1', 'session':str(session),
        'generated_post_run':True, 'delivered_to_actor':False,
        'scope':'Last committed action prefix only; consistency under frozen encoder, not world truth.',
        'implementation_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'guard_source_sha256':hashlib.sha256(guard_source.read_bytes()).hexdigest(),
        'raw_inputs_sha256':{name:hashlib.sha256((original/name).read_bytes()).hexdigest()
                            for name in ('model_predictions.jsonl','transitions.jsonl') if (original/name).exists()},
        'feedback':result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    parser.add_argument('--chassis', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Keep derived interpretation outside the complete downloaded run tree.
    session = args.session.resolve()
    raw_root = next((p for p in session.parents if p.name == 'raw'), session)
    if args.output.resolve().is_relative_to(raw_root):
        parser.error('output must be outside raw downloaded evidence')
    result = replay(session, args.chassis.resolve())
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
