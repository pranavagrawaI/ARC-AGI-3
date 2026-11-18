"""Reconcile pinned SGLang tool serialization on CPU; never import an engine.

Run from any cwd with cached tokenizers/Jinja2/Pydantic and a 1800s cap.
Only this directory receives outputs. Trial 1--4 inputs remain read-only.
"""
import ast
import collections
import hashlib
import json
from pathlib import Path
import signal
import sys
import time
import typing
import zipfile

import jinja2
import pydantic
import tokenizers
from pydantic import BaseModel, Field, model_serializer, model_validator
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
SURVEY = HERE.parents[1]
ROOT = SURVEY.parents[1]
PRIOR = SURVEY / 'trial-3-matched-preparation'
sys.path.insert(0, str(SURVEY / 'trial-4-saturated-local'))
sys.path.insert(0, str(PRIOR))
import corpus
import processor_fit


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def source_models(protocol):
    """Execute the two actual wheel classes, not a hand-written approximation."""
    classes = [n for n in ast.parse(protocol).body
               if isinstance(n, ast.ClassDef) and n.name in ('Function', 'Tool')]
    assert [n.name for n in classes] == ['Function', 'Tool']
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future] + classes, type_ignores=[]))
    namespace = dict(BaseModel=BaseModel, Field=Field, model_serializer=model_serializer,
                     model_validator=model_validator, Optional=typing.Optional)
    exec(compile(tree, 'pinned-wheel-protocol-classes', 'exec'), namespace)
    namespace['Tool'].model_rebuild(_types_namespace=namespace)
    return namespace['Tool']


def normalize_tools(tools, model):
    return [model(**tool).model_dump() for tool in tools]


CANONICAL_TOOL = """{'type': tool.type, 'function': {'description': tool.function.description,
        'name': tool.function.name, 'parameters': tool.function.parameters,
        'strict': tool.function.strict | default(false, true)},
        'defer_loading': tool.defer_loading | default(none)} | tojson"""


def main():
    signal.alarm(1800)
    began = time.monotonic()
    contract = json.loads((PRIOR / 'proposed-config.json').read_text())['control']['native_contract']
    wheel = ROOT / 'analysis/serving-headroom-v3-20261008/evidence/packages' / contract['wheel']['filename']
    assert digest(wheel.read_bytes()) == contract['wheel']['sha256']
    evidence = HERE / 'source'; evidence.mkdir(exist_ok=True)
    sources = {}
    with zipfile.ZipFile(wheel) as archive:
        for short in ('protocol.py', 'serving_chat.py'):
            name = 'sglang/srt/entrypoints/openai/' + short
            raw = archive.read(name); (evidence / short).write_bytes(raw); sources[name] = digest(raw)
        name = 'sglang/srt/parser/template_manager.py'
        raw = archive.read(name); (evidence / 'template_manager.py').write_bytes(raw); sources[name] = digest(raw)
    tool_model = source_models((evidence / 'protocol.py').read_text())
    corpus.verify_corpus(); catalog = corpus.load()
    old_raw = (PRIOR / 'processor-fit-records.jsonl').read_bytes()
    assert digest(old_raw) == corpus.RENDER_AUDIT_SHA
    old = {r['game'] + ':' + str(r['index']): r for r in map(json.loads, old_raw.splitlines())}
    templates = {}; toks = {}; asset_hashes = {}
    for label in ('control', 'candidate'):
        directory = PRIOR / 'processor-assets' / label
        templates[label] = processor_fit.template(directory)
        toks[label] = Tokenizer.from_file(str(directory / 'tokenizer.json'))
        for name in ('chat_template.jinja', 'tokenizer.json', 'tokenizer_config.json', 'preprocessor_config.json'):
            asset_hashes[label + '/' + name] = digest((directory / name).read_bytes())
    original_template = (PRIOR / 'processor-assets/control/chat_template.jinja').read_text()
    assert original_template.count('tool | tojson') == 1
    proposed = original_template.replace('tool | tojson', CANONICAL_TOOL)
    (HERE / 'chat_template.normalized.jinja').write_text(proposed)
    normalized_template = templates['control'].environment.from_string(proposed)
    report = dict(scope='CPU_ONLY; engine paths traced, exact Tool/Function classes executed, no model/runtime/GPU',
                  wheel_sha256=contract['wheel']['sha256'], source_hashes=sources, asset_hashes=asset_hashes,
                  corpus_manifest_sha=corpus.MANIFEST_SHA, catalog_sha=corpus.CATALOG_SHA,
                  old_render_audit_sha=corpus.RENDER_AUDIT_SHA,
                  versions=dict(tokenizers=tokenizers.__version__, jinja2=jinja2.__version__, pydantic=pydantic.__version__),
                  requests=0, missing_initial_system=0, historical_minus_old=collections.Counter(),
                  historical_minus_normalized=collections.Counter(), arm_render_mismatches=0,
                  arm_token_id_mismatches=0, shared_template_mismatches=0,
                  max_joint_tokens=0, tool_schema_hashes=set())
    with (HERE / 'records.jsonl').open('w') as out:
        for index, (key, record) in enumerate(catalog['records'].items()):
            body = corpus.payload_for(record)
            assert corpus.wire_hash(body) == record['wire_sha']
            # Normalization proposal is scoped to this existing function-tool schema.
            assert all(set(t) == {'type', 'function'} and set(t['function']) == {'name', 'description', 'parameters'}
                       for t in body['tools'])
            report['tool_schema_hashes'].add(digest(corpus.canonical(body['tools'])))
            messages = processor_fit.messages_for_template(body['messages'])
            report['missing_initial_system'] += messages[0]['role'] != 'system'
            kwargs = dict(messages=messages, add_generation_prompt=True, **body.get('chat_template_kwargs', {}))
            normalized = normalize_tools(body['tools'], tool_model)
            original = templates['control'].render(tools=body['tools'], **kwargs)
            assert digest(original.encode()) == old[key]['render_sha256']['control']
            rendered = {label: template.render(tools=normalized, **kwargs) for label, template in templates.items()}
            ids = {label: toks[label].encode(text, add_special_tokens=False).ids for label, text in rendered.items()}
            counts = {label: len(value) + len(record['images']) * 399 for label, value in ids.items()}
            report['arm_render_mismatches'] += rendered['control'] != rendered['candidate']
            report['arm_token_id_mismatches'] += ids['control'] != ids['candidate']
            for supplied_tools in (body['tools'], normalized):
                report['shared_template_mismatches'] += normalized_template.render(tools=supplied_tools, **kwargs) != rendered['control']
            report['historical_minus_old'][record['prompt'] - old[key]['prompt_tokens']['control']] += 1
            report['historical_minus_normalized'][record['prompt'] - counts['control']] += 1
            report['max_joint_tokens'] = max(report['max_joint_tokens'], counts['control'] + record['cap'])
            report['requests'] += 1
            row = dict(key=key, images=len(record['images']), cap=record['cap'], historical_prompt=record['prompt'],
                       old_prompt=old[key]['prompt_tokens']['control'], normalized_prompt=counts,
                       old_render_sha=old[key]['render_sha256']['control'],
                       normalized_render_sha={label: digest(text.encode()) for label, text in rendered.items()},
                       normalized_token_ids_sha={label: digest(corpus.canonical(value)) for label, value in ids.items()})
            out.write(json.dumps(row, sort_keys=True) + '\n')
            if index == 0:
                ablations = {}
                for name, tools in [('original', body['tools']), ('normalized', normalized)]:
                    text = templates['control'].render(tools=tools, **kwargs)
                    ablations[name] = dict(prompt_tokens=len(toks['control'].encode(text, add_special_tokens=False).ids) + len(record['images']) * 399,
                                           tools=tools)
                (HERE / 'tool-counterfactual.json').write_text(json.dumps(ablations, indent=2) + '\n')
            if (index + 1) % 250 == 0: print(index + 1, 'requests', round(time.monotonic() - began, 1), 'seconds', flush=True)
    report['tool_schema_hashes'] = sorted(report['tool_schema_hashes'])
    report['records_sha256'] = digest((HERE / 'records.jsonl').read_bytes())
    report['normalized_template_sha256'] = digest((HERE / 'chat_template.normalized.jinja').read_bytes())
    report['audit_source_sha256'] = digest(Path(__file__).read_bytes())
    report['elapsed_s'] = time.monotonic() - began
    report['complete'] = True
    (HERE / 'receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    assert report['requests'] == 1369 and not report['missing_initial_system']
    assert report['historical_minus_old'] == {11: 1369} and report['historical_minus_normalized'] == {0: 1369}
    assert not any(report[k] for k in ('arm_render_mismatches', 'arm_token_id_mismatches', 'shared_template_mismatches'))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
