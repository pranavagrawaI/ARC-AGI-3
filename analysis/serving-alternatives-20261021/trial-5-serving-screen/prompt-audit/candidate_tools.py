"""Execute exact candidate tool classes and check the shared rendering proposal.

This checks the one unique tool schema established by audit.py; it does not load
vLLM or claim runtime multimodal processor equivalence.
"""
import ast
import json
import logging
import typing
import pydantic
from tokenizers import Tokenizer
import audit


def load_candidate_model():
    root = audit.HERE / 'source-vllm'
    manifest = json.loads((root / 'manifest.json').read_text())
    for entry in manifest:
        assert audit.digest((root / entry['path']).read_bytes()) == entry['sha256']
    definitions = []
    for name, selected in [
        ('vllm/entrypoints/serve/engine/protocol.py', 'OpenAIBaseModel'),
        ('vllm/entrypoints/generate/base/protocol.py', 'FunctionDefinition'),
        ('vllm/entrypoints/openai/chat_completion/protocol.py', 'ChatCompletionToolsParam'),
    ]:
        parsed = ast.parse((root / name).read_text())
        definitions.extend(n for n in parsed.body if isinstance(n, ast.ClassDef) and n.name == selected)
    assert len(definitions) == 3
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    tree = ast.fix_missing_locations(ast.Module(body=[future] + definitions, type_ignores=[]))
    namespace = dict(BaseModel=pydantic.BaseModel, ConfigDict=pydantic.ConfigDict,
                     Field=pydantic.Field, model_validator=pydantic.model_validator,
                     model_serializer=pydantic.model_serializer, Any=typing.Any,
                     Literal=typing.Literal, ClassVar=typing.ClassVar,
                     logger=logging.getLogger(__name__), _REMOVED_GUIDED_FIELDS=frozenset())
    exec(compile(tree, 'exact-candidate-protocol-classes', 'exec'), namespace)
    model = namespace['ChatCompletionToolsParam']; model.model_rebuild(_types_namespace=namespace)
    return model, manifest


def main():
    model, manifest = load_candidate_model()
    evidence = json.loads((audit.HERE / 'tool-counterfactual.json').read_text())
    original = evidence['original']['tools']; sglang = evidence['normalized']['tools']
    candidate = [model(**tool).model_dump() for tool in original]
    assert audit.corpus.canonical(candidate) == audit.corpus.canonical(original)
    # Key order affects rendered bytes, so equality of canonical JSON alone is insufficient.
    assert json.dumps(candidate) == json.dumps(original)
    catalog = audit.corpus.load(); key, record = next(iter(catalog['records'].items()))
    body = audit.corpus.payload_for(record)
    kwargs = dict(messages=audit.processor_fit.messages_for_template(body['messages']),
                  add_generation_prompt=True, **body['chat_template_kwargs'])
    tpl = audit.processor_fit.template(audit.PRIOR / 'processor-assets/control')
    proposed = tpl.environment.from_string((audit.HERE / 'chat_template.normalized.jinja').read_text())
    expected = tpl.render(tools=sglang, **kwargs)
    assert proposed.render(tools=candidate, **kwargs) == expected
    assert proposed.render(tools=sglang, **kwargs) == expected
    # Sending explicit null/false fields still loses outer null in candidate parsing.
    explicit = [model(**tool).model_dump() for tool in sglang]
    assert tpl.render(tools=explicit, **kwargs) != expected
    tok = Tokenizer.from_file(str(audit.PRIOR / 'processor-assets/candidate/tokenizer.json'))
    count = lambda text: len(tok.encode(text, add_special_tokens=False).ids) + 399 * len(record['images'])
    report = dict(scope='CPU source-class serialization only; no serving runtime', complete=True,
                  source_hashes={r['path']: r['sha256'] for r in manifest},
                  script_sha256=audit.digest(audit.HERE.joinpath('candidate_tools.py').read_bytes()),
                  first_key=key, pydantic=pydantic.__version__,
                  raw_candidate_tools=candidate, explicit_fields_candidate_tools=explicit,
                  original_candidate_prompt=count(tpl.render(tools=candidate, **kwargs)),
                  explicit_fields_candidate_prompt=count(tpl.render(tools=explicit, **kwargs)),
                  normalized_both_arms_prompt=count(expected), historical_prompt=record['prompt'],
                  normalized_render_sha256=audit.digest(expected.encode()),
                  shared_template_matches_both=True, payload_fields_alone_insufficient=True)
    audit_report = json.loads((audit.HERE / 'receipt.json').read_text())
    assert len(audit_report['tool_schema_hashes']) == 1
    assert audit.digest(audit.corpus.canonical(original)) == audit_report['tool_schema_hashes'][0]
    report['all_corpus_same_tool_schema'] = audit_report['requests']
    (audit.HERE / 'candidate-tools-receipt.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('all_corpus_same_tool_schema', 'original_candidate_prompt',
                                          'explicit_fields_candidate_prompt', 'normalized_both_arms_prompt',
                                          'historical_prompt', 'payload_fields_alone_insufficient')}))


if __name__ == '__main__':
    main()
