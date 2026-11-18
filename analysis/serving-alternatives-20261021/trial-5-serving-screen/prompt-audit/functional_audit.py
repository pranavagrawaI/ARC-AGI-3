"""Hash/count the six actual candidate feasibility wire bodies on CPU."""
import json
import signal
import time
from tokenizers import Tokenizer
import audit
from candidate_tools import load_candidate_model


def main():
    signal.alarm(1800); began = time.monotonic()
    root = audit.HERE.parent / 'candidate-feasibility/cases'
    paths = sorted(root.glob('*.json')); assert len(paths) == 6
    template_path = audit.HERE / 'chat_template.normalized.jinja'
    template_sha = audit.digest(template_path.read_bytes())
    sglang = audit.source_models((audit.HERE / 'source/protocol.py').read_text())
    candidate, source_manifest = load_candidate_model()
    env = audit.processor_fit.template(audit.PRIOR / 'processor-assets/control').environment
    template = env.from_string(template_path.read_text())
    toks = {label: Tokenizer.from_file(str(audit.PRIOR / 'processor-assets' / label / 'tokenizer.json'))
            for label in ('control', 'candidate')}
    render_dir = audit.HERE / 'functional-renders'; render_dir.mkdir(exist_ok=True)
    rows = []
    for path in paths:
        raw = path.read_bytes(); body = json.loads(raw)
        assert body['tool_choice'] == {'type': 'function', 'function': {'name': 'python'}}
        assert len(body['tools']) == 1 and body['tools'][0]['function']['name'] == 'python'
        assert body['chat_template_kwargs']['enable_thinking'] is False and body['max_tokens'] == 256
        kwargs = dict(messages=audit.processor_fit.messages_for_template(body['messages']),
                      add_generation_prompt=True, **body['chat_template_kwargs'])
        tools = {'control': audit.normalize_tools(body['tools'], sglang),
                 'candidate': audit.normalize_tools(body['tools'], candidate)}
        rendered = {label: template.render(tools=value, **kwargs) for label, value in tools.items()}
        assert rendered['control'] == rendered['candidate']
        # Literal raw dictionaries must converge too; nested schema order remains
        # exactly the order in the actual serialized feasibility request.
        assert template.render(tools=body['tools'], **kwargs) == rendered['control']
        images = audit.corpus.image_observations(body)
        ids = {label: tok.encode(rendered[label], add_special_tokens=False).ids for label, tok in toks.items()}
        assert ids['control'] == ids['candidate']
        for label in ids: assert ids[label].count(toks[label].token_to_id('<|image_pad|>')) == len(images)
        counts = {label: len(value) + 399 * len(images) for label, value in ids.items()}
        assert counts['control'] + body['max_tokens'] <= 139264
        render_file = render_dir / (path.stem + '.txt'); render_file.write_text(rendered['control'])
        rows.append(dict(case=path.name, request_sha256=audit.digest(raw),
                         image_count=len(images), image_sha256=images,
                         expected_grid_thw=[[1, 40, 40] for _ in images],
                         expected_prompt_tokens=counts, completion_cap=body['max_tokens'],
                         joint_tokens=counts['control'] + body['max_tokens'],
                         expected_render_sha256=audit.digest(render_file.read_bytes()),
                         expected_text_token_ids_sha256={label: audit.digest(audit.corpus.canonical(value)) for label, value in ids.items()},
                         expected_render_file=str(render_file.relative_to(audit.HERE))))
    report = dict(scope='CPU_EXPECTATIONS_ONLY; actual engine pixels/render/counts not yet observed',
                  cases=rows, complete=True, template_sha256=template_sha,
                  audit_source_sha256=audit.digest(audit.HERE.joinpath('functional_audit.py').read_bytes()),
                  candidate_class_audit_source_sha256=audit.digest(audit.HERE.joinpath('candidate_tools.py').read_bytes()),
                  candidate_source_hashes={r['path']: r['sha256'] for r in source_manifest},
                  geometry_source='Pinned model configurations: 640x640, grid1x40x40,400image tokens; runtime pending',
                  elapsed_s=time.monotonic()-began)
    for path,row in zip(paths,rows):assert audit.digest(path.read_bytes()) == row['request_sha256']
    (audit.HERE/'functional-receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps([dict(case=r['case'],prompt=r['expected_prompt_tokens']['candidate'],images=r['image_count'],render_sha=r['expected_render_sha256']) for r in rows]))


if __name__ == '__main__':
    main()
