"""Materialize only small source/metadata/case inputs; no model/runtime execution."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[1]
REPO=HERE.parents[3]


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    trial1=BASE/'trial-1';prior=BASE/'trial-3-matched-preparation';local=BASE/'trial-4-saturated-local'
    source=(trial1/'trial.py').read_bytes();tree=ast.parse(source)
    names=('prepend_path_entries','read_runtime_manifest','build_runtime_python_paths','resolve_runtime_cuda_home',
           'build_runtime_library_paths','build_image_runtime_env','verify_layer_blob','clear_opaque_whiteout_targets',
           'remove_whiteout_markers_and_targets','extract_layer','apply_runtime_patch','install_image_runtime',
           'validate_prefix_cache_runtime','build_vllm_server_command')
    constants={}
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
            name=node.targets[0].id
            if name.startswith(('VLLM_','FINE_PREFIX_CACHE_')) or name in ('SERVED_MODEL_NAME','RUNTIME_MANIFEST_FILE_NAME'):
                try:constants[name]=ast.literal_eval(node.value)
                except (ValueError,TypeError):pass
    constants.update(VLLM_MAX_NUM_SEQS=10,VLLM_MAX_MODEL_LEN=139264,VLLM_GPU_MEMORY_UTILIZATION='0.96',
                     VLLM_IMAGE_MAX_PIXELS=409600,VLLM_PORT=8001,VLLM_HOST='127.0.0.1',SERVED_MODEL_NAME='flashnext')
    selected=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    if {n.name for n in selected}!=set(names):raise ValueError('missing source function')
    library='import os,sys,json,math,shutil,subprocess,time\nfrom pathlib import Path\n'
    library+='\n'.join(name+' = '+repr(value) for name,value in sorted(constants.items()))+'\n'
    library+='\n\n'.join(ast.get_source_segment(source.decode(),node) for node in selected)+'\n'
    compile(library,'candidate_library.py','exec');(HERE/'candidate_library.py').write_text(library)
    vocab=(REPO/'analysis/m2-winners-20261001/audit/lord-bundle/src/ARC3-Inference/configs/draft_vocab_32k.json').read_bytes()
    if digest(vocab)!='55176ffc7ae9545d174118477354fe660adfeed9530430a639eda1633f18f4a3':raise ValueError('draft vocab changed')
    (HERE/'draft_vocab_32k.json').write_bytes(vocab)
    receipt=json.loads((prior/'processor-assets/receipt.json').read_text())['models']['candidate']
    shutil.copyfile(trial1/'input-meta/runtime/runtime-manifest.json',HERE/'runtime-manifest.reference.json')
    spec=importlib.util.spec_from_file_location('corpus',local/'corpus.py');corpus=importlib.util.module_from_spec(spec);spec.loader.exec_module(corpus)
    corpus.verify_corpus();catalog=corpus.load()
    proof=json.loads((prior/'functional-cases.json').read_text());cases=[]
    directory=HERE/'cases';directory.mkdir(exist_ok=True)
    for case in proof['cases']:
        key=case['game']+':'+str(case['index']);record=catalog['records'][key]
        body=corpus.payload_for(record)
        transform=proof['transformation']
        body['messages'].append(dict(role='user',content=transform['append_user_text']))
        for field in ('max_tokens','temperature','top_p','tool_choice','chat_template_kwargs'):body[field]=transform[field]
        raw=corpus.canonical(body)
        # Confirm same actual functional source bytes already audited; no image reduction.
        if corpus.image_observations(body)!=record['images']:raise ValueError('functional image loss')
        path=directory/(key.replace(':','-')+'.json');path.write_bytes(raw)
        cases.append(dict(key=key,images=len(record['images']),request_path='cases/'+path.name,
                          request_sha256=digest(raw),source_wire_sha=record['wire_sha'],
                          local_prompt_tokens=case['prompt_tokens']['candidate'],processor_runtime_qualified=False))
    (HERE/'cases.json').write_text(json.dumps(dict(cases=cases,concurrency=2,per_request_seconds=300),indent=2)+'\n')
    audit=json.loads((HERE.parent/'prompt-audit/functional-receipt.json').read_text())
    for case in cases:
        name=case['key'].replace(':','-')+'.json'
        row=next(r for r in audit['cases'] if r['case']==name)
        if row['request_sha256']!=case['request_sha256']:raise ValueError('normalized expectation is for another actual request')
        case['local_prompt_tokens']=row['expected_prompt_tokens']['candidate']
        case['expected_render_sha256']=row['expected_render_sha256']
    (HERE/'cases.json').write_text(json.dumps(dict(cases=cases,concurrency=2,per_request_seconds=280),indent=2)+'\n')
    applier=HERE/'input-reference/apply_vllm_main_e975732_arc3.py'
    model=dict(model_version=receipt['model_version'],files=receipt['files'],assets=receipt['assets'])
    template=HERE.parent/'prompt-audit/chat_template.normalized.jinja'
    shutil.copyfile(template,HERE/'chat_template.normalized.jinja')
    identity=HERE/'input-reference/PATCH_IDENTITY.json'
    plan=dict(run_id='serving-candidate-feas5-20261023',runtime_dataset='lordhansolo/vllm-main-e975732-arc3',
              runtime_manifest_sha=digest((HERE/'runtime-manifest.reference.json').read_bytes()),
              applier_sha=digest(applier.read_bytes()),patch_identity_sha=digest(identity.read_bytes()),model=model,
              normalized_template_sha=digest(template.read_bytes()),
              candidate_frontend_source_sha=audit['candidate_source_hashes'],
              original_trial1_source_sha=digest(source),library_sha=digest(library.encode()),
              original_functions={node.name:digest(ast.get_source_segment(source.decode(),node).encode()) for node in selected},
              draft_vocab_sha=digest(vocab),corpus_manifest_sha=corpus.MANIFEST_SHA,catalog_sha=corpus.CATALOG_SHA,
              native_overlay_enabled=False,rate_comparison=False)
    (HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(dict(cases=len(cases),max_images=max(c['images'] for c in cases),library_bytes=len(library),weights_downloaded=False)))


if __name__=='__main__':main()
