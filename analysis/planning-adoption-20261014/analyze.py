"""Independent post-collection receipt/native audit and conservative adoption tallies."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def core():
    os.environ['PLANNING_NATIVE_AUDIT_ROOT']=str(ROOT)
    # Match the frozen notebook and TAAF session's level-reset policy.
    os.environ['ONLY_RESET_LEVELS']='true'
    spec=importlib.util.spec_from_file_location('adoption_audit',HERE/'runtime/audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def mechanisms(directory):
    result=json.loads((directory/'result.json').read_text())
    responses=[]
    for line in (directory/'wire.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row.get('event')=='response':responses.append(row)
    attempted={'bounded_search':0,'probe_transition':0}
    no_tool_tokens=0;cutoff_no_tool_tokens=0;parse_failures=0;tool_calls=0
    for row in responses:
        body=row.get('body',{});usage=body.get('usage',{}).get('completion_tokens')
        choices=body.get('choices') or []
        if not choices:continue
        choice=choices[0];calls=choice.get('message',{}).get('tool_calls') or []
        if not calls and type(usage) is int:
            no_tool_tokens+=usage
            if choice.get('finish_reason')=='length':cutoff_no_tool_tokens+=usage
        for call in calls:
            if call.get('function',{}).get('name')!='python':continue
            tool_calls+=1
            try:
                args=json.loads(call['function']['arguments']);tree=ast.parse(args['code'])
            except (ValueError,KeyError,TypeError,SyntaxError):parse_failures+=1;continue
            definitions={node.name for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)}
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in attempted and node.func.id not in definitions:
                    attempted[node.func.id]+=1
    saved=result.get('saved_data',{})
    probe_rows=[];search_rows=[]
    for name,record in saved.items():
        value=record.get('value') if isinstance(record,dict) else None
        if not isinstance(value,dict):continue
        if {'predicted','before','receipt','matched','status'}<=set(value):
            probe_rows.append({'name':name,'status':value['status'],'matched':value['matched']})
        if {'stored_nodes','depth_frontier_states','status','expanded'}<=set(value):
            search_rows.append({'name':name,'status':value['status'],'expanded':value['expanded'],'depth':value.get('depth')})
    return {'attempted_name_calls':attempted,'persisted_probe_records':probe_rows,'persisted_search_records':search_rows,
            'responses':len(responses),'python_tool_calls':tool_calls,'python_parse_failures':parse_failures,
            'no_tool_tokens':no_tool_tokens,'length_cutoff_no_tool_tokens':cutoff_no_tool_tokens,
            'limit':'Name calls are attempts, not proof of builtin execution. Saved result schemas support observed execution; missing saved search results leave uptake unconfirmed. No-tool tokens are not causal waste.'}


def receipt_integrity(directory,manifest,engine_hash):
    """Check frozen identities and the action budget independently of worker flags."""
    required=['identity.json','result.json','checkpoint.json','actions.json']
    if any(not (directory/name).exists() for name in required):
        return ['missing identity/action receipts']
    identity=json.loads((directory/'identity.json').read_text())
    result=json.loads((directory/'result.json').read_text())
    checkpoint=json.loads((directory/'checkpoint.json').read_text())
    actions=json.loads((directory/'actions.json').read_text())
    issues=[]
    limits={'tokens':32768,'model_calls':32,'native_actions':150,'episode_seconds':900,'request_seconds':120}
    if identity.get('limits')!=limits or identity.get('variant')!=result.get('variant'):
        issues.append('frozen episode limits/variant mismatch')
    protected=identity.get('protected_files',{})
    if not protected:issues.append('empty protected-file coverage')
    matched=set();engine_seen=False
    for path,sha in protected.items():
        names=[name for name in manifest if path.endswith('/'+name)]
        if len(names)==1:
            expected=manifest[names[0]]
            matched.add(names[0])
        elif path.endswith('/s5i5/18d95033/s5i5.py'):
            expected=engine_hash
            engine_seen=True
        else:
            issues.append('unknown protected source: '+path);continue
        if sha!=expected:issues.append('protected source differs from frozen package: '+path)
    checkpoint_name=result.get('checkpoint')
    fixed={'analysis/planning-transfer-20261014/continuation.py','solvers/m2_planning_tools/variants.py',
           'analysis/counterexample-continuations-v3-20261013/restore.py',
           'analysis/counterexample-continuations-v3-20261013/worker.py',
           'analysis/s5i5-inspection-20261012/engine-replay.json',
           f'analysis/counterexample-continuations-v3-20261013/inputs/{checkpoint_name}.json'}
    required={name for name in manifest if name in fixed or
              name.endswith('.py') and name.startswith((f'sources/{checkpoint_name}/','sources/taaf/','solvers/m2_planning_tools/overlay/'))}
    if not required<=matched or not engine_seen:
        issues.append('incomplete protected source coverage')
    count=result.get('new_native_actions')
    observed=len(actions)-len(checkpoint.get('prefix_actions',[]))
    if type(count) is not int or not 0<=count<=150 or observed!=count:
        issues.append('native action budget/accounting mismatch')
    return issues


def main():
    receipt=json.loads((HERE/'collection.json').read_text())
    for row in receipt['downloaded']:
        path=HERE/'output'/row['file']
        if path.stat().st_size!=row['bytes'] or hashlib.file_digest(path.open('rb'),'sha256').hexdigest()!=row['sha256']:
            raise RuntimeError('Collected raw hash mismatch: '+row['file'])
    identity=json.loads((HERE/'identity.json').read_text())
    matrix_bytes=(HERE/'matrix.json').read_bytes()
    if hashlib.sha256(matrix_bytes).hexdigest()!=identity['manifest']['analysis/planning-transfer-20261014/matrix.json']:
        raise RuntimeError('Local matrix differs from frozen package')
    expected=json.loads(matrix_bytes)
    output=HERE/'output/planning-adoption/episodes'
    if not (output/'matrix.json').exists():
        report={'verdict':'INVALID_SETUP_OR_MISSING_MATRIX','selected':16,'executed':0,
                'terminal_status':receipt['status'],'scope':'No efficacy evidence; inspect terminal log'}
        (HERE/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));return
    if json.loads((output/'matrix.json').read_text())!=expected:raise RuntimeError('Remote matrix changed')
    remote_manifest=json.loads((HERE/'output/planning-adoption/package-manifest.json').read_text())
    package_match=remote_manifest==identity['manifest']
    local_engine=json.loads((ROOT/'analysis/s5i5-inspection-20261012/engine-replay.json').read_text())
    audit=core();rows=[audit.audit_episode(output/row['id'],row) for row in expected]
    for row in rows:
        row['issues']+=receipt_integrity(output/row['id'],identity['manifest'],local_engine['engine_sha256'])
        row['valid']=not row['issues']
    summary=audit.summarize(rows,expected)
    adoption={row['id']:mechanisms(output/row['id']) for row in expected if (output/row['id']/'wire.jsonl').exists() and (output/row['id']/'result.json').exists()}
    server_info=json.loads((HERE/'output/planning-adoption/server-info.json').read_text())
    source=json.loads((HERE/'output/planning-adoption/source-receipts.json').read_text())
    engine_match=source['engine_sha256']==local_engine['engine_sha256']
    runtime_keys=['max_running_requests','context_length','max_mamba_cache_size','quantization','speculative_num_steps','speculative_num_draft_tokens']
    runtime={key:server_info.get(key) for key in runtime_keys}
    runtime_match=all(runtime.get(key)==value for key,value in {'max_running_requests':10,'context_length':139264,
                      'max_mamba_cache_size':60,'quantization':'auto-round','speculative_num_steps':3,'speculative_num_draft_tokens':4}.items())
    valid=summary['valid']==16 and engine_match and runtime_match and package_match
    report={**summary,'verdict':'VALID_SCREEN' if valid else 'INCOMPLETE_OR_INVALID','episodes':rows,'mechanisms':adoption,
            'runtime':runtime,'runtime_matches':runtime_match,'engine_matches':engine_match,'package_matches':package_match,
            'raw_hashes_verified':len(receipt['downloaded']),'kernel_id':receipt['kernel_id'],'version':receipt['version']}
    (HERE/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['episodes','mechanisms']},indent=2))


if __name__=='__main__':main()
