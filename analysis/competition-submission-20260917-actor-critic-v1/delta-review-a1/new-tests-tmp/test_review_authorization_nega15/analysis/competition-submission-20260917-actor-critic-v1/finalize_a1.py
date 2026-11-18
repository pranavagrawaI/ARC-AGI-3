"""Reconcile the A1-repaired bytes against fresh cp312 evidence under checks/repair-a1/; write handoff.
Never rewrites RESULT.md, finalize.py output or independent-review/ (pre-repair history stays as authored).
"""
from pathlib import Path
import hashlib,json,os,re,subprocess
RUN=Path(__file__).resolve().parent;ROOT=RUN.parents[1];PACKAGE=ROOT/'deployment/kaggle/submission-20260917-actor-critic-competition-v1'
CHECKS=RUN/'checks/repair-a1'
REFUSAL='deep investigation is disabled under competition request accounting'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def notebook_code(path):
    n=json.loads(Path(path).read_text());return '\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
def main():
    import launch_control as gate
    manifest=gate.reconcile(str(RUN))
    for name in ('authorization.json','final-review.json','CONSUMED.json'):assert not (RUN/name).exists(),name
    hold=gate.main(['check','--run-dir',str(RUN)]);assert hold==2
    # Repair surface: guard present, advertisement gone, chassis contract rehashed.
    solver=(PACKAGE/'runtime/chassis/solver_agent.py').read_text();headless=(PACKAGE/'runtime/headless.py').read_text()
    assert solver.count(REFUSAL)==1 and solver.index('if os.environ.get("ARC_GAME_REQUEST_LEDGER"):\n            raise RuntimeError("'+REFUSAL)<solver.index('maximum = int(os.environ.get("ARC_DEEP_MAX_CALLS", "1"))')
    assert 'Use await self.deep_investigate' not in headless and 'deep_investigate' in solver
    contract=json.loads((PACKAGE/'runtime/CHASSIS.json').read_text())
    for name,digest in contract['files'].items():assert sha(PACKAGE/'runtime'/name)==digest,name
    subprocess.run([str(ROOT/'deployment/kaggle/.venv/bin/python'),'-B',str(PACKAGE/'runtime/freeze_candidate.py'),'--check'],check=True,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',ARC_DISABLE_DOTENV='1'))
    # Pre-repair identities are preserved and bound to the retained pre-repair copies.
    pre=json.loads((RUN/'PRE-REPAIR-HASHES.json').read_text());before=CHECKS/'before-package'
    assert sha(CHECKS/'before-current-artifacts/manifest.json')==pre['manifest']
    assert sha(CHECKS/'before-current-artifacts/EXPERIMENT-CARD.md')==pre['card']==sha(RUN/'EXPERIMENT-CARD.md')==manifest['card_sha256']
    assert sha(before/'submission.ipynb')==pre['notebook']!=manifest['notebook_sha256']
    for name,digest in pre['package'].items():assert sha(before/name)==digest,name
    assert {str(p.relative_to(before)) for p in before.rglob('*') if p.is_file()}==set(pre['package'])
    for name,digest in pre['independent_review'].items():assert sha(ROOT/name)==digest,name
    changed=sorted(n for n,d in pre['package'].items() if sha(PACKAGE/n)!=d)
    assert changed==['ASSETS.json','runtime/CHASSIS.json','runtime/chassis/solver_agent.py','runtime/headless.py','submission.ipynb'],changed
    # Reviewer's own fixture: pre-repair bytes reproduce the defect; repaired bytes close the contract.
    def deep_summary(base):
        s=json.loads((base/'DEEP-SUMMARY.json').read_text());r=json.loads((base/'RESULT.json').read_text())
        reqs=[json.loads(l) for l in (base/'model-requests.jsonl').read_text().splitlines()]
        return s,r,reqs,len(list(base.rglob('deep_investigations.jsonl')))
    s,r,reqs,files=deep_summary(CHECKS/'deep-before')
    assert (CHECKS/'deep-before/execute-notebook.py').read_text()==notebook_code(before/'submission.ipynb')
    assert r['status']=='reconciliation_failed' and s['server_requests_total']==14 and s['shared_ledger_rows']==13==s['expected_launcher_solver_requests']
    assert s['deep_investigation_records']==1==files and s['deep_max_tokens_requested']==[32768] and (CHECKS/'deep-before/RECONCILE-ERROR.txt').exists()
    assert not any(REFUSAL in json.dumps(x) for x in reqs)
    s,r,reqs,files=deep_summary(CHECKS/'deep-after')
    assert (CHECKS/'deep-after/execute-notebook.py').read_text()==notebook_code(PACKAGE/'submission.ipynb')
    assert r['status']=='passed' and r['games']==2 and s['server_requests_total']==s['shared_ledger_rows']==s['expected_launcher_solver_requests']==13
    assert s['deep_investigation_records']==0==files and s['deep_max_tokens_requested']==[] and r['supervisor']['whole_model_request_count_verified'] is True
    assert sum('deep_investigate(' in json.dumps(x) for x in reqs)>0 and any(REFUSAL in json.dumps(x) for x in reqs)
    assert all(x['request'].get('max_tokens')==192 for x in reqs if x['purpose']=='critic')
    deep_evidence=dict(before=json.loads((CHECKS/'deep-before/DEEP-SUMMARY.json').read_text()),after=json.loads((CHECKS/'deep-after/DEEP-SUMMARY.json').read_text()))
    tests={}
    for name,count in [('inherited-cp312',145),('entrypoints-cp312',37),('new-tests-cp312',49)]:
        log=CHECKS/f'{name}.log';text=log.read_text();assert re.search(rf'\b{count} passed\b',text),text[-2000:]
        assert not re.search(r'\b[1-9]\d* failed\b',text)
        tests[name]=dict(passed=count,failed=0,log=str(log.relative_to(ROOT)),sha256=sha(log))
    assets=json.loads((PACKAGE/'ASSETS.json').read_text());fixtures={};raw_hashes={}
    for mode,expected in [('110',(110,550,110)),('reserve',(2,2,2)),('critic-cap',(2,24,8)),('deep',(2,11,2))]:
        base=CHECKS/f'gateway-{mode}-cp312';r=json.loads((base/'RESULT.json').read_text());assert r['status']=='passed'
        assert (r['games'],r['actor'],r['critic'])==expected,(mode,r)
        assert r['supervisor']['whole_model_request_count_verified'] is True
        assert (base/'execute-notebook.py').read_text()==notebook_code(PACKAGE/'submission.ipynb')
        origins=json.loads((base/'dependency-origins.json').read_text());assert origins['verified_solver_sources']==32
        assert all('site-packages' in Path(p).parts for p in origins['origins'].values())
        for name,digest in assets.items():assert sha(base/'notebook'/name)==digest,name
        materialized=base/'notebook/working/nooa-join'
        assert json.loads((materialized/'CHASSIS.json').read_text())==contract
        for name,digest in contract['files'].items():assert sha(materialized/name)==digest,name
        q=base/'notebook/working/qualification';io=json.loads((q/'import-origins.json').read_text())
        assert all(Path(p).is_relative_to(base/'notebook/working/ARC-AGI-3-Agents') for p in io.values())
        progress=[json.loads(line) for line in (q/'game-progress.jsonl').read_text().splitlines()]
        assert len(progress)==expected[0] and len({g['game_id'] for g in progress})==expected[0]
        for i,g in enumerate(progress):
            ledger=[json.loads(line) for line in (q/f'request-budget-{i}.jsonl').read_text().splitlines()]
            actors=sum(x['request_purpose']=='actor' for x in ledger);critics=sum(x['request_purpose']=='critic' for x in ledger)
            assert actors<=12 and critics<=4 and len(ledger)==actors+critics
        raw_counts={'actor':0,'critic':0}
        for log in q.rglob('model_calls.jsonl'):
            for line in log.read_text().splitlines():
                call=json.loads(line);raw_counts[call.get('request_purpose') or 'actor']+=1
        assert raw_counts==dict(actor=expected[1],critic=expected[2]),raw_counts
        assert not list(q.rglob('deep_investigations.jsonl')),mode
        reqs=[json.loads(l) for l in (base/'model-requests.jsonl').read_text().splitlines()]
        assert len(reqs)==expected[1]+expected[2] and all(x['request'].get('max_tokens')==192 for x in reqs if x['purpose']=='critic')
        if mode=='deep':
            s=json.loads((base/'DEEP-SUMMARY.json').read_text())
            assert s['refusal_observed'] and s['deep_model_requests']==0==s['deep_investigation_files']
            assert s['server_requests_total']==s['shared_ledger_rows']==s['expected_launcher_solver_requests']==13
            fixtures[mode]=dict(r,deep_summary=s)
        else:fixtures[mode]=r
        for p in [base/'RESULT.json',base/'dependency-origins.json',base/'model-requests.jsonl',base/'gateway-scorecard.json',base/'notebook/working/submission.parquet',q/'launcher-result.json',q/'game-progress.jsonl',q/'budget-allocations.jsonl',q/'import-origins.json']:
            raw_hashes[str(p.relative_to(ROOT))]=sha(p)
    for base in (CHECKS/'deep-before',CHECKS/'deep-after'):
        for p in [base/'RESULT.json',base/'DEEP-SUMMARY.json',base/'model-requests.jsonl',base/'execute-notebook.py']:raw_hashes[str(p.relative_to(ROOT))]=sha(p)
    # Prior 110-game evidence binds to the pre-repair assets only; it is supplemental, not qualification of these bytes.
    prior=json.loads((CHECKS/'prior-110-audit.json').read_text())
    assert prior['audit_against_current_assets']['110']['materialized_assets_match'] is False
    assert prior['materialized_assets_match_pre_repair']=={'110':True,'reserve':True,'critic-cap':True}
    assert prior['audit_against_current_assets']['110']['RESULT']==dict(status='passed',games=110,actor=550,critic=110)
    assert not list(PACKAGE.rglob('*.pyc')) and not list(PACKAGE.rglob('__pycache__'))
    gate.reconcile(str(RUN))
    baseline=(RUN/'baseline-status.txt').read_text();current=subprocess.check_output(['git','status','--porcelain=v1','-uno'],cwd=ROOT,text=True)
    assert baseline==current,'Tracked status changed; inspect concurrent work before claiming preservation'
    hashes=dict(manifest=sha(RUN/'manifest.json'),card=sha(RUN/'EXPERIMENT-CARD.md'),notebook=sha(PACKAGE/'submission.ipynb'))
    assert hashes['manifest']!=pre['manifest'] and hashes['card']==pre['card'] and hashes['notebook']!=pre['notebook']
    changed_sources={n:dict(before=pre['package'][n],after=sha(PACKAGE/n)) for n in changed}
    for n in ('qualify_gateway.py','test_local.py','freeze.py','finalize_a1.py'):
        key=str((RUN/n).relative_to(ROOT));changed_sources[key]=dict(before=json.loads((CHECKS/'before-current-artifacts/manifest.json').read_text())['sources'].get(key),after=manifest['sources'][key])
    unexercised=['GPU/container offline bootstrap and model warmup on RTX PRO 6000','real model latency and gameplay capability','private Kaggle gateway parquet writer','actual private 110-game inventory','provider competition timeout/queue-overhead semantics','live one-push and competition submission','fresh quota and remote exact-version retrieval']
    result=dict(task='T3R',repair='A1',status='locally_requalified_not_authorized',hashes=hashes,pre_repair_hashes={k:pre[k] for k in ('manifest','card','notebook')},changed_sources=changed_sources,chassis_contract=dict(before=json.loads((before/'runtime/CHASSIS.json').read_text())['contract_sha256'],after=contract['contract_sha256']),pytest=dict(passed=231,failed=0,suites=tests),gateway_fixtures=fixtures,deep_reviewer_fixture=deep_evidence,prior_110_evidence='supplemental only: materialized assets bind to pre-repair ASSETS.json, not to the repaired bytes',gate='HOLD exit 2; review and authorization absent',source_bindings_recomputed=len(manifest['sources']),donor_bindings_recomputed=len(json.loads((PACKAGE/'DONORS.json').read_text())),candidate_bytes_unchanged_by_final_checks=True,tracked_status_unchanged=True,raw_artifact_hashes=raw_hashes,unexercised=dict(count=len(unexercised),items=unexercised),gpu_calls=0,paid_model_calls=0,kernel_pushes=0,competition_submissions=0)
    (CHECKS/'final-checks.json').write_text(json.dumps(result,indent=2)+'\n')
    paths=sorted(str(p.relative_to(ROOT)) for base in (RUN,PACKAGE) for p in base.rglob('*'))
    paths.append(str((RUN/'CREATED-FILES.json').relative_to(ROOT)))
    (RUN/'CREATED-FILES.json').write_text(json.dumps(sorted(set(paths)),indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],hashes=hashes,pre_repair_hashes=result['pre_repair_hashes'],pytest_passed=231,fixtures_passed=4,unexercised_remote_categories=7),indent=2))
if __name__=='__main__':main()
