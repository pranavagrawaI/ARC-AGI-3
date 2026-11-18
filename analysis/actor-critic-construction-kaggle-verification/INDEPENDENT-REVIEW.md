# Independent Kaggle package review — HOLD

## Decision and scope

**HOLD. Do not launch these bytes.** Read-only review of the actual prepared package, not an assumed rebuilt candidate. Source review covered `build.py`, generated notebook and nested bootstrap, `kaggle_probe.py`, `local_wire.py`, `audit.py`, `headless.py`, and relevant worker, constructor, containment and test-driver dependencies. No candidate edits, launch, model request, provider call, network access, or edits to old OpenRouter artifacts occurred. Reproductions use Python 3.12 from `deployment/kaggle/.venv/bin/python`, disposable `/tmp` fixtures, bytecode disabled, and a socket audit hook denying connect/bind/name resolution.

All source hashes recorded in the coordinator's verification RESULTS.json still matched at this review's final reconciliation. Package manifest assets/files reconcile, but `complete` is false. Hash consistency is not approval.

## Blocking findings

### P1 — Notebook cannot bootstrap its own required files

`build.py:90–127`; actual notebook code cells 1–2. The notebook contains no payload for the allowlisted local files or `donor_bootstrap.py`. Cell 1 only copies files if they already exist relative to the working directory, silently skips absent files, then imports `accelerator`. Executing the actual cell with only `/kaggle/working` redirected into a disposable empty working directory raises `ModuleNotFoundError: No module named 'accelerator'`, before GPU discovery or any model activity. `donor_bootstrap.py` is not in even that copy list; cell 2 unconditionally executes its absolute working-directory path. If files already reside in the working directory, the copy branch instead raises `SameFileError` on its first file. Thus neither absent-assets nor already-in-working placement is qualified.

Repair needed: an explicit self-contained packaging/materialization contract, including bootstrap, exercised as the actual notebook in a clean runtime. Do not treat adjacent local package files as evidence of notebook delivery.

### P1 — Packaged worker keeps a repository-only layout; missing driver dependencies

`headless.py:13–25` computes `ROOT=parents[2]`, then appends `deployment/kaggle/submission-runtime-20260912-v11/headless.py`. Importing the actual packaged file fails at its explicit file loader with a duplicated `deployment/deployment/...` path. In flat `/kaggle/working` deployment it likewise points outside the deployed assets. The builder copies the overlay headless entry but not the v11 headless/chassis tree it requires.

`kaggle_probe.py:210–231` also expects a `nooa-join` tree and imports `test_join_qualification`; that helper imports absent `test_nooa_join` at line 29. `convert_frame()` falls back to absent `convert_frame.py` when the historical repository Agent source is absent. These are definite missing package members/path assumptions, not proof that every external runtime distribution is unavailable. Resolve the entire worker/test-driver import closure and execute that closure from the deployment layout, not the repository.

### P1 — Valid local receipts can never satisfy the packaged success audit

`audit.py:148–170`, especially line 168, still requires response `model == "qwen/qwen3.8-27b"` AND `provider == "CoreWeave"`. The local relay projects requests to the mounted path and forwards raw SGLang responses (`kaggle_probe.py:142–170`; `local_wire.py:35–41,68–78`). A synthetic local response with valid choices and integer usage is accepted by `usage_receipt()` but rejected by `request_response_accounted()`. Changing only its identity to the OpenRouter/CoreWeave control makes that same audit pass. This is a bounded audit reproduction, not fabricated provider evidence. The success predicate and early-success stop therefore cannot recognize the intended local run.

Repair needed: a local runtime identity/provenance audit, not fabricated CoreWeave response fields or a change to the old successful OpenRouter audit artifacts.

### P1 — Probe final output violates its actual supervisor's acceptance contract

`kaggle_probe.py:267–271` writes a final summary containing only `error`, `stop`, `calls`, `actions`, `audit`, and `driver`. The nested real `official_launcher_runtime.py` encoded in `donor_bootstrap.py` requires `launcher_result.get("status") == "passed"` (decoded line 2245), then reconciles Solver-session request evidence (lines 2252–2257). The probe writes neither the required final status nor Solver sessions. Even hypothetical successful game/audit execution would be rejected by the supervisor. Executing only its extracted pure `expected_launcher_solver_requests` function on the probe's actual summary key schema raises `TypeError: official launcher omitted Solver session evidence`.

Repair needed: adapt and qualify both final acceptance and request-accounting contracts end-to-end; adding a status field alone is insufficient.

### P1 — Nested donor asset bypasses the declared forbidden-asset boundary

`build.py:83–89` copies the donor cell wholesale into `donor_bootstrap.py`; AST-decoding that file's `ASSETS` finds `case.json` alongside old integration/remote-entry assets. Bootstrap lines 5–8 materialize these assets. The case digest is `d0e85e5a0d5e100b3aafab45fb003007197a25de603e91c79d1ccb0bd38a5b95`. The experiment card line 10 explicitly invalidates inclusion of game-specific constructor `case.json`. The outer ALLOWLIST does not constrain embedded assets.

Additionally, fixture/local-base-URL routing at bootstrap lines 13–15 still executes the old `remote_entry.py` text despite changing the compile filename/argv to `kaggle_probe.py`. A donor fixture run does not establish that this candidate probe ran. Do not use that branch as integration evidence.

## Non-findings and remaining qualification

- The donor itself has only a markdown cell and one code cell; its nested code contains runtime installation, solver virtualenv creation and provenance producers. I do **not** assert that selecting donor cell 1 omitted extra donor prerequisite cells. The proven problems are materialization, dependency layout, retained forbidden payload and protocol incompatibility above.
- `launch_once.py` always refuses, including when its OpenRouter PASS input says PASS. This is consistent with the current card's packaging-only/no-push posture, not an accidental provider launch path. It is not a completed one-shot authorized launch gate. The card is not launch permission; the manifest explicitly says incomplete.
- Containment source inspection found `containment_base.bounded()` uses DEVNULL stdin, close_fds and an allowlisted default environment; `qualified_runner` implements identity/capability drop, chroot and seccomp. This review did not execute a real containment canary suite or target-image qualification. No claim of a containment escape is made. The broken notebook/worker path prevents treating this package as end-to-end containment-qualified.
- No HTTP continuation, model-backed GPU run, remote teardown/quota/cancellation or target-absence checks were performed. No remote call was needed to establish HOLD.
- Coordinator evidence reports four existing packaging tests passing in a disposable build and preserved successful OpenRouter evidence. I read that report; those results are attributed to the coordinator, not independently rerun here. Those tests do not exercise the failing notebook bootstrap, deployed worker imports, local receipt identity or real supervisor acceptance.

## Executable reproduction and observed output

Run from the exact repository:

```bash
cd /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus
deployment/kaggle/.venv/bin/python -B /tmp/actor-critic-kaggle-independent-repro.py
```

The script is preserved separately and in full below. It does not execute either bootstrap/model server or a launch entrypoint. Its expected defects are caught and printed; exit code 0 means the review script completed, **not** candidate PASS.

```text
CLEAN_NOTEBOOK ModuleNotFoundError No module named 'accelerator'
ASSETS_ALREADY_IN_WORKING SameFileError PosixPath('local_wire.py') and PosixPath('/tmp/independent-notebook-2wt7vw0q/local_wire.py') are the same file
HEADLESS_IMPORT FileNotFoundError [Errno 2] No such file or directory: '/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/deployment/deployment/kaggle/submission-runtime-20260912-v11/headless.py'
LOCAL_RECEIPT {'prompt_tokens': 2, 'completion_tokens': 1} ACCOUNTED False
CHANGING_ONLY_IDENTITY_TO_OPENROUTER_ACCOUNTED True
DONOR_ASSET_NAMES ['PROVIDER-SELECTION.json', 'accelerator.py', 'case.json', 'constructor.py', 'containment_base.py', 'inspection_runner.py', 'integration.py', 'integration_boundary.py', 'local_transport.py', 'model-metadata.json', 'qualified_runner.py', 'real_trace.py', 'remote_entry.py', 'request_contract.py']
EMBEDDED_CASE_SHA256 d0e85e5a0d5e100b3aafab45fb003007197a25de603e91c79d1ccb0bd38a5b95
FINAL_PROBE_KEYS ['error', 'stop', 'calls', 'actions', 'audit', 'driver'] SUPERVISOR_REQUIRES_STATUS_PASSED True
SUPERVISOR_ACCOUNTING TypeError official launcher omitted Solver session evidence
EMBEDDED_LAUNCHER_SHA256 fa7a1519595ba7f7306b2071931dc70c6784738da1f720398b0db5b1c8348afd
MANIFEST_COMPLETE False MISMATCH []
MISSING_DEPENDENCIES ['convert_frame.py', 'test_nooa_join.py', 'chassis', 'nooa-join', 'tycho_nooa']
HASHES
6ee02c0fa5a780d8d00db998553d0e4f069a8b7a02e036589685e7916df7330f analysis/actor-critic-construction-kaggle-launch/EXPERIMENT-CARD.md
892c84ac59cece5d9576ac2f17139a28d2a35bc07ec2ab3c397958b9afa47506 analysis/actor-critic-construction-kaggle-launch/MANIFEST.json
0dfa69ba645bfd680ea36543e7ec72709bf87b60b0e2f25ae5344143136d4225 analysis/actor-critic-construction-kaggle-launch/build.py
db167d0b96567337a91d1d8723b80c0518c2fd5548a18ba48dfc86f58e8f306d analysis/actor-critic-construction-kaggle-launch/kaggle_probe.py
1bcdf24fa76303614704e228628ea7939b11a07c9d489b09f9c1e6ff32b5d383 analysis/actor-critic-construction-kaggle-launch/launch_once.py
753b6bf3503c8bd3b1e485bcbc5cbd7c8c27bf57762e31aa9ee4692ebef77e8e analysis/actor-critic-construction-kaggle-launch/local_wire.py
6ee02c0fa5a780d8d00db998553d0e4f069a8b7a02e036589685e7916df7330f deployment/kaggle/actor-critic-construction-kaggle-smoke/EXPERIMENT-CARD.md
892c84ac59cece5d9576ac2f17139a28d2a35bc07ec2ab3c397958b9afa47506 deployment/kaggle/actor-critic-construction-kaggle-smoke/MANIFEST.json
e291f21804896ca6d110749352b484b7a65c4f75f67b79efd2fcdf55caaf5916 deployment/kaggle/actor-critic-construction-kaggle-smoke/accelerator.py
9c2fb321e361bc8b46cb120d9c7a51c329aaa1ec482e1a6cd2130b201a206048 deployment/kaggle/actor-critic-construction-kaggle-smoke/adapter.py
94c1a981b36a3655c6c30c6bf68f39ea66a4331aca02a0e38575fcade214bdb1 deployment/kaggle/actor-critic-construction-kaggle-smoke/audit.py
15d9bb9fe7feb1395eba0a81d8fe1575c2512ba1f5f335bb82955df83fada558 deployment/kaggle/actor-critic-construction-kaggle-smoke/construction.ipynb
91912a08294b0fd55296dd35483d2a3ec2838f9fca7b15a886df1e909482ee0d deployment/kaggle/actor-critic-construction-kaggle-smoke/construction_agent.py
1e724ffe9dd51cf0d615276de70a74cef05f2c7acc3817a5320bea356c4f3bf2 deployment/kaggle/actor-critic-construction-kaggle-smoke/construction_job.py
061207f24bf73edad2b3d45152c45654b857d2a705932d6f8288637557a32f1a deployment/kaggle/actor-critic-construction-kaggle-smoke/construction_service.py
70104a1e9c99a003926e4ed12ae8b03f371742939829065a2811830091e1bd5c deployment/kaggle/actor-critic-construction-kaggle-smoke/constructor.py
5591394bf65e6245c5204b7fc705bb0a82e00b74fcb3b26b5e5427378733327a deployment/kaggle/actor-critic-construction-kaggle-smoke/containment_base.py
649cf86b9dcdde1403eef0d6731bc9a8586fe24afc9811fee285e74beaa2e55d deployment/kaggle/actor-critic-construction-kaggle-smoke/donor_bootstrap.py
d452855b5e453abee09e55b76303890bbde6fd47d919f01478e981cfc13fc5ef deployment/kaggle/actor-critic-construction-kaggle-smoke/evaluate_visible.py
6fcdd5fe8f4ac09c9f2f437c91de747bade5cc23310120a3d24e88c930b2d21f deployment/kaggle/actor-critic-construction-kaggle-smoke/frozen_client.py
80814258d6ec65a3dcc914e64896a851f5214e2cc9f76fc0da55e139c21e5d7a deployment/kaggle/actor-critic-construction-kaggle-smoke/headless.py
eed9ca7eef0a52ef28271d168d37b48f993e671ccb63e5527330c78e52bee6a2 deployment/kaggle/actor-critic-construction-kaggle-smoke/host.py
cf4a27c87b132bd34f419f50174ae8c382693cccbe0bc51405651f24eff33892 deployment/kaggle/actor-critic-construction-kaggle-smoke/inspection_runner.py
a096ad4db6dfdec7ea8e15c0847b9ac9fbb67f26517d45b176643d8412fdaf25 deployment/kaggle/actor-critic-construction-kaggle-smoke/integration_boundary.py
db167d0b96567337a91d1d8723b80c0518c2fd5548a18ba48dfc86f58e8f306d deployment/kaggle/actor-critic-construction-kaggle-smoke/kaggle_probe.py
96bb1f90721ae8c53dc7cdbc1f77dbc434a29986f677ed6bf09abf5f11009442 deployment/kaggle/actor-critic-construction-kaggle-smoke/kernel-metadata.json
47a27785cd9cf50d757478e20536d330fa4d8f3e83b8c6d5062f995a0ee82d4d deployment/kaggle/actor-critic-construction-kaggle-smoke/live_agent.py
3c3242551d86e4c78e57d949a426d90812b363887fc3936e33f1436ecba0d0e4 deployment/kaggle/actor-critic-construction-kaggle-smoke/local_transport.py
753b6bf3503c8bd3b1e485bcbc5cbd7c8c27bf57762e31aa9ee4692ebef77e8e deployment/kaggle/actor-critic-construction-kaggle-smoke/local_wire.py
85006a4bab3ed012bdf9fb2792fb672ff5bcb00990c86b263e2a9bd543f8cf4c deployment/kaggle/actor-critic-construction-kaggle-smoke/persist.py
177d0129ba1ad32e219b6370819d635b55ff30cce7d0f523a9738f4c5151239b deployment/kaggle/actor-critic-construction-kaggle-smoke/persist_job.py
c26ae8c691cf65ba4a1eff74414343cf1d31ca3fdf963e07ba363551971f3038 deployment/kaggle/actor-critic-construction-kaggle-smoke/prefix.py
e16f93b9e148686f1a14b7755c68f7e191705c57a61a259bc2df8f76d1315807 deployment/kaggle/actor-critic-construction-kaggle-smoke/probe_lifecycle.py
4ab9116f25c88a82afdf4bf8629ce56035d7b3edd322f44f166a80625abc8111 deployment/kaggle/actor-critic-construction-kaggle-smoke/qualified_runner.py
f6fe0f88e06ab51ebd14d56967e9429fb85f67c36991b825b11c3139a8129ffe deployment/kaggle/actor-critic-construction-kaggle-smoke/real_trace.py
192f7989841d940ae290de811420fb060c8eca5968c69978e7fa7669d41cb32a deployment/kaggle/actor-critic-construction-kaggle-smoke/request_contract.py
6864e0480c215636f114eada8c8516be6a82cee86154d459d9b084715c1d43e0 deployment/kaggle/actor-critic-construction-kaggle-smoke/shared_local_solver.py
280361b2f2c0689369907b21e1e13be7ed501e185a803c077d4129d90d8822f7 deployment/kaggle/actor-critic-construction-kaggle-smoke/test_join_qualification.py
657f3e97e4b12e4ff73a4e537fe02ebc8788f5440cb401dc4023f4562fe96f70 deployment/kaggle/actor-critic-construction-kaggle-smoke/test_my_agent.py
```

## Reproduction source

```python
import ast, base64, hashlib, importlib.util, json, os, pathlib, subprocess, sys, tempfile
R=pathlib.Path('/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus')
P=R/'deployment/kaggle/actor-critic-construction-kaggle-smoke'
A=R/'analysis/actor-critic-construction-kaggle-launch'
sys.dont_write_bytecode=True
# No network allowed, including accidental library calls.
def deny(event,args):
    if event in {'socket.connect','socket.bind','socket.getaddrinfo'}: raise RuntimeError('review forbids network: '+event)
sys.addaudithook(deny)
def load(name):
    s=importlib.util.spec_from_file_location(name,P/(name+'.py')); m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
nb=json.loads((P/'construction.ipynb').read_text())
code=''.join(nb['cells'][1]['source'])
with tempfile.TemporaryDirectory(prefix='independent-notebook-') as d:
    # Redirect only absolute working directory in the actual first cell.
    old=os.getcwd(); os.chdir(d)
    try:
        try: exec(compile(code.replace("Path('/kaggle/working')",'Path('+repr(d)+')'),'actual-notebook-cell-1','exec'),{})
        except Exception as e: print('CLEAN_NOTEBOOK',type(e).__name__,str(e))
        pathlib.Path(d,'local_wire.py').write_text('')
        try: exec(compile(code.replace("Path('/kaggle/working')",'Path('+repr(d)+')'),'actual-notebook-cell-1','exec'),{})
        except Exception as e: print('ASSETS_ALREADY_IN_WORKING',type(e).__name__,str(e))
    finally: os.chdir(old)
try: load('headless')
except Exception as e: print('HEADLESS_IMPORT',type(e).__name__,str(e))
wire=load('local_wire'); audit=load('audit')
response={'model':wire.LOCAL_MODEL,'choices':[{'message':{'role':'assistant','content':'ok'}}],'usage':{'prompt_tokens':2,'completion_tokens':1}}
with tempfile.TemporaryDirectory(prefix='independent-receipt-') as d:
    p=pathlib.Path(d)
    (p/'requests.jsonl').write_text(json.dumps({'call':1,'request':{'model':wire.LOCAL_MODEL,'messages':[]}})+'\n')
    (p/'responses.jsonl').write_text(json.dumps({'call':1,'response':response,'usage':wire.usage_receipt(response)})+'\n')
    print('LOCAL_RECEIPT',wire.usage_receipt(response),'ACCOUNTED',audit.request_response_accounted(p))
    response.update(model='qwen/qwen3.8-27b',provider='CoreWeave')
    (p/'responses.jsonl').write_text(json.dumps({'call':1,'response':response})+'\n')
    print('CHANGING_ONLY_IDENTITY_TO_OPENROUTER_ACCOUNTED',audit.request_response_accounted(p))
tree=ast.parse((P/'donor_bootstrap.py').read_text())
assets=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ASSETS' for t in n.targets)))
print('DONOR_ASSET_NAMES',sorted(assets))
print('EMBEDDED_CASE_SHA256',hashlib.sha256(base64.b64decode(assets['case.json'])).hexdigest())
# Decode nested constants; never execute bootstrap or model/server code.
launcher=None
for n in ast.walk(tree):
    if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='exec' and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):
        nested=ast.parse(n.args[0].value)
        for sub in ast.walk(nested):
            if isinstance(sub,ast.Call) and isinstance(sub.func,ast.Attribute) and sub.func.attr=='b64decode' and sub.args and isinstance(sub.args[0],ast.Constant):
                raw=base64.b64decode(sub.args[0].value)
                try: text=raw.decode()
                except UnicodeDecodeError: continue
                if 'def expected_launcher_solver_requests(' in text: launcher=text
assert launcher
lt=ast.parse(launcher)
fn=next(n for n in lt.body if isinstance(n,ast.FunctionDef) and n.name=='expected_launcher_solver_requests')
ns={}; exec('from __future__ import annotations\n'+ast.unparse(fn),ns)
probe=ast.parse((P/'kaggle_probe.py').read_text())
summary=next(n.value for n in ast.walk(probe) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='summary' for t in n.targets))
keys=[ast.literal_eval(k) for k in summary.keys]
print('FINAL_PROBE_KEYS',keys,'SUPERVISOR_REQUIRES_STATUS_PASSED', 'launcher_result.get("status") != "passed"' in launcher)
try: print('SUPERVISOR_ACCOUNTING',ns['expected_launcher_solver_requests'](dict.fromkeys(keys)))
except Exception as e: print('SUPERVISOR_ACCOUNTING',type(e).__name__,str(e))
print('EMBEDDED_LAUNCHER_SHA256',hashlib.sha256(launcher.encode()).hexdigest())
m=json.loads((P/'MANIFEST.json').read_text()); mismatch=[]
for name,h in {**m['assets'],**m['files']}.items():
    if hashlib.sha256((P/name).read_bytes()).hexdigest()!=h:mismatch.append(name)
print('MANIFEST_COMPLETE',m['complete'],'MISMATCH',mismatch)
print('MISSING_DEPENDENCIES',[x for x in ['convert_frame.py','test_nooa_join.py','chassis','nooa-join','tycho_nooa'] if not (P/x).exists()])
print('HASHES')
for p in sorted([*P.glob('*'),*A.glob('*.py'),A/'EXPERIMENT-CARD.md',A/'MANIFEST.json']):
    if p.is_file():print(hashlib.sha256(p.read_bytes()).hexdigest(),str(p.relative_to(R)))

```
