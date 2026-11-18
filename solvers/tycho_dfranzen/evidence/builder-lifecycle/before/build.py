"""Deterministic local assembly; no GPU, Kaggle push, or submission."""
from __future__ import annotations
import base64
import difflib
import hashlib
import json
import shutil
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CANONICAL = ROOT/"solvers/tycho_nooa"
DONOR = ROOT/"analysis/m2-winners-20261001/audit/dfranzen-repo"
CONTROL = ROOT/"analysis/m2-winners-20261001/runs/control"
WHEELS = ROOT/"deployment/kaggle/tycho-nooa-full-wheelhouse-v11"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError(f"source seam changed: {before[:80]}")
    return source.replace(before, after, 1)


def assemble(destination):
    files, provenance, patches = {}, [], []
    if destination.exists():
        shutil.rmtree(destination)
    disposition = json.loads((CANONICAL/"MODULE-DISPOSITION.json").read_text())
    for name in disposition["runtime_closure"]:
        path = CANONICAL/"src/tycho_nooa"/(name+".py")
        source = path.read_text()
        original = source
        if name == "__init__":
            source = replace_once(source, "from tycho_nooa.agent import TychoNooaAgent",
                "def __getattr__(name):\n    if name == 'TychoNooaAgent':\n        from tycho_nooa.agent import TychoNooaAgent\n        return TychoNooaAgent\n    raise AttributeError(name)")
        if name == "worker":
            source = replace_once(source, "from tycho_nooa.agent import TychoNooaAgent",
                "from hybrid.agent import HybridAgent as TychoNooaAgent")
        if name == "agent":
            source = replace_once(source, "from tycho_nooa.llm_trace import LLMTrace",
                "from hybrid.policy import HybridLLMTrace as LLMTrace")
        if name == "builder":
            source = replace_once(source, "super().__init__(llm=llm)",
                "from hybrid.policy import unbounded_render\n        super().__init__(llm=llm, truncation=unbounded_render())")
        if name == "strategy":
            source = replace_once(source, "                prompt=prompt,\n                metadata={",
                "                prompt=prompt + '\\nOptional computation: run_python preloads dfr; dfr.segment(grid) supplies connected components with numeric colors. See notes/hybrid_tools.md.',\n                images=__import__('hybrid.policy', fromlist=['actor_images']).actor_images(runtime.agent, turn),\n                metadata={")
            source = replace_once(source, '                "chat_template_kwargs": {"enable_thinking": False}',
                '                "top_k": 20,\n                "chat_template_kwargs": {"enable_thinking": False}')
        if name == "wm_probe":
            source = replace_once(source, "    _PREAMBLE = ToolExecutor._PREAMBLE + (",
                "    _PREAMBLE = ToolExecutor._PREAMBLE + (\n        \"import dfranzen_helpers as dfr\\n\"")
            source = replace_once(source, "    directories = _mount_parent_directories(workspace, venv)",
                "    base_python = Path(sys.base_prefix).resolve(strict=True)\n    directories = _mount_parent_directories(workspace, venv, base_python)")
            source = replace_once(source, '        str(venv),\n        str(venv),\n        "--dev",',
                '        str(venv),\n        str(venv),\n        "--ro-bind",\n        str(base_python),\n        str(base_python),\n        "--dev",')
        if name == "llm":
            source = replace_once(source, "    client_type = \"responses\" if use_responses == \"1\" else \"completion\"",
                "    overrides['extra_body'] = {**overrides.get('extra_body', {}), 'chat_template_kwargs': {'enable_thinking': True}, 'top_k': 20}\n    overrides['top_p'] = 0.95\n    client_type = \"responses\" if use_responses == \"1\" else \"completion\"")
        rel = "tycho_nooa/"+path.name
        if source != original:
            source = "# Modified for the Tycho/dfranzen candidate; see canonical-candidate.patch.\n"+source
        files[rel] = source.encode()
        provenance.append({"path":rel,"source":str(path.relative_to(ROOT)),
                           "before":sha(path.read_bytes()),"after":sha(files[rel])})
        if source != original:
            patches.append(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),
                        fromfile="a/"+rel,tofile="b/"+rel)))
    for item in disposition["runtime_files"]:
        path = CANONICAL/"src/tycho_nooa"/item['path']
        files['tycho_nooa/'+item['path']] = path.read_bytes()
    for prefix, base in [("inference",DONOR/"ARC3-Inference/inference"),
                         ("taaf",DONOR/"tufa-arc-agi-framework/src/taaf"),
                         ("hybrid",HERE/"src/hybrid")]:
        for path in sorted(base.rglob("*.py")):
            rel = prefix+"/"+path.relative_to(base).as_posix()
            files[rel] = path.read_bytes()
            provenance.append({"path":rel,"source":str(path.relative_to(ROOT)),
                               "before":sha(files[rel]),"after":sha(files[rel])})
        if prefix == "inference":
            for path in base.rglob("*.json"):
                files[prefix+"/"+path.relative_to(base).as_posix()] = path.read_bytes()
    transport_path = ROOT/"deployment/kaggle/runtime/local_solver.py"
    transport = transport_path.read_text()
    transport = replace_once(transport, "_MODEL_GENERATION_ADMISSION = threading.BoundedSemaphore(1)",
                            "from contextlib import nullcontext\n_MODEL_GENERATION_ADMISSION = nullcontext()")
    transport = replace_once(transport, '        "ARC_LLM_API_KEY",', '        "TYCHO_NOOA_SCRIPTED_TOOL_CALLS",\n        "ARC_LLM_API_KEY",')
    transport = replace_once(transport, '                cwd=session_root,',
                            '                cwd=session_root,\n                start_new_session=True,')
    files["hybrid_local_solver.py"] = transport.encode()
    provenance.append({"path":"hybrid_local_solver.py",
        "source":str(transport_path.relative_to(ROOT)),
        "before":sha(transport_path.read_bytes()),"after":sha(files['hybrid_local_solver.py'])})
    files["LICENSE-APACHE-2.0"] = (HERE/"LICENSE-APACHE-2.0").read_bytes()
    files["NOTICE.md"] = (HERE/"NOTICE.md").read_bytes()
    files["worker_entry.py"] = b"from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom hybrid.worker import main\nmain()\n"
    files["runner_entry.py"] = b"from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent))\nfrom hybrid.runner import main\nmain()\n"
    for rel,data in files.items():
        if rel.endswith(".py"):
            compile(data,rel,"exec")
        path = destination/rel
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    return files, provenance, ''.join(patches)


def build():
    files, provenance, patch = assemble(HERE/"runtime")
    (HERE/"canonical-candidate.patch").write_text(patch)
    packed = base64.b64encode(zlib.compress(json.dumps({rel:base64.b64encode(data).decode()
                    for rel,data in sorted(files.items())}).encode(),9)).decode()
    payload = base64.b64decode(packed)
    bundle = HERE/'source-bundle'
    bundle.mkdir(exist_ok=True)
    (bundle/'hybrid-runtime.zlib').write_bytes(payload)
    for notice in ('NOTICE.md','LICENSE-APACHE-2.0'):
        shutil.copy2(HERE/notice,bundle/notice)
    (bundle/'dataset-metadata.json').write_text(json.dumps({
        'id':'pranavagrawai/tycho-dfranzen-source-v5',
        'title':'Tycho dfranzen source v5',
        'licenses':[{'name':'other'}]},indent=2)+'\n')
    hashes = {rel:sha(data) for rel,data in sorted(files.items())}
    wheel_manifest = (WHEELS/"wheelhouse-manifest.json").read_bytes()
    nb_path = CONTROL/"dfranzen-m2-control.ipynb"
    notebook = json.loads(nb_path.read_text())
    for cell in notebook['cells']:
        if cell['cell_type']=='code':
            cell['outputs']=[]
            cell['execution_count']=None
    notebook['cells'][0]['source'] = ["# Tycho/Nooa on dfranzen's Milestone 2 runtime\n\n",
        "Derived from dfranzen's notebook and Apache-2.0 source bundle; modified for the local hybrid. Source notices and license are included in the runtime payload.\n\n",
        "Offline candidate. Default: 3-game/16-action/10-minute smoke. Set HYBRID_RUN_MODE=public25 before running to use inherited public-25 budgets. CPU qualification does not establish GPU compatibility or score improvement.\n"]
    # Preserve model, serving, source patch, monitoring and game-loading cells.
    source = ''.join(notebook['cells'][4]['source'])
    source = replace_once(source,"WORKING_DIR.mkdir(parents=True, exist_ok=True)",
        "WORKING_DIR.mkdir(parents=True, exist_ok=True)\nif TRUE_SUBMISSION:\n    raise RuntimeError('Hybrid candidate requires offline qualification first')")
    notebook['cells'][4]['source']=source.splitlines(True)
    extras_manifest=(HERE/"runtime-deps/hybrid-deps-manifest.json").read_bytes()
    install = f'''# Install hash-bound candidate source and isolated Solver dependencies.
import base64, hashlib, json, os, shutil, subprocess, sys, zlib
from pathlib import Path
HYBRID_ROOT = Path('/kaggle/working/tycho-dfranzen-runtime')
HYBRID_ROOT.mkdir(parents=True, exist_ok=True)
_hybrid_hashes = {hashes!r}
_payload_candidates=[p for p in Path('/kaggle/input').rglob('hybrid-runtime.zlib')
                     if hashlib.sha256(p.read_bytes()).hexdigest()=={sha(payload)!r}]
if len(_payload_candidates)!=1:
    raise RuntimeError('Attach exact hash-bound tycho-dfranzen-source-v5 dataset')
_hybrid_files = json.loads(zlib.decompress(_payload_candidates[0].read_bytes()))
for _rel, _encoded in _hybrid_files.items():
    _data = base64.b64decode(_encoded)
    if hashlib.sha256(_data).hexdigest() != _hybrid_hashes[_rel]:
        raise RuntimeError('Hybrid payload hash mismatch: '+_rel)
    _path = HYBRID_ROOT / _rel
    _path.parent.mkdir(parents=True, exist_ok=True)
    _path.write_bytes(_data)
_candidates = [p for p in Path('/kaggle/input').rglob('wheelhouse-manifest.json')
               if hashlib.sha256(p.read_bytes()).hexdigest() == {sha(wheel_manifest)!r}]
if len(_candidates) != 1:
    raise RuntimeError('Attach the exact tycho-nooa-solver-wheelhouse-cp312-v11 dataset')
_wh = _candidates[0].parent
_manifest = json.loads(_candidates[0].read_text())
for _item in _manifest['wheels']:
    _wheel = _wh / _item['filename']
    if hashlib.sha256(_wheel.read_bytes()).hexdigest() != _item['sha256']:
        raise RuntimeError('Solver dependency hash mismatch: '+_wheel.name)
_extra_candidates=[p for p in Path('/kaggle/input').rglob('hybrid-deps-manifest.json')
                   if hashlib.sha256(p.read_bytes()).hexdigest() == {sha(extras_manifest)!r}]
if len(_extra_candidates) != 1:
    raise RuntimeError('Attach exact tycho-dfranzen-runtime-deps dataset')
_extra_root=_extra_candidates[0].parent
for _item in json.loads(_extra_candidates[0].read_text())['wheels']:
    if hashlib.sha256((_extra_root/_item['filename']).read_bytes()).hexdigest()!=_item['sha256']:
        raise RuntimeError('Hybrid dependency wheel mismatch')
HYBRID_ENV = Path('/tmp/tycho-dfranzen-env')
_uv = shutil.which('uv')
if not _uv:
    raise RuntimeError('uv required to prepare isolated Solver environment')
subprocess.run([_uv,'venv','--python',sys.executable,str(HYBRID_ENV)],check=True)
HYBRID_PYTHON = HYBRID_ENV/'bin/python'
subprocess.run([_uv,'pip','install','--python',str(HYBRID_PYTHON),'--offline','--no-index',
    '--find-links',str(_wh),'--find-links',str(_extra_root),
    'nooa==0.0.1.dev95','arc-agi-3-tycho==0.1.0',
    'imageio==2.37.3','imageio-ffmpeg==0.6.0','scipy==1.18.1'],check=True)
# Match the ARC runtime installed by the donor notebook, using competition wheels.
import importlib.metadata
_arc_reqs = ['arc-agi=='+importlib.metadata.version('arc-agi'),
             'arcengine=='+importlib.metadata.version('arcengine')]
subprocess.run([_uv,'pip','install','--python',str(HYBRID_PYTHON),'--offline','--no-index',
    '--find-links','/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels',
    '--find-links',str(_wh),*_arc_reqs],check=True)
os.environ.update({{'ARC_LLM_MODEL': SERVED_MODEL_NAME,
    'ARC_LLM_BASE_URL': SERVER_BASE_URL, 'ARC_LLM_API_KEY': 'local-not-a-secret',
    'ARC_LLM_CONTEXT_WINDOW': str(128*1024), 'ARC_LLM_MAX_TOKENS': str(12*1024),
    'ARC_LLM_USE_RESPONSES': '0', 'ARC_LLM_RETRIES': '0',
    'ARC_LLM_TIMEOUT_SECONDS': '900',
    'TYCHO_MODE': 'orchestrator', 'TYCHO_NOOA_TREATMENT': 'control',
    'TYCHO_SANDBOX_RUNTIME': 'host', 'TYCHO_KAGGLE_HARDENED_HOST': '1',
    'LITELLM_LOCAL_MODEL_COST_MAP': 'True'}})
os.environ.setdefault('HYBRID_RUN_MODE','smoke')
'''
    # Insert after the original ARC runtime installation, before source/serving.
    notebook['cells'].insert(9,{'cell_type':'code','metadata':{},'execution_count':None,
                               'outputs':[],'source':install.splitlines(True)})
    run_cell=notebook['cells'][21]
    source=''.join(run_cell['source'])
    old='await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)'
    replacement='''# Keep game construction and donor budgets; replace only the reasoning solver.
_hybrid_bm = WORKING_DIR/'hybrid-benchmark.pkl'
_hybrid_target = WORKING_DIR/'hybrid-target.pkl'
with _hybrid_bm.open('wb') as _stream: pickle.dump(bm,_stream)
with _hybrid_target.open('wb') as _stream: pickle.dump(target,_stream)
_command=[str(HYBRID_PYTHON),str(HYBRID_ROOT/'runner_entry.py'),
          str(_hybrid_bm),str(_hybrid_target)]
if soft_end: _command.extend(['--soft-end',soft_end.isoformat()])
_hybrid_process=subprocess.Popen(_command,env=dict(os.environ))
if '_qualification_root' in globals():
    _fields=Path(f'/proc/{_hybrid_process.pid}/stat').read_text().rsplit(') ',1)[1].split()
    (_qualification_root/'qualification-runner.json').write_text(json.dumps({
        'pid':_hybrid_process.pid,'start':int(_fields[19])}))
try:
    while _hybrid_process.poll() is None:
        await __import__('asyncio').sleep(1)
    if _hybrid_process.returncode:
        raise RuntimeError(f'Hybrid benchmark exited {_hybrid_process.returncode}; inspect tycho-dfranzen artifacts')
finally:
    if _hybrid_process.poll() is None:
        _hybrid_process.terminate()
        try:
            _hybrid_process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            _hybrid_process.kill()
            _hybrid_process.wait(timeout=5)
'''
    source=replace_once(source,old,replacement)
    run_cell['source']=source.splitlines(True)
    # Viewer is Duck-specific; do not display it as a hybrid reasoning trace.
    notebook['cells'][-1]['source']=["print('Hybrid results: /kaggle/working/tycho-dfranzen/benchmark.json')\n",
        "print('Inspect per-game host-actions.jsonl, llm-trace.jsonl and decision-checkpoints.')\n"]
    out=HERE/'kernel/tycho-dfranzen.ipynb'
    out.write_text(json.dumps(notebook,indent=1)+'\n')
    metadata=json.loads((CONTROL/'kernel-metadata.json').read_text())
    metadata.update(id='pranavagrawai/tycho-dfranzen-candidate',title='Tycho Nooa on dfranzen runtime',
                    code_file=out.name)
    metadata['dataset_sources'] += ['pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11',
                                    'pranavagrawai/tycho-dfranzen-runtime-deps',
                                    'pranavagrawai/tycho-dfranzen-source-v5']
    (HERE/'kernel/kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    manifest={'schema':'tycho-dfranzen-candidate-v3','source_payload_sha256':sha(payload),'donor_notebook':str(nb_path.relative_to(ROOT)),
              'donor_notebook_sha256':sha(nb_path.read_bytes()),'notebook_sha256':sha(out.read_bytes()),
              'canonical_lock_sha256':sha((CANONICAL/'uv.lock').read_bytes()),
              'wheelhouse_manifest_sha256':sha(wheel_manifest),
              'extra_deps_manifest_sha256':sha(extras_manifest),'source_files':hashes,
              'source_provenance':provenance,
              'canonical_sources':[p for p in provenance if p['path'].startswith('tycho_nooa/')],
              'build_script_sha256':sha(Path(__file__).read_bytes()),
              'status':'local candidate; GPU/gameplay unqualified'}
    (HERE/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'notebook':str(out),'sha256':manifest['notebook_sha256'],'files':len(files)}))
    return manifest

if __name__=='__main__':
    build()
