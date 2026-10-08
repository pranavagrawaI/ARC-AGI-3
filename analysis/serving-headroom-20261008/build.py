"""Reuse pinned installation cells and replay payload, with one model load."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent / "mtp-depth-retry1-20261007"
SLUG = "serving-headroom-20261008"
TAG = SLUG + "-prelaunch"


def build():
    nb = json.loads((PARENT / "kernel/mtp-depth-sweep-retry1-20261007.ipynb").read_text())
    for cell in nb["cells"]:
        cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs", [])
    nb["cells"][0]["source"] = ["# Serving headroom and critical path diagnostic\n", "See EXPERIMENT-CARD.md; no gameplay efficacy claim.\n"]
    guard = "".join(nb["cells"][1]["source"]).replace("mtp-depth-retry1-20261007", SLUG).replace("/kaggle/working/mtp", "/kaggle/working/headroom")
    nb["cells"][1]["source"] = guard.splitlines(keepends=True)
    setup = "".join(nb["cells"][2]["source"]).replace("4 * 3600", "6600").replace("14400", "6600").replace("/ 'mtp'", "/ 'headroom'")
    assert "6600" in setup and "OUT = WORKING_DIR / 'headroom'" in setup
    nb["cells"][2]["source"] = setup.splitlines(keepends=True)
    launcher = "".join(nb["cells"][4]["source"])
    assert launcher.count("MAXREQ=10,") == launcher.count("CUDAGRAPH_MAXBS=10,") == 1
    launcher = launcher.replace("MAXREQ=10,", "MAXREQ=16,").replace("CUDAGRAPH_MAXBS=10,", "CUDAGRAPH_MAXBS=16,")
    nb["cells"][4]["source"] = launcher.splitlines(keepends=True)
    audit = '''import hashlib, json, shutil
from pathlib import Path
import sglang, torch
root = Path(sglang.__file__).parent
out = Path('/kaggle/working/headroom/runtime-source'); out.mkdir(parents=True, exist_ok=True)
selected = ['srt/managers/scheduler_components/profiler_manager.py',
 'srt/managers/scheduler_components/metrics_reporter.py', 'srt/managers/io_struct.py',
 'srt/models/qwen4_exp.py', 'srt/models/qwen4_exp_mtp.py',
 'srt/speculative/eagle_worker_v2.py', 'srt/layers/quantization/gptq_marlin.py']
receipt = {'sglang':sglang.__version__, 'torch':torch.__version__, 'files':{}}
for relative in selected:
    source = root / relative
    if source.is_file():
        dest = out / relative; dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, dest)
        receipt['files'][relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    else: receipt['files'][relative] = 'absent'
(out / 'identity.json').write_text(json.dumps(receipt, indent=1))
print(json.dumps(receipt))
'''
    audit_cell = "run([PYTHON, '-c', " + repr(audit) + "], env=env)\n"
    audit_cell += "shutil.copy2(Path(MODEL_DIR) / 'config.json', OUT / 'target-config.json')\n"
    nb["cells"].insert(5, {"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": audit_cell.splitlines(keepends=True)})
    driver = (HERE / "probe.py").read_text()
    end = '''
import signal
import sweep
sys.path.insert(0, str(DATA))
proc = None
def cap_handler(signum, frame):
    raise TimeoutError('headroom notebook deadline')
signal.signal(signal.SIGALRM, cap_handler)
signal.alarm(max(1, int(SESSION_DEADLINE-time.monotonic())))
try:
    precache_model_thread.start()
    (OUT / 'server-args.json').write_text(json.dumps(args))
    (OUT / 'server-env.json').write_text(json.dumps({k:v for k,v in env.items() if k.startswith(('SGLANG_', 'CUDA_', 'TORCH_', 'OMP_', 'MKL_'))}))
    with (OUT / 'server.log').open('ab', buffering=0) as server_log:
        proc = subprocess.Popen(args, env=env, stdout=server_log, stderr=subprocess.STDOUT, start_new_session=True)
        ready, why = sweep.wait_health(proc, 'http://127.0.0.1:8001', 1800)
        (OUT / 'startup.json').write_text(json.dumps({'ready':ready,'why':why,'pid':proc.pid}))
        if not ready: raise RuntimeError(why)
        run_probe(DATA, OUT, SESSION_DEADLINE)
finally:
    signal.alarm(0)
    if proc is not None: sweep.stop_process_group(proc, grace=30)
    manifest = {str(p.relative_to(OUT)): _sha(p) for p in OUT.rglob('*') if p.is_file()}
    (OUT / 'output-hashes.json').write_text(json.dumps(manifest, indent=1))
'''
    # sys.path precedes imports in the embedded driver.
    source = "sys.path.insert(0, str(DATA))\n" + driver + end
    nb["cells"][-1]["source"] = source.splitlines(keepends=True)
    meta = json.loads((PARENT / "kernel/kernel-metadata.json").read_text())
    meta.update(id="pranavagrawai/" + SLUG, title=SLUG, code_file=SLUG + ".ipynb")
    dest = HERE / "kernel"
    dest.mkdir(exist_ok=True)
    (dest / meta["code_file"]).write_text(json.dumps(nb, indent=1) + "\n")
    (dest / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    build()
