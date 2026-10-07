"""Build the Kaggle sweep notebook and the private replay dataset (CARD.md pre-launch gate).

python3 -B build_notebook.py [--dataset-slug SLUG]

kaggle/notebook/mtp-depth-sweep-20261006.ipynb + kernel-metadata.json:
  cells reuse the control notebook's precache cell and its server launcher verbatim up to
  the launch; the launch becomes KaggleFactory.start(), which swaps only the speculative
  depth flags. The sweep cell runs sweep.run_sweep() under a session deadline.
kaggle/dataset/: replay code, workload.json, alpha.json, the 25 request logs, manifest.json
  (sha256 of every file). The notebook re-hashes the mounted copy before any server starts.
"""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).parent
CONTROL_NB = HERE / "../m2-winners-20261001/runs/control/dfranzen-m2-control.ipynb"
LOGS = HERE / "../m2-winners-20261001/runs/out-control"
CODE = ["replay.py", "sim.py", "payload.py", "sweep.py", "analyze.py", "mock_server.py",
        "workload.py", "CARD.md", "Q7-AUDIT.md", "workload.json", "alpha.json"]
LAUNCH_MARKER = "# ---- launch detached and wait for health ----"
KERNEL_SLUG = "mtp-depth-sweep-20261006"
SESSION_BUDGET_S = 4 * 3600  # self-imposed cap; plan needs <= 2.75 h incl. the profile launch
TAG = "mtp-depth-sweep-prelaunch-20261007"


def provenance(allow_dirty):
    """The commit that holds the shipped code; refuses uncommitted code unless allow_dirty."""
    git = lambda *a: subprocess.run(["git", *a], cwd=HERE, capture_output=True, text=True, check=True).stdout.strip()
    dirty = git("status", "--porcelain", "--", *CODE, "build_notebook.py")
    if dirty and not allow_dirty:
        raise SystemExit(f"commit the shipped code first:\n{dirty}")
    return {"code_commit": git("rev-parse", "HEAD"), "code_dirty": bool(dirty), "prelaunch_tag": TAG}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_dataset(out, slug, owner, prov):
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)  # flat: the Kaggle CLI zips subdirectories on upload
    for name in CODE:
        shutil.copy2(HERE / name, out / name)
    for f in sorted(LOGS.glob("*_p0_requests.jsonl")):
        shutil.copy2(f, out / f.name)
    (out / "provenance.json").write_text(json.dumps(prov, indent=1))
    files = sorted(p for p in out.rglob("*") if p.is_file())
    manifest = {str(p.relative_to(out)): sha256(p) for p in files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{owner}/{slug}", "licenses": [{"name": "other"}]}, indent=1))
    return manifest


def code_cell(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": src.splitlines(keepends=True)}


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


SETUP = '''import hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

NOTEBOOK_START_TIME = time.time()
SESSION_DEADLINE = time.monotonic() + {budget}  # self-imposed cap (CARD.md operations)
WHEELHOUSE_DIR    = '/kaggle/input/datasets/dfranzen/pennyroyal-v253'
MODEL_DIR         = '/kaggle/input/models/dfranzen/intel-qwen3.8-flash-next-w4a16-autoround/transformers/default/1'
DRAFT_MODEL_DIR   = '/kaggle/input/models/dfranzen/albucino-qwen3-8-flash-next-drafter/transformers/default/1'
SERVED_MODEL_NAME = 'flashnext'
SERVED_MODEL_HOST = '127.0.0.1'
SERVED_MODEL_PORT = 8001
WORKING_DIR = Path('/kaggle/working')
WORKING_DIR.mkdir(parents=True, exist_ok=True)
OUT = WORKING_DIR / 'mtp'
OUT.mkdir(exist_ok=True)

# replay dataset: locate by its manifest, re-hash every file, refuse to run on a mismatch
cands = [p.parent for p in Path('/kaggle/input').rglob('manifest.json') if (p.parent / 'replay.py').is_file()]
assert len(cands) == 1, f'expected one replay dataset, found {{cands}}'
DATA = cands[0]
manifest = json.loads((DATA / 'manifest.json').read_text())
def _sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
readback = {{k: _sha(DATA / k) for k in manifest}}
bad = sorted(k for k in manifest if readback[k] != manifest[k])
(OUT / 'readback.json').write_text(json.dumps({{'data_dir': str(DATA), 'files': readback, 'mismatches': bad}}, indent=1))
assert not bad, f'dataset readback mismatch: {{bad}}'
print(f'replay dataset {{DATA}}: {{len(readback)}} files, hashes match manifest')
NSYS = shutil.which('nsys')
print('nsys:', NSYS or 'absent -> profile deferred')
subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'])
'''

SWEEP = '''# ---- the sweep: KaggleFactory wraps the incumbent launcher; only spec depth changes ----
import signal
sys.path.insert(0, str(DATA))
import sweep, analyze

BASE_ARGS, SERVER_ENV = list(args), dict(env)

def _with_steps(base, steps):
    a = list(base)
    for flag, value in (('--speculative-num-steps', steps), ('--speculative-num-draft-tokens', steps + 1)):
        i = a.index(flag)
        a[i + 1] = str(value)
    return a

assert _with_steps(BASE_ARGS, 3) == BASE_ARGS, 'incumbent args must already be M=3'

class KaggleFactory:
    def __init__(self):
        self.precached = False
    def start(self, steps, nsys=False, log_path=None):
        if not self.precached:
            precache_model_thread.start()
            self.precached = True
        a = _with_steps(BASE_ARGS, steps)
        if nsys:
            a = [NSYS, 'launch', '--session-new=mtpprof', '--trace=cuda,nvtx',
                 '--cuda-graph-trace=node'] + a
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        (Path(log_path).parent / 'server-args.json').write_text(json.dumps(a))
        logf = open(log_path, 'ab', buffering=0)
        return subprocess.Popen(a, env=SERVER_ENV, stdout=logf, stderr=subprocess.STDOUT,
                                start_new_session=True)
    def stop(self, proc):
        sweep.stop_process_group(proc, grace=60)
        # the next launch reserves 96% of VRAM: wait until this one has really let go
        t0 = time.monotonic()
        while time.monotonic() - t0 < 180:
            q = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True)
            used = int((q.stdout.strip().splitlines() or ['0'])[0] or 0)
            if used < 2048:
                break
            time.sleep(5)
        print(f'GPU memory after stop: {used} MiB ({time.monotonic() - t0:.0f}s)', flush=True)
    def profile_hooks(self, proc):
        out = str(OUT / 'PROFILE_M3' / 'profile')
        start = lambda: subprocess.run([NSYS, 'start', '--session=mtpprof', '-o', out, '--force-overwrite=true']).returncode == 0
        stop = lambda: subprocess.run([NSYS, 'stop', '--session=mtpprof']).returncode == 0
        return start, stop

alpha = json.loads((DATA / 'alpha.json').read_text())['alpha']
PROVENANCE = {**json.loads((DATA / 'provenance.json').read_text()),
              'dataset_manifest_sha256': _sha(DATA / 'manifest.json')}
print('provenance:', PROVENANCE)
verdict = sweep.run_sweep(
    factory=KaggleFactory(), base_url=f'http://{{SERVED_MODEL_HOST}}:{{SERVED_MODEL_PORT}}', out=OUT,
    workload=DATA / 'workload.json', logs_dir=DATA, alpha=alpha, model=SERVED_MODEL_NAME,
    deadline_monotonic=SESSION_DEADLINE, profile_mode='separate' if NSYS else 'off',
    provenance=PROVENANCE)
print(json.dumps(verdict, indent=1))
'''


def build_notebook(out_dir, dataset_ref):
    nb = json.loads(CONTROL_NB.read_text())
    cells = nb["cells"]
    precache = "".join(cells[6]["source"])
    launcher = "".join(cells[12]["source"])
    assert LAUNCH_MARKER in launcher, "launcher layout changed"
    assert "precache_model_thread = threading.Thread" in precache
    launcher = launcher.split(LAUNCH_MARKER)[0] + "print('incumbent server args assembled; launches are driven by the sweep')\n"
    new = dict(nb)
    new["cells"] = [
        md_cell("# MTP-depth serving sweep (2026-10-06)\n\nSix launches M=3,1,2,3,4,3 (+ a separate "
                "profiled M=3) of the dfranzen "
                "incumbent server under closed-loop replay of the 2026-10-01 control traffic. "
                "Design, gate and decision rule: CARD.md in the replay dataset."),
        code_cell(SETUP.format(budget=SESSION_BUDGET_S)),
        code_cell(precache),
        code_cell(launcher),
        code_cell(SWEEP.replace("{{", "{").replace("}}", "}")),
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{KERNEL_SLUG}.ipynb").write_text(json.dumps(new, indent=1))
    meta = json.loads((CONTROL_NB.parent / "kernel-metadata.json").read_text())
    meta.update({"id": f"pranavagrawai/{KERNEL_SLUG}", "title": KERNEL_SLUG,
                 "code_file": f"{KERNEL_SLUG}.ipynb",
                 "dataset_sources": ["dfranzen/pennyroyal-v253", dataset_ref],
                 "competition_sources": []})
    (out_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    return out_dir / f"{KERNEL_SLUG}.ipynb"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-slug", default="mtp-depth-replay-data-20261006")
    ap.add_argument("--owner", default="pranavagrawai")
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--allow-dirty", action="store_true", help="local dry runs only")
    args = ap.parse_args()
    if not args.skip_dataset:
        prov = provenance(args.allow_dirty)
        m = build_dataset(HERE / "kaggle/dataset", args.dataset_slug, args.owner, prov)
        print("provenance:", prov)
        print(f"dataset: {len(m)} files in kaggle/dataset")
    nb = build_notebook(HERE / "kaggle/notebook", f"{args.owner}/{args.dataset_slug}")
    print("notebook:", nb)


if __name__ == "__main__":
    main()
