"""Generate the Kaggle probe notebook from the dfranzen control notebook (cells 0-12 verbatim, speculative decoding off).

  python3 make_kernel.py NAME VARIANTS   -> kernel-NAME/ (notebook + metadata); dataset/ (items, scorer, KF system prompt)
"""
import json, shutil, sys
from pathlib import Path
HERE = Path(__file__).parent
SRC = HERE.parents[0] / "m2-winners-20261001/runs/control/dfranzen-m2-control.ipynb"
NAME, VARIANTS = sys.argv[1], sys.argv[2]
KID, DSID = f"pranavagrawai/{NAME}", "pranavagrawai/arc3-likelihood-probe-items-v1"
KDIR = HERE / f"kernel-{NAME}"; KDIR.mkdir(exist_ok=True)

nb = json.loads(SRC.read_text())
cells = nb["cells"][:13]
c12 = "".join(cells[12]["source"])
for old, new in [("    SPEC=True,\n", "    SPEC=False,  # probe: prompt logprobs only; no drafter\n"), ("    FRSPEC=True,\n", "    FRSPEC=False,\n")]:
    assert c12.count(old) == 1, old; c12 = c12.replace(old, new)
cells[12]["source"] = c12
c4 = "".join(cells[4]["source"])
old = "SERVER_STARTUP_TIMEOUT = 12 * 60"; assert c4.count(old) == 1; cells[4]["source"] = c4.replace(old, "SERVER_STARTUP_TIMEOUT = 30 * 60")
probe = r'''# Likelihood probe: wait for the server, then score every item (see probe.py docstring).
import glob, subprocess, sys, time, urllib.request
from pathlib import Path
items = glob.glob('/kaggle/input/**/items.jsonl.xz', recursive=True); assert len(items) == 1, items
D = Path(items[0]).parent
t0 = time.time()
while True:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{SERVED_MODEL_PORT}/health", timeout=5) as r:
            if r.status == 200: break
    except Exception: pass
    if proc.poll() is not None: show_log_tail(); raise RuntimeError("server exited")
    if time.time() - t0 > 40 * 60: show_log_tail(); raise RuntimeError("server not healthy after 40 min")
    time.sleep(5)
print(f"server healthy; {time.time() - NOTEBOOK_START_TIME:.0f}s since notebook start", flush=True)
out = WORKING_DIR / "probe_results.jsonl"
PROBE_PY = PYTHON if subprocess.run([PYTHON, "-c", "import jinja2, tokenizers"], env=env).returncode == 0 else sys.executable
print("probe interpreter", PROBE_PY, flush=True)
cmd = [PROBE_PY, str(D / "probe.py"), str(D / "items.jsonl.xz"), str(out), "--model-dir", MODEL_DIR,
       "--kf-system", str(D / "kf_system.txt"), "--local-template", str(D / "chat_template.local.jinja"),
       "--url", f"http://127.0.0.1:{SERVED_MODEL_PORT}", "--model", SERVED_MODEL_NAME, "--budget-s", str(3 * 3600), "--variants", "%s"]''' % VARIANTS + r'''
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
for line in p.stdout: print(line, end="", flush=True)
print("probe exit", p.wait(), flush=True)
show_log_tail()
'''
cells.append({"cell_type": "markdown", "metadata": {}, "source": "## Likelihood probe\n\nTeacher-forced scoring of logged decision points; no gameplay."})
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": probe})
nb["cells"] = cells
(KDIR / f"{NAME}.ipynb").write_text(json.dumps(nb, indent=1))
meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
meta.update(id=KID, title=KID.split("/")[1], code_file=f"{NAME}.ipynb",
            dataset_sources=meta["dataset_sources"] + [DSID],
            )
meta["competition_sources"] = meta["competition_sources"]
(KDIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
for f, dst in [("items.jsonl.xz", "items.jsonl.xz"), ("probe.py", "probe.py")]: shutil.copy(HERE / f, HERE / "dataset" / dst)
shutil.copy("/tmp/lp/kf_system.txt", HERE / "dataset/kf_system.txt"); shutil.copy("/tmp/lp/kf_system.txt", HERE / "kf_system.txt")
shutil.copy(Path.home() / ".cache/flashnext-tok/chat_template.jinja", HERE / "dataset/chat_template.local.jinja")
(HERE / "dataset/dataset-metadata.json").write_text(json.dumps({"id": DSID, "title": DSID.split("/")[1], "licenses": [{"name": "CC0-1.0"}]}, indent=2))
print("ok", len(json.dumps(nb)) // 1024, "KiB notebook")
