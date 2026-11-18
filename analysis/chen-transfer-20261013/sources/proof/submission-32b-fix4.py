# CELL 0
# Proof Pilot — **int4-MLP DFlash + v2** submission (offline)

Single RTX PRO 6000 Blackwell, **no network**. Boots one offline sglang server (olmo3_sink 32B, GQA-packed triton attn + kv-fp8 + hybrid-SWA + **DFlash** with the **int4-MLP-quantized draft**, `--speculative-draft-model-quantization compressed-tensors`) and keeps it up, then runs the **v2 streaming** prove/verify/refine/select loop (`run_v2.py`) over the competition `test.csv` **one problem at a time** (server reused) and writes `/kaggle/working/submission.csv` as `id,answer`.

The int4-MLP draft needs the newer `dflash.py` (quant_config threaded to DFlashMLP) — re-applied at runtime from the `proof-pilot-code` dataset, so **no env rebake**. **Tune the CONFIG cell.**

**Add datasets:** `proof-pilot-code`, `proof-pilot-env`, the gptq target model, the **int4-MLP draft** model, **and** the competition. Then *Run All*.

# CELL 1
# ===== TUNABLE CONFIG — edit here; NO code-dataset re-upload needed =====
# serving knobs (forwarded as env to serve_final.sh)
SERVE_ENV = {
    'MEMFRAC':     '0.88',          # static mem-fraction (weights+KV). 0.85 safe | 0.9 OOMs
    'SWA_RATIO':   '0.2',           # hybrid-SWA full-attention token ratio
    'SGLANG_SWA_EVICTION_INTERVAL_MULTIPLIER':'0.125',  # evict SWA ~every 512 steps (default 1.0) -> ~6k/stream footprint, 3-4x concurrency
    'SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW':'1', # let prefill-prefix SWA locks evict past the window (default 0)
    'PREFILL_CG':  'tc_piecewise',  # prefill cuda-graph: tc_piecewise (few shapes) | disabled
    'CHUNKED':     '2048',          # prefill chunk size (this 95GB GPU auto=8192; pin 2048 = less prefill mem). serve_final derives the graph buckets from it
    'CTX':         '200000',        # context length
    'MAXREQ':      '48',            # max running requests (= cuda-graph max decode bs)
    'DISABLE_RADIX':'0',        # 1 = turn off radix/prefix cache (frees its KV to the pool; loses prompt-prefix sharing)
    'KV_SPLITS':   '32',            # triton decode kv-splits (sm120 long-ctx occupancy)
    'STREAM_INTERVAL':'16',         # SSE: emit every 16 tokens (default 1 = per-token, heavy HTTP)
    'NUM_DRAFT':   '8',             # dflash draft tokens (8 -> M=64 fast verify path)
    'BLOCK':       '8',             # dflash block size (8 -> M=64 fast path; 11 = draft-native/slow)
    'DRAFT_QUANT': 'compressed-tensors',  # int4-MLP draft (compressed-tensors w4a16)
    # 'LOAD_KV_SCALE':'1',          # EXPERIMENTAL: load checkpoint's calibrated fp8 KV scales (else scale 1.0). needs boot+proof-coherence check
    # WINDOW auto-derived by serve_final from the draft's sliding_window (32B int4-MLP = 512)
}
# proof-loop knobs (forwarded as CLI to run_v2.py)
LOOP = dict(init_provers=6, verify_k=3, refine_inputs=4, refine_min_seeds=1, select_bundle_n=4, selectors=5,
            call_cap=60000, concurrency=12, gen_cap=6,
            temperature=1.0, top_p=0.95,          # ALL roles use temp 1.0 / top-p 0.95
            verify_temp=1.0, select_temp=1.0)     # verifier / selector temperature
# per-problem wall-clock (budget_s, select_reserve_s): trivial public set vs the real hidden test
BUDGET_PUBLIC = (300, 60)        # 5 min/problem for the 3 visible example rows
BUDGET_HIDDEN = (4200, 900)      # 70 min/problem (Kaggle 1h + 30min grace) for the real test
CONFIG, PORT = 'w4a8-dflash', 30000

# CELL 2
# --- 1. locate + extract the env archive into writable scratch (Kaggle input is read-only) ---
import os, subprocess, glob, re, json, time, textwrap, shutil, urllib.request

WORK = os.environ.get("WORK", "/tmp/pp")
VENV, PYBASE = f"{WORK}/venv", f"{WORK}/pybase"
# persistent, DOWNLOADABLE log/output dir on Kaggle (/tmp is wiped at session end); falls back to
# WORK locally. server.log / humming stderr / proof traces land here so a crash stays debuggable.
LOGDIR = "/kaggle/working" if os.path.isdir("/kaggle/working") else WORK
os.makedirs(WORK, exist_ok=True); os.makedirs(LOGDIR, exist_ok=True)
print("LOGDIR (persistent) =", LOGDIR)

def find_archive():
    cands = ([os.environ["DS_ENV"]] if os.environ.get("DS_ENV") else []) + [
        "/kaggle/input/proof-pilot-env",
        "/kaggle/input/datasets/threerabbits/proof-pilot-env"]
    for d in cands:
        for p in ("proof-pilot-env.bin", "proof-pilot-env.tar.*", "*.bin", "*.tar.gz", "*.tar.zst"):
            g = sorted(glob.glob(f"{d}/{p}"))
            if g: return d, g[0]
    g = [x for x in sorted(glob.glob("/kaggle/input/**/proof-pilot-env.*", recursive=True))
         if x.endswith((".bin", ".tar.gz", ".tar.zst"))]
    return (os.path.dirname(g[0]), g[0]) if g else (None, None)

if not os.path.exists(f"{VENV}/bin/python"):
    DS_ENV, arc = find_archive()
    assert arc, "env archive not found — add the proof-pilot-env dataset"
    magic = open(arc, "rb").read(4)
    print("env archive =", arc, "| magic =", magic.hex())
    if magic[:2] == b"\x1f\x8b":
        subprocess.run(["tar", "-xzf", arc, "-C", WORK, "--strip-components=1"], check=True)
    elif magic == b"\x28\xb5\x2f\xfd":
        assert shutil.which("zstd"), "zstd archive but no zstd binary"
        subprocess.run(["tar", "-x", "-I", "zstd -d --long=31", "-f", arc, "-C", WORK, "--strip-components=1"], check=True)
    else:
        raise SystemExit(f"unknown archive magic {magic!r}")
assert os.path.exists(f"{VENV}/bin/python"), "venv missing after extract"
print("env staged ->", WORK)

# CELL 3
# --- 2. relocate the venv (stdlib in pybase) + restore warm caches + export paths ---
cfg = f"{VENV}/pyvenv.cfg"
txt = re.sub(r"(?m)^home = .*$", f"home = {PYBASE}/bin", open(cfg).read())
open(cfg, "w").write(txt)
HOME = os.path.expanduser("~")
for src, dst in [(f"{WORK}/flashinfer_cache", f"{HOME}/.cache/flashinfer"),
                 (f"{WORK}/humming_cache",    f"{HOME}/.humming/cache")]:
    if os.path.isdir(src):
        os.makedirs(dst, exist_ok=True); subprocess.run(["cp", "-rn", f"{src}/.", f"{dst}/"])
REPO = f"{WORK}/proof-pilot"
print("pyvenv:", next(l for l in txt.splitlines() if l.startswith("home")), "| REPO:", REPO)

# CELL 4
# --- 2b. prefer the proof-pilot-code dataset for REPO (REQUIRED: int4-MLP needs the newer
#         dflash.py + run_v2.py, which only the code dataset carries) ---
def _find_code_repo():
    cands = ([os.environ['DS_CODE']] if os.environ.get('DS_CODE') else []) + [
        '/kaggle/input/proof-pilot-code', '/kaggle/input/datasets/threerabbits/proof-pilot-code']
    for d in cands:
        for r in (d, os.path.join(d, 'proof-pilot')):
            if os.path.isdir(os.path.join(r, 'kaggle_deploy/final/notebook')): return r
    g = sorted(glob.glob('/kaggle/input/**/kaggle_deploy/final/notebook/run_v2.py', recursive=True))
    return os.path.normpath(os.path.join(os.path.dirname(g[0]), '..', '..', '..')) if g else None

_code_repo = _find_code_repo()
if _code_repo:
    REPO = _code_repo
    print('CODE dataset -> REPO =', REPO)
else:
    raise SystemExit('proof-pilot-code dataset not found — REQUIRED for the int4-MLP dflash.py + run_v2')

# CELL 5
# --- 3. locate the gptq TARGET (+ writable SWA view) and the DFlash DRAFT model ---
# TARGET = the gptq 32B (path has no 'draft'); DRAFT = the int4-MLP draft (path has 'draft'/'int4').
# Split by PATH, not by quantization_config -- the int4-MLP draft HAS one too. The target gets a
# symlink view with a writable config so enable_swa_config can patch it (Kaggle mount is read-only);
# the draft is loaded read-only as-is.
def find_models():
    md, dd = os.environ.get("MODEL_DIR"), os.environ.get("DRAFT_DIR")
    if md and dd: return md, dd
    dirs = sorted(set(os.path.dirname(c) for c in glob.glob("/kaggle/input/**/config.json", recursive=True)
                  if glob.glob(os.path.join(os.path.dirname(c), "*.safetensors*"))))
    # the int4-MLP draft ALSO has a quantization_config now, so split by path, not by quant:
    drafts  = [d for d in dirs if "draft" in d.lower()]
    targets = [d for d in dirs if "draft" not in d.lower()]
    md = md or (targets[0] if targets else None)
    dd = dd or next((d for d in drafts if "int4" in d.lower()), drafts[0] if drafts else None)
    return md, dd

MODEL_SRC, DRAFT = find_models()
assert MODEL_SRC, "gptq target not found — add the opd-32b-...-gptq-w4a16 model"
assert DRAFT, "int4-MLP draft not found — add the dflash-32b-draft-...-int4mlp model"
MODEL = f"{WORK}/model_view"; os.makedirs(MODEL, exist_ok=True)
for f in os.listdir(MODEL_SRC):
    dst = f"{MODEL}/{f}"
    if not os.path.lexists(dst): os.symlink(f"{MODEL_SRC}/{f}", dst)
cfgp = f"{MODEL}/config.json"
if os.path.islink(cfgp): os.remove(cfgp)
shutil.copy(f"{MODEL_SRC}/config.json", cfgp)                       # writable copy
subprocess.run([f"{VENV}/bin/python", f"{REPO}/kaggle_deploy/final/serve/enable_swa_config.py", MODEL], check=True)
print("target src:", MODEL_SRC)
print("target view:", MODEL)
print("draft     :", DRAFT)

# CELL 6
# --- 3b. patch the venv weight loader to SKIP baked fp8-KV scales (k_scale/v_scale) ---
# (also baked into the shipped venv; this is a no-op belt-and-suspenders for older bundles)
import py_compile
olmo2 = glob.glob(f"{VENV}/lib/python*/site-packages/sglang/srt/models/olmo2.py")[0]
src = open(olmo2).read()
if "k_scale" in src:
    print("olmo2.py already skips k_scale/v_scale (no-op)")
else:
    anchor = "        for name, loaded_weight in weights:\n"
    guard = ('            if name.endswith((".k_scale", ".v_scale")) and name not in params_dict:\n'
             "                continue\n")
    assert src.count(anchor) == 1, f"olmo2.py load-loop anchor not unique ({src.count(anchor)})"
    open(olmo2, "w").write(src.replace(anchor, anchor + guard, 1))
    print("patched olmo2.py: skip baked k_scale/v_scale")
py_compile.compile(olmo2, doraise=True)
print("olmo2.py OK")

# CELL 7
# --- 3c. expose CCCL (cuda/std/*) headers to humming's NVRTC (serve_final also symlinks these;
#         kept here for older bundles). ---
_cccl = textwrap.dedent(f"""
    import os, sys, glob
    os.environ.setdefault("HUMMING_PATH", "{WORK}")
    sys.path.insert(0, os.environ["HUMMING_PATH"])
    from humming.utils.cuda import filter_cuda_paths
    fi = sorted(glob.glob(os.path.join(sys.prefix, "lib/python*/site-packages/flashinfer/data/cccl/libcudacxx/include")))
    assert fi, "flashinfer libcudacxx (CCCL) headers not found in venv"
    fi = fi[0]
    roots = filter_cuda_paths(required_headers=["cuda_runtime.h"])["include_paths"]
    made = []
    for p in roots:
        if not os.path.isdir(p):
            continue
        cccl = os.path.join(p, "cccl")
        if not os.path.exists(os.path.join(cccl, "cuda", "std", "cstdint")) and not os.path.exists(cccl):
            try:
                os.symlink(fi, cccl); made.append(cccl)
            except OSError as e:
                print("skip", cccl, e)
    reach = any(os.path.exists(os.path.join(p, "cccl", "cuda/std/cstdint"))
                or os.path.exists(os.path.join(p, "cuda/std/cstdint")) for p in roots)
    print("cuda include roots:", roots, "| cccl reachable:", reach)
    assert reach, "CCCL still not reachable -- humming NVRTC will fail"
""")
r = subprocess.run([f"{VENV}/bin/python", "-c", _cccl],
                   env=dict(os.environ, PYBASE=PYBASE), capture_output=True, text=True)
print(r.stdout)
if r.returncode != 0:
    print("--- stderr ---\n", r.stderr[-2500:]); raise SystemExit("CCCL setup failed")

# CELL 8
# --- 3d. re-apply the DFlash sglang patches from the (latest) code dataset to the staged venv ---
# The env's venv has the dflash patches baked at BUILD time (pre-int4-MLP). The quantized draft
# needs the newer dflash.py that threads quant_config to DFlashMLP. Patches are pure .py cp
# (offline-safe), so overlay the latest from REPO onto the writable venv — no env rebake.
r = subprocess.run(['bash', f'{REPO}/deploy_kaggle/apply_dflash_patches.sh', VENV],
                   capture_output=True, text=True)
print(r.stdout[-1500:])
if r.returncode != 0:
    print('--- stderr ---\n', r.stderr[-1500:]); raise SystemExit('dflash patch re-apply failed')
dpy = glob.glob(f'{VENV}/lib/python*/site-packages/sglang/srt/models/dflash.py')[0]
assert 'quant_config' in open(dpy).read(), 'venv dflash.py lacks quant_config threading (refresh proof-pilot-code)'
print('dflash.py int4-MLP (quant_config) support: OK')

# CELL 9
# --- 4. launch the w4a8 + DFlash 32B sglang server (offline) and wait for health ---
nvrtc_lib = f"{VENV}/lib/python3.12/site-packages/nvidia/cu13/lib"   # libnvrtc-builtins.so.13.0
link_dir = f"{WORK}/_link"; os.makedirs(link_dir, exist_ok=True)     # libcuda.so for flashinfer JIT link
_libcuda = next((p for p in (
    glob.glob("/usr/local/cuda*/targets/*/lib/stubs/libcuda.so")
    + glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so")
    + glob.glob("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
    + glob.glob("/usr/local/cuda*/compat/libcuda.so*")) if os.path.exists(p)), None)
if _libcuda and not os.path.lexists(f"{link_dir}/libcuda.so"):
    os.symlink(_libcuda, f"{link_dir}/libcuda.so")
print("libcuda for flashinfer JIT link:", _libcuda)
env = dict(os.environ, CONFIG=CONFIG, PORT=str(PORT), CUDA_VISIBLE_DEVICES="0", **SERVE_ENV,
           LIBRARY_PATH=link_dir + ((":" + os.environ["LIBRARY_PATH"]) if os.environ.get("LIBRARY_PATH") else ""),
           VENV=VENV, REPO=REPO, GPTQ=MODEL, DRAFT=DRAFT, HUMMING_DIR=WORK,
           HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
           FLASHINFER_CUDA_ARCH_LIST="12.0f",   # "12.0" raises on Kaggle's <12.9 driver; "f" skips that
           LD_LIBRARY_PATH=nvrtc_lib + ((":" + os.environ["LD_LIBRARY_PATH"]) if os.environ.get("LD_LIBRARY_PATH") else ""))

SERVER_LOG = f"{LOGDIR}/server.log"
def _dump_diag():
    print("=== server.log:", SERVER_LOG, "(tail) ===\n", open(SERVER_LOG).read()[-8000:])
    hd = f"{LOGDIR}/humming_stderr"; os.makedirs(hd, exist_ok=True)
    for f in glob.glob(os.path.expanduser("~/.humming/cache/*/stderr.log")):
        t = open(f).read().strip()
        if t:
            shutil.copy(f, f"{hd}/{os.path.basename(os.path.dirname(f))}.log")
            print(f"=== {f} ===\n", t[-2500:])

logf = open(SERVER_LOG, "w")
print("sglang server log ->", SERVER_LOG)
srv = subprocess.Popen(["bash", f"{REPO}/kaggle_deploy/final/serve/serve_final.sh"],
                       env=env, stdout=logf, stderr=subprocess.STDOUT)
print("server starting (32B + draft load + humming JIT + cuda-graph; can take several minutes)...")
ready = False
for _ in range(216):                                   # up to ~18 min (draft adds load/capture)
    try:
        if urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5).status == 200:
            ready = True; break
    except Exception: pass
    if srv.poll() is not None:
        print("SERVER DIED:"); _dump_diag(); raise SystemExit(1)
    time.sleep(5)
if not ready:
    _dump_diag(); raise SystemExit("server did not become healthy in time")
print("server READY on :%d (w4a8 + DFlash)" % PORT)

# CELL 10
# --- 5. smoke generation: does the served model produce coherent math text? ---
base = f"http://127.0.0.1:{PORT}"
model_id = json.load(urllib.request.urlopen(f"{base}/v1/models", timeout=10))["data"][0]["id"]
payload = json.dumps({
    "model": model_id,
    "messages": [{"role": "user", "content": "Prove that there are infinitely many prime numbers. Keep it short."}],
    "temperature": 0.7, "max_tokens": 600,
}).encode()
req = urllib.request.Request(f"{base}/v1/chat/completions", data=payload,
                             headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
msg = r["choices"][0]["message"]
text = (msg.get("content") or "") + (msg.get("reasoning_content") or "")
print("served model:", model_id)
print("--- generation (first 800 chars) ---\n", text[:800])
smoke_ok = len(text.strip()) > 80 and ("prime" in text.lower())
print("\nsmoke_ok:", smoke_ok)

# CELL 11
# --- 5. solve the competition problems SEQUENTIALLY on the one (reused) server, via v2 ---
import csv as _csv
assert smoke_ok, 'pre-flight smoke generation failed — fix the server before the real run'

def find_test_csv():
    for p in ('/kaggle/input/competitions/ai-mathematical-olympiad-proof-pilot/test.csv',
              '/kaggle/input/ai-mathematical-olympiad-proof-pilot/test.csv'):
        if os.path.exists(p): return p
    g = sorted(glob.glob('/kaggle/input/**/test.csv', recursive=True))
    return g[0] if g else None

INPUT_CSV = os.environ.get('INPUT_CSV') or find_test_csv()
assert INPUT_CSV, 'competition test.csv not found — add the competition dataset'
OUTPUT_CSV = '/kaggle/working/submission.csv' if os.path.isdir('/kaggle/working') else f'{LOGDIR}/submission.csv'
rows = list(_csv.DictReader(open(INPUT_CSV)))
print('test.csv:', INPUT_CSV, '| problems:', len(rows), '| ->', OUTPUT_CSV)
print('ids:', [r.get('id') for r in rows])

# PUBLIC example set (3 trivial visible rows) vs the real HIDDEN test -> 5 min vs full budget.
PUBLIC_IDS = {'000aaa', '111bbb', '222ccc'}
ids = {r.get('id') for r in rows}
is_public = len(ids) > 0 and ids <= PUBLIC_IDS
DEFAULT_BUDGET, DEFAULT_RESERVE = BUDGET_PUBLIC if is_public else BUDGET_HIDDEN
BUDGET_S = int(os.environ.get('BUDGET_S', DEFAULT_BUDGET))
SELECT_RESERVE = int(os.environ.get('SELECT_RESERVE', DEFAULT_RESERVE))
print('public example set:', is_public, '| budget/problem:', BUDGET_S, 's | select_reserve:', SELECT_RESERVE, 's')

# v2 streaming pool loop (proof_agent/v2). run_v2 pre-seeds every id with a fallback answer and
# rewrites the full submission after each problem, so a wall-clock kill still leaves a complete,
# grader-valid id,answer file.
cmd = [f'{VENV}/bin/python', f'{REPO}/kaggle_deploy/final/notebook/run_v2.py',
       '--model_path', MODEL, '--input_csv', INPUT_CSV, '--output_csv', OUTPUT_CSV,
       '--logdir', f'{LOGDIR}/logs', '--base-url', base,
       '--budget-s', str(BUDGET_S), '--select-reserve', str(SELECT_RESERVE),
       '--init-provers', str(LOOP['init_provers']), '--verify-k', str(LOOP['verify_k']),
       '--refine-inputs', str(LOOP['refine_inputs']), '--refine-min-seeds', str(LOOP['refine_min_seeds']),
       '--select-bundle-n', str(LOOP['select_bundle_n']),
       '--selectors', str(LOOP['selectors']), '--call-cap', str(LOOP['call_cap']),
       '--concurrency', str(LOOP['concurrency']), '--gen-cap', str(LOOP['gen_cap']),
       '--temperature', str(LOOP['temperature']), '--top-p', str(LOOP['top_p']),
       '--verify-temp', str(LOOP['verify_temp']), '--select-temp', str(LOOP['select_temp'])]
print('running v2 streaming loop over', len(rows), 'problems, budget', BUDGET_S, 's each ...')
print('   (tail realtime events:  /kaggle/working/logs/trace_<id>.json.events.jsonl)')
subprocess.run(cmd, env=env, check=True)

# CELL 12
# --- 6. inspect submission.csv + shut the server down ---
import pandas as pd
df = pd.read_csv(OUTPUT_CSV)
print('submission.csv:', OUTPUT_CSV, '| shape', df.shape, '| columns', list(df.columns))
for _, r in df.iterrows():
    a = str(r['answer'])
    print(f"\nid {r['id']} | answer_len {len(a)}\n  {a[:300]}")

srv.terminate()
try: srv.wait(timeout=30)
except Exception: srv.kill()
print('\nserver stopped. logs/traces in', LOGDIR)