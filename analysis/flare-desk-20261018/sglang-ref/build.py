"""Build the same-session SGLang reference for the FLARE smoke: full-image vs history-stripped replay.

Smoke 3d replays the 10-01 control traffic with history images stripped (ARC3_HISTORY_IMAGE_KEEP=0),
because Flare accepts at most 4 images per prompt. The SGLang M3A reference (U_overlap 477.6) was
measured with full images. This kernel starts the incumbent SGLang server once (the MTP sweep's frozen
setup/precache/launcher cells, M=3) and runs three S=10 replay blocks in ABA order: FULL, KEEP0, FULL,
each with warm-up ≤600 s, 30 s settle and a 480 s measured block, the same as the Flare smoke. Outputs:
U_resp/U_overlap per block, so r = Flare_keep0 / SGLang_keep0 is same-workload, and KEEP0/FULL on
SGLang gives the image-strip effect on the incumbent itself.

  python3 -B build.py   # writes kernel/{sglang-keep0-ref-20261020.ipynb, kernel-metadata.json}
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SWEEP_NB = HERE.parents[1] / "mtp-depth-20261006" / "kaggle" / "notebook" / "mtp-depth-sweep-20261006.ipynb"
SLUG = "sglang-keep0-ref-20261020"

REF = r'''# ---- same-session reference: one SGLang M=3 launch, three S=10 blocks (FULL, KEEP0, FULL) ----
import threading
sys.path.insert(0, str(DATA))
import sweep, replay, payload
BASE_URL = f'http://{SERVED_MODEL_HOST}:{SERVED_MODEL_PORT}'
REF = OUT / 'ref'; REF.mkdir(parents=True, exist_ok=True)
RESULT = {'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'args': list(args)}
def save():
    (REF / 'result.json').write_text(json.dumps(RESULT, indent=1, default=str))

_HIST_PLACEHOLDER = '[grid image omitted from history]'   # same transform as flare smoke 3d
def _is_img(part):
    return isinstance(part, dict) and part.get('type') == 'image_url'
def keep_current_images(messages):
    seen, out = 0, []
    for m in reversed(messages):
        c = m.get('content')
        if isinstance(c, list) and any(_is_img(x) for x in c):
            seen += 1
            if seen > 1:
                m = {**m, 'content': [{'type': 'text', 'text': _HIST_PLACEHOLDER} if _is_img(x) else x for x in c]}
        out.append(m)
    return out[::-1]
FULL_BUILD = payload.build
def KEEP0_BUILD(rec, model, max_tokens):
    body = FULL_BUILD(rec, model, max_tokens)
    body['messages'] = keep_current_images(body['messages'])
    return body

def u_estimates(events_path):   # identical to the flare smoke's estimator
    ev = [json.loads(l) for l in open(events_path)]
    ph = {e['name']: e['t'] for e in ev if e['kind'] == 'phase'}
    m0, m1 = ph.get('measure_start'), ph.get('measure_end')
    if m0 is None or m1 is None:
        return {'valid': False}
    resp = [e for e in ev if e['kind'] == 'response' and e['status'] == 200]
    inb = [e for e in resp if m0 <= e['t'] <= m1]
    tok = sum(e.get('completion_tokens') or 0 for e in inb)
    ov = sum((e.get('completion_tokens') or 0) * max(0, min(e['t'], m1) - max(e['t_send'], m0)) / (e['t'] - e['t_send'])
             for e in resp if e['t'] > e['t_send'])
    return {'valid': True, 'block_s': m1 - m0, 'responses_block': len(inb), 'U_resp': tok / (m1 - m0),
            'U_overlap': ov / (m1 - m0), 'errors': sum(1 for e in ev if e['kind'] == 'response' and e['status'] != 200)}

precache_model_thread.start()
logf = open(REF / 'server.log', 'ab', buffering=0)
proc = subprocess.Popen(list(args), env=dict(env), stdout=logf, stderr=subprocess.STDOUT, start_new_session=True)
ok, why = sweep.wait_health(proc, BASE_URL, 20 * 60)
RESULT['health'] = why; save()
print('health:', ok, why, flush=True)
alpha = json.loads((DATA / 'alpha.json').read_text())['alpha']
RESULT['blocks'] = []
try:
    for label, build in (('FULL_a', FULL_BUILD), ('KEEP0', KEEP0_BUILD), ('FULL_b', FULL_BUILD)):
        if not ok or SESSION_DEADLINE - time.monotonic() < 20 * 60:
            RESULT['blocks'].append({'label': label, 'skipped': why if not ok else 'deadline'}); save(); continue
        payload.build = build
        replay.payload.build = build
        d = REF / label; d.mkdir(exist_ok=True)
        before = set(threading.enumerate())
        r = replay.Replay(workload=DATA / 'workload.json', logs_dir=DATA, alpha=alpha, base_url=BASE_URL,
                          model=SERVED_MODEL_NAME, events_path=d / 'events.jsonl', slots=10,
                          header={'label': label, 'server': 'sglang-m3', 'slots': 10})
        r.run(measure_s=480.0, warmup_max_s=600.0, settle_s=30.0)
        time.sleep(5)
        row = {'label': label, **u_estimates(d / 'events.jsonl')}
        try:
            row['summarize'] = replay.summarize(d / 'events.jsonl')
        except Exception as e:
            row['summarize'] = repr(e)[:300]
        RESULT['blocks'].append(row); save()
        print(label, {k: row.get(k) for k in ('U_resp', 'U_overlap', 'responses_block', 'errors')}, flush=True)
        t = time.monotonic()
        while time.monotonic() - t < 900 and [th for th in threading.enumerate() if th not in before]:
            time.sleep(5)
finally:
    sweep.stop_process_group(proc, grace=60)
    full = [b for b in RESULT['blocks'] if b.get('label', '').startswith('FULL') and b.get('valid')]
    keep = [b for b in RESULT['blocks'] if b.get('label') == 'KEEP0' and b.get('valid')]
    if full and keep:
        f = sum(b['U_overlap'] for b in full) / len(full)
        RESULT['keep0_over_full_U_overlap'] = keep[0]['U_overlap'] / f
    save()
print(json.dumps({k: RESULT.get(k) for k in ('health', 'keep0_over_full_U_overlap')}, indent=1))
'''


def code_cell(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": src.splitlines(keepends=True)}


def main():
    nb = json.loads(SWEEP_NB.read_text())
    cells = nb["cells"]
    setup, precache, launcher = ("".join(cells[i]["source"]) for i in (1, 2, 3))
    assert "SESSION_DEADLINE" in setup and "precache_model_thread = threading.Thread" in precache
    assert launcher.rstrip().endswith("print('incumbent server args assembled; launches are driven by the sweep')")
    compile(REF, SLUG, "exec")
    new = dict(nb)
    new["cells"] = [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# SGLang same-workload reference for the FLARE smoke (2026-10-20)\n\nOne incumbent SGLang M=3 launch; "
            "S=10 closed-loop replay of the 2026-10-01 control traffic in three blocks: full images, history images "
            "stripped (ARC3_HISTORY_IMAGE_KEEP=0), full images. Card: analysis/flare-desk-20261018/smoke3d/SMOKE3D-CARD.md."]},
        code_cell(setup), code_cell(precache), code_cell(launcher), code_cell(REF)]
    d = HERE / "kernel"
    d.mkdir(exist_ok=True)
    (d / f"{SLUG}.ipynb").write_text(json.dumps(new, indent=1))
    meta = json.loads((SWEEP_NB.parent / "kernel-metadata.json").read_text())
    meta.update(id=f"pranavagrawai/{SLUG}", title=SLUG.replace("-", " "), code_file=f"{SLUG}.ipynb")
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(d, meta["dataset_sources"], meta["model_sources"])


if __name__ == "__main__":
    main()
