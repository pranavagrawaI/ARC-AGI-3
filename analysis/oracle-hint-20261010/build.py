"""Build oracle-hint kernel directories from the dfranzen control notebook. Never uploads or runs anything.

  python3 -B build.py a0      -> sessions/a0/kernel/{oracle-hint-a0-20261010.ipynb, kernel-metadata.json}, attempts.json
  python3 -B build.py hints   -> sessions/hints/... (needs hints.json: {"<game>-L<level>": {"goal": str, "plan": str}})

Differences from the control notebook, all in code cells and asserted below:
  1. USE_PRIORITY_SCHEDULING = False (a plain 10-way pool, so each attempt owns its slot for its 20 min);
  2. the customization cell: 1,200 s per attempt, concurrency 10, request logs on, refuses a competition rerun;
  3. an overlay cell before the run cell that installs overlay.py and the attempt list;
  4. the run cell builds the games with overlay.install instead of _offline_games;
  5. a results cell after the run.
"""
import hashlib, json, random, sys
from pathlib import Path

from routes import plan_text

HERE = Path(__file__).resolve().parent
CONTROL = HERE.parent / "m2-winners-20261001/runs/control"
SESSIONS = {"a0": dict(slug="oracle-hint-a0-20261010", title="Oracle Hint A0 20261010", arms=["A0", "A0"]),
            "hints": dict(slug="oracle-hint-a12-20261010", title="Oracle Hint A12 20261010", arms=["A1", "A1", "A2"])}
CAP_S = 1200

PRIORITY = "USE_PRIORITY_SCHEDULING = True\n"
RUN_GAMES = "    bm.games = _offline_games(competition_env_files)\n"
CUSTOM = f"""# ORACLE-HINT: one level-entry sprint per attempt, {CAP_S} s each, plain 10-way pool. Never a submission.
if TRUE_SUBMISSION:
    raise RuntimeError('oracle-hint experiment notebook must not run as a competition submission')
bm.solver.max_runtime_s_per_game = {CAP_S}
bm.solver.analyzer_timeout = 900.0
bm.solver.concurrency = 10
bm.solver.max_actions_per_game = None
bm.solver.save_request_logs = True
bm.n_passes = 1
bm.game_weights = None
demo_excluded_games = []
if float(getattr(target, 'max_runtime_s', 0.0) or 0.0) != 32400.0:
    raise RuntimeError(f'Expected the 32400-second notebook budget, got {{target.max_runtime_s!r}}.')
print('benchmark.solver:', bm.solver)
"""
RESULTS = """# ORACLE-HINT results: one row per attempt (benchmark.json and oracle/events.jsonl carry the same facts).
rows = [dict(name=r.game_id, state=r.state, levels_completed=r.levels_completed, actions=len(r.history),
             generated_tokens=sum(h.generated_tokens for h in r.history) + r.final_generated_tokens,
             final_wallclock_seconds=r.final_wallclock_seconds, note=r.solver_note) for r in bm.game_runs]
(WORKING_DIR / 'oracle' / 'results.json').write_text(json.dumps(rows, indent=1))
print(f"{sum(r['state'] == 'won' for r in rows)}/{len(rows)} attempts solved their level")
"""


def sha(b): return hashlib.sha256(b).hexdigest()


def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}


def attempts_for(session):
    starts = json.loads((HERE / "starts.json").read_text())
    hints = json.loads((HERE / "hints.json").read_text()) if session == "hints" else {}
    out = []
    for rep, arm in enumerate(SESSIONS[session]["arms"], 1):
        for s in starts:
            key = f"{s['game']}-L{s['level']}"
            e = hints.get(key, {})
            if arm != "A0" and not e:
                continue  # levels without a verified hint are not in the hint session (card amendment 1)
            hint = {"A0": "", "A1": e.get("goal", ""), "A2": plan_text(e["goal"], e["route"]) if e else ""}[arm]
            if arm != "A0" and not hint:
                raise SystemExit(f"missing {arm} hint for {key}")
            out.append(dict(name=f"{key}-{arm}-r{rep}", key=key, arm=arm,
                            gid=s["gid"], level=s["level"], prefix=s["prefix"], hint=hint))
    random.Random(20261010).shuffle(out)  # arm/level order independent of GPU load over the session
    return out


def main():
    session = sys.argv[1]
    cfg = SESSIONS[session]
    nb = json.loads((CONTROL / "dfranzen-m2-control.ipynb").read_text())
    cells = nb["cells"]
    src = lambda c: "".join(c["source"])
    idx = lambda pred: [i for i, c in enumerate(cells) if c["cell_type"] == "code" and pred(src(c))]
    (i_env,) = idx(lambda s: s.count(PRIORITY) == 1)
    (i_custom,) = idx(lambda s: s.startswith("# Make one-off changes to `bm`"))
    (i_run,) = idx(lambda s: s.count(RUN_GAMES) == 1)
    control_code = [src(c) for c in cells if c["cell_type"] == "code"]

    attempts = attempts_for(session)
    overlay_src = (HERE / "overlay.py").read_text()
    cells[i_env]["source"] = src(cells[i_env]).replace(PRIORITY, "USE_PRIORITY_SCHEDULING = False  # ORACLE-HINT: plain pool\n").splitlines(True)
    cells[i_custom]["source"] = CUSTOM.splitlines(True)
    cells[i_run]["source"] = src(cells[i_run]).replace(
        RUN_GAMES, "    oracle_overlay.install(bm, ORACLE_ATTEMPTS, competition_env_files)  # ORACLE-HINT\n").splitlines(True)
    overlay_cell = code(
        "# ORACLE-HINT overlay: start states replayed through the engine, per-attempt hint in the system prompt.\n"
        "import json, os, sys\n"
        "ORACLE_DIR = WORKING_DIR / 'oracle'\nORACLE_DIR.mkdir(parents=True, exist_ok=True)\n"
        f"(ORACLE_DIR / 'overlay.py').write_text({overlay_src!r})\n"
        f"(ORACLE_DIR / 'attempts.json').write_text({json.dumps(attempts)!r})\n"
        "os.environ['ORACLE_EVENTS'] = str(ORACLE_DIR / 'events.jsonl')\n"
        "sys.path.insert(0, str(ORACLE_DIR))\n"
        "import overlay as oracle_overlay\n"
        "ORACLE_ATTEMPTS = json.loads((ORACLE_DIR / 'attempts.json').read_text())\n"
        "print(f'oracle-hint: {len(ORACLE_ATTEMPTS)} attempts; arms', sorted({a[\"arm\"] for a in ORACLE_ATTEMPTS}))\n")
    cells.insert(i_run, overlay_cell)
    cells.insert(i_run + 2, code(RESULTS))

    new_code = [src(c) for c in cells if c["cell_type"] == "code"]
    kept = [s for s in new_code if not s.startswith(("# ORACLE-HINT overlay", "# ORACLE-HINT results"))]
    changed = [i for i, (a, b) in enumerate(zip(control_code, kept)) if a != b]
    assert len(kept) == len(control_code) and len(changed) == 3, changed  # env flag, customization, run cell
    assert len(new_code) == len(control_code) + 2
    assert sum(s.startswith("# ORACLE-HINT") for s in new_code) == 3  # custom, overlay, results

    d = HERE / "sessions" / session
    if (d / "launch-attempt.json").exists():
        raise SystemExit(f"{session}: launched; its notebook is evidence and is never rebuilt")
    (d / "kernel").mkdir(parents=True, exist_ok=True)
    nb_bytes = (json.dumps(nb, indent=1) + "\n").encode()
    (d / "kernel" / f"{cfg['slug']}.ipynb").write_bytes(nb_bytes)
    meta = json.loads((CONTROL / "kernel-metadata.json").read_text())
    meta.update(id=f"pranavagrawai/{cfg['slug']}", title=cfg["title"], code_file=f"{cfg['slug']}.ipynb", is_private=True)
    (d / "kernel" / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    (d / "attempts.json").write_text(json.dumps(attempts, indent=1))
    ident = dict(session=session, slug=cfg["slug"], notebook_sha256=sha(nb_bytes), overlay_sha256=sha(overlay_src.encode()),
                 starts_sha256=sha((HERE / "starts.json").read_bytes()), attempts=len(attempts), cap_s=CAP_S,
                 control_cells_changed=changed, metadata=meta)
    (d / "identity.json").write_text(json.dumps(ident, indent=2) + "\n")
    print(json.dumps({k: v for k, v in ident.items() if k != "metadata"}, indent=1))


if __name__ == "__main__":
    main()
