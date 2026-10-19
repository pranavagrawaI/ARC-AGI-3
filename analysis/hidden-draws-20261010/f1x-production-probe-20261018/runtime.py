"""Probe-only evidence capture; embedded into the diagnostic notebook."""
import json
import threading
import time
import traceback


def probe_snapshot(benchmark):
    rows = []
    for run in list(benchmark.game_runs):
        rows.append({
            'game_id': run.game_id, 'state': run.state,
            'actions': len(run.history), 'levels': run.levels_completed,
            'tokens': run.final_generated_tokens, 'solver_note': run.solver_note,
        })
    return {'time': time.time(), 'games': rows}


async def probe_run(benchmark, target, working, soft_end, server):
    stop = threading.Event()
    progress = working / 'probe-progress.jsonl'

    def sample():
        while not stop.is_set():
            try:
                row = probe_snapshot(benchmark)
                with progress.open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
            except Exception:
                with (working / 'probe-observer-errors.txt').open('a') as stream:
                    stream.write(traceback.format_exc())
            stop.wait(15)

    observer = threading.Thread(target=sample, daemon=True)
    observer.start()
    try:
        await benchmark.run(soft_end_time=soft_end, runtime_environment=target,
                            minimal_diagnostics=True)
    except BaseException:
        (working / 'probe-exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        stop.set()
        observer.join(timeout=3)
        try:
            final = probe_snapshot(benchmark)
            (working / 'probe-final.json').write_text(json.dumps(final, indent=2) + '\n')
            print('PROBE_FINAL', json.dumps(final), flush=True)
        finally:
            server.stop()
    rows = final['games']
    bad = [r for r in rows if r['state'] == 'crashed'
           or str(r['solver_note'] or '').startswith(('error:', 'analyzer failed'))]
    gates = {
        '110_unique_games': len(rows) == len({r['game_id'] for r in rows}) == 110,
        'no_game_errors': not bad,
        '10_games_with_actions': sum(r['actions'] > 0 for r in rows) >= 10,
        'reported_generated_tokens': sum(r['tokens'] for r in rows) > 0,
    }
    (working / 'probe-gates.json').write_text(json.dumps(gates, indent=2) + '\n')
    if not all(gates.values()):
        raise RuntimeError('Production probe failed: ' + json.dumps(gates))
    print('PROBE_OPERATIONAL_PASS', flush=True)
