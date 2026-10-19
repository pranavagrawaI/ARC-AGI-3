"""Fault injection proves that the probe retains errors instead of accepting COMPLETE."""
import asyncio
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from runtime import probe_run

HERE = Path(__file__).resolve().parent


async def verify():
    checks = {}
    for mode in ('exception', 'swallowed-game-error'):
        working = HERE / ('observer-fixture-' + mode)
        working.mkdir(exist_ok=True)
        server = SimpleNamespace(stopped=False)
        server.stop = lambda: setattr(server, 'stopped', True)
        row = SimpleNamespace(game_id='fixture', state='crashed', history=[],
                              levels_completed=0, final_generated_tokens=0,
                              solver_note='error: injected failure')
        async def run(**kwargs):
            assert kwargs['minimal_diagnostics'] is True
            if mode == 'exception':
                raise RuntimeError('injected benchmark failure')
        benchmark = SimpleNamespace(game_runs=[row], run=run)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                await probe_run(benchmark, None, working, None, server)
        except RuntimeError as exc:
            checks[mode + '_rejected'] = ('injected benchmark failure' in str(exc)
                                           if mode == 'exception' else 'Production probe failed' in str(exc))
        else:
            checks[mode + '_rejected'] = False
        final = json.loads((working / 'probe-final.json').read_text())
        checks[mode + '_partial_evidence_retained'] = final['games'][0]['solver_note'] == 'error: injected failure'
        checks[mode + '_gateway_stopped'] = server.stopped
    checks['traceback_retained'] = 'injected benchmark failure' in (HERE / 'observer-fixture-exception/probe-exception.txt').read_text()
    assert all(checks.values()), checks
    (HERE / 'CPU-OBSERVER.json').write_text(json.dumps(checks, indent=2) + '\n')
    print(json.dumps(checks))


if __name__ == '__main__':
    asyncio.run(verify())
