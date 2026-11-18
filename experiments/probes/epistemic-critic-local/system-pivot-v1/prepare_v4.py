"""Prepare one immutable paired run for the action-tempo contract."""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / 'analysis/local-probes/20260911-system-pivot-test-v4'
SOURCE_FIXTURE = ROOT / 'analysis/local-probes/20260911-system-pivot-test-v2/fixture-v2'
FIXTURE = OUT / 'fixture-v4'
OWN = ROOT / 'experiments/probes/epistemic-critic-local'
sys.path.insert(0, str(OWN))

CARD = '''# System pivot comparison v4: action as epistemic computation
Question: With the same preserved LS20 evidence, does an Actor tempo contract produce a reversible action within two Actor responses, and does frequent tempo-aware criticism improve subsequent pivot speed without causing false pivots?
Posture: Matched live diagnostic after v3 showed 15 Actor calls and seven reviews before its first action.
Contrast: Both sequential arms receive identical persistent evidence and Actor tempo rules. Baseline receives no critic; critic receives reasoning-off review after two Actor calls or two actions. Both start from the identical seven-action LS20 successor with fresh Python workers. Baseline runs first.
Prediction: Both arms execute a reversible probe by Actor response two unless they name a concrete blocker. The critic arm has no greater time to first action and reaches retained correction sooner without additional false pivots.
Measure: Exact requests/responses and host actions. Record Actor calls and wall seconds to first action; every pre-action call that creates no environmental evidence; review verdict/delivery; first warranted correction; three-decision retention; false pivots; levels; time and cost. The two-response threshold is prompt uptake, not level-solving efficacy.
Cap: Per arm 600s plus 95s retirement, 64 actions, 96 Actor calls, 20 critic calls, $0.50 admission ceiling. Pair <=$1. Actor 4096 and critic 256 completion tokens, reasoning off, Qwen3.8-27B/CoreWeave FP8 only, zero retries or fallbacks.
Stop: Finish both arms at terminal or cap. Abort before the second arm on transport failure, unresolved accounting, evidence loss, source drift, or process-retirement failure. Never relaunch this started card.
Invalid if: Initial grids differ; required evidence or tempo markers are absent before an Actor reservation; diagnosis leaks; source/input drift; wrong provider or reasoning mode; unaccounted request; unresolved transport or retirement. A missing arm prevents paired interpretation. Invalid critic output is a delivery failure, not no_concern.
Result map: Probe by response two in both arms -> tempo contract shows uptake; only critic does so -> frequent review has a candidate tempo benefit; neither does so -> prompt constraint is insufficient and host enforcement becomes the next candidate; a harmful or false critic pivot -> do not promote critic. Retained faster correction without false pivots -> expand to fresh games; otherwise preserve the diagnostic result.
Outputs: This directory contains the immutable fixture, card, source hashes, exact requests/responses, full host action grids, review records, snapshots, timings, accounting, grades and audit. No Kaggle launch or deployed-runtime mutation.
'''


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(exist_ok=False)
    FIXTURE.mkdir()
    shutil.copyfile(SOURCE_FIXTURE / 'evidence.json', FIXTURE / 'evidence.json')
    from tempo_contract import ACTOR_TEMPO_CONTRACT
    seed = (SOURCE_FIXTURE / 'seed.txt').read_text().rstrip() + ACTOR_TEMPO_CONTRACT + '\n'
    (FIXTURE / 'seed.txt').write_text(seed)
    manifest = {
        'files': {name: digest(FIXTURE / name) for name in ('evidence.json', 'seed.txt')},
        'source_files': {
            str(SOURCE_FIXTURE / 'evidence.json'): digest(SOURCE_FIXTURE / 'evidence.json')
        },
        'note': 'V2 compact evidence plus the persistent Actor tempo contract. Full transitions remain in evidence.json.',
    }
    (FIXTURE / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (OUT / 'EXPERIMENT-CARD.md').write_text(CARD)
    sources = [OWN / name for name in [
        'live_system_pivot_v4.py', 'live_hook_v2.py', 'paired_commitment.py',
        'tempo_contract.py', 'reasoning_screen.py', 'claim_screen.py',
        'framework-v1/system.md', 'system-pivot-v1/fixture.py',
        'system-pivot-v1/score.py', 'system-pivot-v1/prepare_v4.py',
    ]]
    inputs = sources + [OUT / 'EXPERIMENT-CARD.md', FIXTURE / 'seed.txt',
                        FIXTURE / 'evidence.json', FIXTURE / 'MANIFEST.json']
    config = {
        'order': 'baseline-first',
        'fixture_sha256': digest(FIXTURE / 'evidence.json'),
        'source_sha256': {str(path.resolve()): digest(path) for path in inputs},
    }
    (OUT / 'CONFIG.json').write_text(json.dumps(config, indent=2) + '\n')
    (OUT / 'BUILD-RESULT.json').write_text(json.dumps({
        'status': 'prepared_only',
        'paid_calls': 0,
        'seed_bytes': len((FIXTURE / 'seed.txt').read_bytes()),
        'comparison': 'same preserved state; Actor tempo contract in both arms; tempo critic in critic arm',
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
