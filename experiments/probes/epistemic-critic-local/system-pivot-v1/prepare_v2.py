"""Prepare immutable v2 pair cards after the oversized-seed invalidation."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'analysis/local-probes/20260911-system-pivot-test-v2'
FIXTURE=OUT/'fixture-v2'
OWN=ROOT/'experiments/probes/epistemic-critic-local'

CARD='''# System pivot comparison v2: {pair}
Question: Does frequent reasoning-off review reduce time/actions to warranted, retained correction, regardless of insight origin?
Posture: Controlled restart comparison after v1 seed-delivery invalidation.
Contrast: Identical seven-action LS20 successor and compact retained notes/measurements; fresh empty Python workers; baseline versus frequent critic. Order={order}. Pair B reverses order. Evaluator diagnosis is absent from both contexts.
Prediction: Review reduces elapsed time to a correction confirmed in three later executed decisions, without adding false pivots.
Measure: Source-bound grading of observable Actor outputs and executed decisions. Record first warranted correction, actions/time to it, three-decision retention, regressions, false pivots, critic time, total time, and cost. Insight origin is irrelevant; acknowledgement alone earns nothing; missing follow-up is unresolved. Verify required evidence markers in every paid Actor request.
Cap: Per arm 600s plus95s retirement,64 actions,96 Actor calls,20 critic calls,$0.50 admission ceiling. Four arms <=$2. Actor4096 and critic384 completion tokens, reasoning off, Qwen3.8-27B/CoreWeave FP8 only, zero retries/fallback. Reviews after2 Actor calls or2 actions. Fixed window continues after correction to observe relapse.
Stop: Terminal/cap or operational failure. Abort the pair on failed transport/unresolved accounting; do not start another pair after unresolved failure. No relaunch of a started card.
Invalid if: Replay differs; any required seed marker is absent from a paid Actor request; source/input drift; diagnosis leaks; wrong provider/reasoning mode; unaccounted request; unresolved retirement. Missing pair arm prevents paired benefit. Output failure remains separate from semantic quality.
Result map: Faster retained correction without added harmful pivots -> expand across new cases; same -> no benefit demonstrated; worse -> no promotion; censored/invalid -> unresolved. Compare whole-system time including review. Separate-role value versus equal-compute self-review remains untested.
Outputs: This directory and fixture-v2 contain exact inputs, source hashes, requests/responses, actual actions/grids, notes snapshots, timestamps, grades and scorer output. No Kaggle launch.
'''

GRADES='''{
  "review_complete": false,
  "arms": {"baseline": [], "critic": []},
  "annotation_example": {"kind":"correction", "actor_call":0, "evidence_quote":"exact Actor output excerpt", "warranted":true, "rationale":"why it follows from available evidence"},
  "retention_note":"Use retained plus decision_id for each of three later decisions. An unchanged action can count when its active rationale preserves the correction. Repetition or incorporate language alone does not count."
}
'''


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    seed=FIXTURE/'seed.txt'
    if len(seed.read_bytes())>10_000:raise RuntimeError('seed too large')
    sources=[OWN/n for n in ['live_system_pivot.py','live_system_pivot_v2.py','live_hook.py',
      'paired_commitment.py','reasoning_screen.py','claim_screen.py','framework-v1/system.md',
      'system-pivot-v1/fixture.py','system-pivot-v1/score.py']]
    for pair,order in [('pair-a','baseline-first'),('pair-b','critic-first')]:
        path=OUT/pair
        path.mkdir(parents=True,exist_ok=False)
        (path/'EXPERIMENT-CARD.md').write_text(CARD.format(pair=pair,order=order))
        (path/'GRADES-TEMPLATE.json').write_text(GRADES)
        all_sources=sources+[path/'EXPERIMENT-CARD.md',FIXTURE/'seed.txt',FIXTURE/'evidence.json',FIXTURE/'MANIFEST.json']
        config={'order':order,'fixture_sha256':digest(FIXTURE/'evidence.json'),
                'source_sha256':{str(p.resolve()):digest(p) for p in all_sources}}
        (path/'CONFIG.json').write_text(json.dumps(config,indent=2)+'\n')
    (OUT/'BUILD-RESULT.json').write_text(json.dumps({'status':'prepared_only','paid_calls':0,
      'seed_bytes':len(seed.read_bytes()),'v1_invalidator':'required seed absent from serialized Actor requests'},indent=2)+'\n')


if __name__=='__main__':main()
