"""Prepare immutable v3 cards after volatile-seed eviction in v2."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'analysis/local-probes/20260911-system-pivot-test-v3'
FIXTURE=ROOT/'analysis/local-probes/20260911-system-pivot-test-v2/fixture-v2'
OWN=ROOT/'experiments/probes/epistemic-critic-local'

CARD='''# System pivot comparison v3: {pair}
Question: Does frequent reasoning-off review reduce time/actions to warranted, retained correction, regardless of insight origin?
Posture: Controlled restart comparison after v1 oversized-seed and v2 volatile-seed invalidations.
Contrast: Identical seven-action LS20 successor and compact cacheable notes/measurements; fresh empty Python workers; frequent critic versus baseline. Order={order}; pair B reverses it. Evaluator diagnosis is absent from model context.
Prediction: Review reduces elapsed time to a correction confirmed in three later executed decisions without adding false pivots.
Measure: Source-bound grading of observable Actor outputs and executed decisions. Record first warranted correction, actions/time to it, three-decision retention, regressions, false pivots, critic time, total time, and cost. Insight origin is irrelevant; acknowledgement earns nothing; missing follow-up is unresolved. Every Actor request must retain all three evidence markers, checked before spend reservation.
Cap: Per arm600s plus95s retirement,64 actions,96 Actor calls,20 critic calls,$0.50 admission ceiling. Four arms <=$2. Actor4096 and critic384 completion tokens, reasoning off, Qwen3.8-27B/CoreWeave FP8 only, zero retries/fallback. Reviews after2 Actor calls or2 actions. Continue fixed window after correction to observe relapse.
Stop: Terminal/cap or operational failure. Abort pair on transport/unresolved accounting; do not start another pair after unresolved failure. No relaunch of started card.
Invalid if: Replay differs; required evidence absent before any Actor reservation; source/input drift; diagnosis leaks; wrong provider/reasoning mode; unaccounted request; unresolved retirement. Missing pair arm prevents paired benefit. Output failure is separate from semantic quality.
Result map: Faster retained correction without additional harmful pivots -> expand across new cases; same -> no benefit demonstrated; worse -> no promotion; censored/invalid -> unresolved. Compare whole-system time including review; separate-role value versus equal-compute self-review remains untested.
Outputs: This directory plus the immutable v2 compact fixture contain inputs, source hashes, exact requests/responses, actions/grids, snapshots, timestamps, grades and scorer output. No Kaggle launch.
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
    OUT.mkdir(exist_ok=False)
    sources=[OWN/n for n in ['live_system_pivot_v3.py','live_hook_v2.py','paired_commitment.py',
      'reasoning_screen.py','claim_screen.py','framework-v1/system.md','system-pivot-v1/fixture.py',
      'system-pivot-v1/score.py','system-pivot-v1/generate_v3.py']]
    for pair,order in [('pair-a','critic-first'),('pair-b','baseline-first')]:
        path=OUT/pair;path.mkdir()
        (path/'EXPERIMENT-CARD.md').write_text(CARD.format(pair=pair,order=order))
        (path/'GRADES-TEMPLATE.json').write_text(GRADES)
        inputs=sources+[path/'EXPERIMENT-CARD.md',FIXTURE/'seed.txt',FIXTURE/'evidence.json',FIXTURE/'MANIFEST.json']
        (path/'CONFIG.json').write_text(json.dumps({'order':order,'fixture_sha256':digest(FIXTURE/'evidence.json'),
          'source_sha256':{str(p.resolve()):digest(p) for p in inputs}},indent=2)+'\n')
    (OUT/'BUILD-RESULT.json').write_text(json.dumps({'status':'prepared_only','paid_calls':0,
      'seed_bytes':len((FIXTURE/'seed.txt').read_bytes()),'v2_invalidator':'volatile seed evicted at baseline call60'},indent=2)+'\n')

if __name__=='__main__':main()
