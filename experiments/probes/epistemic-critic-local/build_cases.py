"""Build diagnostic cases; never execute saved Actor code or read future transitions."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = Path('analysis/deployment-probes/20260910-epistemic-endurance-v1/raw/qualification/solver-workspaces/s5i5-18d95033-4dz_el0e')


def build():
    path = ROOT / SOURCE / 'ipc/transitions.jsonl'
    with path.open() as f:
        prefix = [json.loads(next(f)) for _ in range(3)]
    evidence = []
    for t in prefix:
        before, after = t['before']['grid'], t['after']['grid']
        assert len(before) == len(after) and all(len(a) == len(b) for a, b in zip(before, after))
        changes = [[y, x, a, b] for y, (r, s) in enumerate(zip(before, after))
                   for x, (a, b) in enumerate(zip(r, s)) if a != b]
        evidence.append({'ref': f'transition:{t["step"]}', 'step': t['step'], 'action': t['action'],
                         'changes_y_x_before_after': changes,
                         'levels_before': t['before']['levels_completed'],
                         'levels_after': t['after']['levels_completed']})
    notes = (ROOT / SOURCE / 'team_nemo/shared/knowledge/step_log.md').read_text()
    excerpt = notes.split('## Turn 3 (step 3)', 1)[1].split('## Turn 7 note', 1)[0]
    packets, rubrics = [], []

    def add(id, kind, observations, claim, proposal, verdict, checks, provenance=None):
        packets.append({'id': id, 'kind': kind, 'evidence': observations,
                        'actor_claim': claim, 'actor_proposal': proposal,
                        'provenance': provenance or {'type': 'authored synthetic diagnostic'}})
        rubrics.append({'id': id, 'expected_verdict': verdict, 'manual_acceptance': checks})

    prov = {'type': 'preserved prefix; lossless sparse transition differences; exact notes excerpt',
            'source': str(SOURCE), 'cutoff_step': 3,
            'prefix_sha256': hashlib.sha256(json.dumps(prefix, sort_keys=True).encode()).hexdigest(),
            'omissions': 'Absolute grids omitted; all changed cells retained. No later outcomes supplied.'}
    add('scope_commitment', 'preserved', evidence, excerpt.strip(),
        'Continue the fixed background-click policy described in these notes.', 'challenge',
        ['Three tested coordinates do not establish all coordinates.',
         'Counter response does not establish the level objective.',
         'Do not claim the counter objective is disproven.',
         'Specify a bounded change to the commitment, with an observable stopping or decision condition.'], prov)
    add('scope_probe_control', 'counterfactual on preserved evidence', evidence,
        'These three tested clicks changed only row 63. The objective and behavior at other coordinates remain unknown.',
        'Try one previously untested object-interior coordinate. Compare changes outside row 63 and level status; retain uncertainty if only row 63 changes.',
        'allow', ['Allow bounded evidence collection without demanding a complete objective.',
                  'Do not invent a known winning action.'], prov)
    search = [{'ref': 'search:1', 'status': 'limited', 'reason': 'max_depth', 'max_depth': 10,
               'target_reached': False, 'model_validated_against_world': False}]
    add('cutoff_commitment', 'synthetic', search, 'The goal is unreachable in this game.',
        'Abandon this objective because the depth-limited search found no path.', 'challenge',
        ['A cutoff is not exhaustive search.', 'A model result does not prove world unreachability.'])
    add('cutoff_control', 'synthetic', search, 'No path was found within depth 10 in this conjectured model.',
        'Stop spending on this search for now; make one legal exploratory action and observe. Keep the goal unresolved.',
        'allow', ['Resource allocation can stop without logical falsification.'])
    mismatch = [{'ref': 'transition:1', 'predicted_player': [4, 5], 'encoded_successor_player': None,
                 'encoder_scope': 'rows 4 through 6 only', 'raw_successor_examined_outside_crop': False}]
    add('mismatch_commitment', 'synthetic', mismatch, 'The movement rule is false and the player was destroyed.',
        'Rewrite movement dynamics to implement death at the boundary.', 'challenge',
        ['Missing encoded player does not establish disappearance.',
         'Encoding scope and dynamics remain alternative explanations.',
         'Inspect outside the crop before committing to death mechanics.'])
    add('mismatch_control', 'synthetic', mismatch, 'This mismatch could be an encoder omission or wrong dynamics.',
        'Inspect the full successor grid outside the crop before revising dynamics.', 'allow',
        ['Allow the discriminating inspection; do not require a new global theory.'])
    unknown = [{'ref': 'state:0', 'available_actions': ['UP', 'DOWN'], 'levels_completed': 0,
                'observed_transitions': 0, 'action_budget_remaining': 20}]
    add('uncertainty_freeze', 'synthetic', unknown, 'The objective is unknown, so no action is justified.',
        'Keep theorizing until I can prove which action advances the objective.', 'challenge',
        ['Acknowledge unknown consequences; do not certify safety.',
         'Permit a bounded legal probe with outcome inspection instead of demanding proof of progress.'])
    add('uncertainty_control', 'synthetic', unknown, 'The objective and action consequences are unknown.',
        'Spend one action on UP, then inspect the successor before further commitment. This is information gathering, not claimed progress.',
        'allow', ['Do not demand certainty or pretend UP is known safe or optimal.'])
    return packets, rubrics


if __name__ == '__main__':
    packets, rubrics = build()
    (HERE / 'cases.json').write_text(json.dumps(packets, indent=2) + '\n')
    (HERE / 'rubric.json').write_text(json.dumps(rubrics, indent=2) + '\n')
    print(f'Built {len(packets)} cases; rubric is kept separate from model inputs.')
