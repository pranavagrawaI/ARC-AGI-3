"""Two recorded turns through the real current Actor; no game implementation."""
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path

import bootstrap
from bootstrap import HERE, RUNTIME
from actor import MdArcSolverAgent, OperationalActor
from client import ActorClient
from theory import Store, IntegrityError, digest, replay, write

COMMON_TASK = (
    'RECORDED CONSTRUCTION TEST. Use the existing full Actor tool surface to inspect '
    'the supplied historical evidence and prepare ONE prospective action with '
    'self.submit_actions. No game executes. After the first proposal the host will '
    'reveal one fixed historical continuation, regardless of your proposed action. '
    'That continuation is NOT the result of your proposal. Update your operational '
    'knowledge and prepare one more decision. This tests construction completion, '
    'not gameplay utility or truth. All model calls share a six-request arm cap; '
    'at most 4096 completion tokens per request, 24576 total. No extra consolidator.'
)


def evidence_rows():
    return [json.loads((HERE / 'evidence' / name).read_text())
            for name in ('prefix.jsonl', 'continuation.jsonl')]


def catalog(rows):
    return {f'transition:{r["step"]}': digest(r) for r in rows}


def state(observation, step, turn):
    return dict(turn=turn, step=step, level=observation['levels_completed'],
                state=observation['state'], levels_completed=observation['levels_completed'],
                available_actions=observation['available_actions'],
                grid_rows=[''.join('0123456789abcdef'[c] for c in row) for row in observation['grid']],
                action_results=[])


def install_evidence(run, rows):
    ipc = Path(run) / 'ipc'
    ipc.mkdir(parents=True, exist_ok=True)
    states = [state(rows[0]['before'], 0, 0)]
    states += [state(r['after'], r['step'], r['step']) for r in rows]
    (ipc / 'states.jsonl').write_text(''.join(json.dumps(s) + '\n' for s in states))
    (ipc / 'transitions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return states[-1]


def protected_snapshot(run):
    return {name: hashlib.sha256((Path(run) / 'ipc' / name).read_bytes()).hexdigest()
            for name in ('states.jsonl', 'transitions.jsonl')}


async def episode(arm, output, transport, *, wall_seconds=240):
    output = Path(output)
    run = output / 'actor'
    run.mkdir(parents=True, exist_ok=True)
    rows = evidence_rows()
    latest = install_evidence(run, rows[:1])
    deadline = time.monotonic() + wall_seconds
    # Same environment in both isolated workers. No .env or implicit credentials.
    os.environ.update(ARC_DISABLE_DOTENV='1', ARC_DISCOVERY_ARM='a', ARC_CONTINGENT_ARM='off',
                      ARC_STALL_REVIEW='0')
    store = Store(output / 'canonical', catalog(rows[:1])) if arm == 'treatment' else None
    checks = protected_snapshot(run)

    def checked_transport(body):
        if protected_snapshot(run) != checks or (run / 'ipc/ABORT').exists():
            raise IntegrityError('immutable Actor evidence or cell containment failed')
        if store:
            store._verify()
        return transport(body)

    kwargs = dict(llm=ActorClient(checked_transport), run_dir=run, game_id='the game', alias='the game',
                  skill_path=RUNTIME / 'chassis/skills/interactive-game-solver/SKILL.md',
                  enable_summarizer=False, max_actions_per_turn=20)
    agent = OperationalActor(**kwargs, operational_store=store) if store else MdArcSolverAgent(**kwargs)
    write(output / 'SURFACE.json', {'class': type(agent).__name__, 'common_task': COMMON_TASK,
                                  'public_methods': sorted(n for n in dir(agent) if not n.startswith('_') and callable(getattr(agent, n)))})
    proposals = []
    started = time.monotonic()
    try:
        for turn in range(2):
            guidance = COMMON_TASK if turn == 0 else (
                'NEW RECORDED OBSERVATION transition:2. Historical action DOWN; this '
                'is a recorded continuation, not the outcome of your proposed action. '
                'Reconcile it in your existing knowledge and prepare the next decision.'
            )
            await asyncio.wait_for(agent.handle({'game_states': [json.dumps(latest)], 'user_messages': [guidance]}),
                                   timeout=max(.001, deadline - time.monotonic()))
            if protected_snapshot(run) != checks or (run / 'ipc/ABORT').exists():
                raise IntegrityError('evidence/cell integrity failed after handle')
            path = run / 'ipc/actions.jsonl'
            actions = [json.loads(s) for s in path.read_text().splitlines()]
            if len(actions) != turn + 1:
                break
            proposals.append(actions[-1])
            if turn == 0:
                latest = install_evidence(run, rows)
                checks = protected_snapshot(run)
                if store:
                    store.evidence = catalog(rows)
        if store:
            replayed = replay(output / 'canonical', catalog(rows), store.artifacts)
            write(output / 'REPLAY.json', replayed)
            completed = output / 'canonical/completed.jsonl'
            cycles = [json.loads(s) for s in completed.read_text().splitlines()] if completed.exists() else []
            revisions = [r for row in store.rows for r in row['theory']['revisions'] if r['reason'] == 'counterevidence']
            decisions = replayed['decisions']
            metrics = dict(construction_completion=bool(cycles and len(proposals) == 2),
                           evidence_entered=sorted({e['id'] for row in store.rows for e in row['theory']['evidence_refs']}),
                           contradiction_driven_revisions=len(revisions),
                           active_uncertainty_or_procedure=[d['policy']['selection'] for d in decisions],
                           declared_modes=[d['policy']['mode'] for d in decisions],
                           prospective_decision_completeness=len(decisions),
                           canonical_action_binding=all(d['actions'] == d['policy']['actions'] for d in decisions) if decisions else False,
                           post_observation_same_lineage=bool(cycles), replayability=bool(decisions),
                           semantic_causation='not judged; exact reference/action binding only')
        else:
            metrics = dict(construction_completion=len(proposals) == 2,
                           evidence_entered=None, contradiction_driven_revisions=None,
                           active_uncertainty_or_procedure=None, declared_modes=None,
                           prospective_decision_completeness=None, canonical_action_binding=None,
                           post_observation_same_lineage=None, replayability=None,
                           interpretation='canonical metrics not applicable to unchanged control; inspect native artifacts')
        result = dict(arm=arm, metrics=metrics, proposals=proposals, elapsed_seconds=time.monotonic() - started,
                      gameplay='not run', truth='not evaluated', provider_transport='shared')
        write(output / 'RESULT.json', result)
        return result
    finally:
        # A process owns each arm; cancel tail tasks before returning from local fixtures.
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
