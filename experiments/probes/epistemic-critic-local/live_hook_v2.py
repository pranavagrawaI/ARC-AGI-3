"""Persistent-prefix submit pause for the system-pivot comparison."""
import asyncio
import functools
import json
from pathlib import Path
import sys
import urllib.request


SKILL = '''HOST EPISTEMIC SELF-REVIEW: No action from this submission was committed.
Before resubmitting, inspect the inference behind your proposal. Accept observed
facts; do not spend this review recounting cells or solving the game again.
Does the conclusion exceed the tested cases? Does compatible evidence actually
distinguish alternatives? Are you uncertain only about parameters of an assumed
objective? Could your commitment survive every outcome by changing the story?
An action can be sensible while its explanation is unproven. Missing evidence is
not a negative test unless the evidence would be expected under that hypothesis.
Parsimony breaks ties among explanations of the evidence; it does not certify one.
Do not demand all possible evidence or invent an objection to bounded exploration.
In at most 60 words, identify one consequential inference gap, or say none found.
State what remains uncertain and what would change your commitment. Then
resubmit an action, explicitly retaining or revising the proposal and saying why.
You may retain it; neither criticism nor self-review proves a better action.'''


def install(cls, seed, review, log, arm):
    from nooa.context_blocks import Context
    init, submit = cls.__init__, cls.submit_actions

    @functools.wraps(init)
    def seeded(self, *args, **kwargs):
        init(self, *args, **kwargs)
        self.context['preserved_prefix'] = Context(seed, prefix=True)
        self._pilot_review_used = False

    @functools.wraps(submit)
    def paused(self, actions, rationale, predicted_path=None, *, search_id=None):
        proposal = dict(actions=actions, rationale=rationale,
                        predicted_path=predicted_path, search_id=search_id)
        log({'event': 'proposal', 'proposal': proposal})
        if arm != 'baseline' and not self._pilot_review_used:
            self._pilot_review_used = True
            feedback = review(proposal) if arm == 'critic' else SKILL
            log({'event': 'review_returned', 'feedback': feedback})
            return feedback
        result = submit(self, actions, rationale, predicted_path, search_id=search_id)
        log({'event': 'submission_result', 'result': result, 'proposal': proposal})
        return result

    class ReviewedAgent(cls):
        __init__ = seeded
        submit_actions = paused
    return ReviewedAgent


def main():
    runtime, arm, seed_path, relay = sys.argv[1:5]
    sys.path[:0] = [runtime, str(Path(runtime)/'chassis')]
    import headless
    def log(row):
        with Path('pilot_events.jsonl').open('a') as f:
            f.write(json.dumps(row)+'\n')
    def review(proposal):
        req = urllib.request.Request(relay+'/review', data=json.dumps(proposal).encode(),
                                     headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req, timeout=55) as response:
            critique = json.load(response)
        return ('HOST EPISTEMIC REVIEW: No action from this submission was committed. '
                'This is advice, not a verified game model. In at most 60 words, '
                'explain whether you incorporate or contest the finding and why; '
                'then resubmit your retained or revised action.\n'+json.dumps(critique))
    headless.MdArcSolverAgent = install(headless.MdArcSolverAgent, Path(seed_path).read_text(), review, log, arm)
    sys.argv = [str(Path(runtime)/'headless.py'), '--run-dir', '.']
    asyncio.run(headless.run())


if __name__ == '__main__':
    main()
