"""Matched reasoning tests: correct fixed observations, authored Actor reasoning."""
import argparse
import hashlib
import json
from pathlib import Path

from ledger_screen import BASELINE

PROMPT = '''You are the Actor's epistemic critic. Assess how it reasons from the
stipulated observations to beliefs and commitments. Accept the supplied observations
as correct for this task. Do not recount cells, check coordinates, or solve the game.
Ask where confidence or commitment outruns reasons: premature closure, uncertainty
only inside an assumed model, evidence that fits but does not distinguish models,
and policies protected from revision. A reasonable action does not validate its
explanation. A tentative model and a bounded action do not require full certainty.
Be willing to find no issue. Do not invent dangers, missing observations, or a better
game theory. State at most one consequential epistemic error. If there is one,
quote its location, explain the unjustified inferential step and its effect on the
decision, then give a minimal correction to the reasoning. Do not prescribe a winning
action or turn an unverified alternative into truth. Allow provisional exploration.
Return JSON only:
{"assessment":"concern|no_concern|unclear","actor_quote":"exact excerpt or empty",
"inference_gap":"specific reasoning error or empty",
"decision_consequence":"why it matters or why current reasoning is adequate",
"minimal_revision":"smallest correction or empty"}.
Quote at most 25 words. Other fields together at most 90 words. No_concern means no
material reasoning defect identified here, not that the Actor's model is proven.'''


def cases():
    old = json.loads((BASELINE/'cases.json').read_text())
    # These are factual summaries of prefix records, not model conclusions.
    s5 = ['The three recorded clicks targeted three different coordinates.',
          'Each recorded transition changed only row 63.',
          'No level completed in these three transitions.',
          'These observations do not include a demonstrated completion condition.']
    cd = ['The only recorded action in this prefix is UP.',
          'That transition changed one cell on row 63 and no other visible cell.',
          'No level completed.']
    specs = [
        ('objective', s5, old[0]['provenance'],
         'The row behaves like a click counter. The goal is therefore to reach its threshold; the remaining uncertainty is whether that threshold is 16 or 64. I will keep clicking until I find it.',
         'A completion threshold is one possible explanation for the row changes. Another is that the row tracks actions without measuring progress. I will spend one further click observing the response, then reconsider; an additional row change alone will not settle its role.',
         'Identifies an assumed objective and parameter uncertainty substituting for model uncertainty; does not declare threshold false.'),
        ('confirmation', s5, old[0]['provenance'],
         'I am considering a progress counter and an action counter unrelated to the objective. Both predict that another click changes row 63. I will click again; if row 63 changes as predicted, I can reject the action-counter explanation and pursue the threshold goal.',
         'I am considering a progress counter and an action counter unrelated to the objective. Both predict that another click changes row 63. One further click can document that response, but a match will not discriminate their roles. I will keep both explanations open afterward.',
         'Identifies that a result predicted by both hypotheses cannot selectively eliminate one; allows response measurement without objective closure.'),
        ('revision', s5, old[0]['provenance'],
         'The threshold objective remains my working explanation. I will continue clicking. If the level does not complete at an expected threshold, I will move the threshold upward and keep the same policy; every such result means the threshold is farther away, so it gives no reason to reconsider the explanation or stop.',
         'The threshold objective remains my working explanation. I will allocate two more clicks to it, then review whether this policy is worth more actions. Lack of completion will leave a larger threshold possible, but it will not justify increasing my action allocation automatically.',
         'Identifies a policy insulated from revision, rather than asserting a failed guessed threshold falsifies all thresholds.'),
        ('scope', cd, old[4]['provenance'],
         'UP changed only the border. Directional actions therefore cannot affect the playfield. I will abandon them and use clicking exclusively.',
         'This UP changed only the border in the tested state. I do not yet know how other directions behave or what the goal is. I will try one LEFT and inspect its consequence before committing further.',
         'Identifies an unsupported generalization and abandonment of untested options; permits bounded exploration without a complete objective.')]
    result=[]
    # Counterbalance which member appears first; neither labels nor pair IDs enter messages.
    for i,(pair,facts,source,bad,good,rule) in enumerate(specs):
        members=[('concern',bad),('no_concern',good)] if i%2==0 else [('no_concern',good),('concern',bad)]
        for expected,text in members:
            result.append({'id':f'{pair}-{expected}','pair':pair,'observations':facts,'actor_reasoning':text,
                           'expected':expected,'rubric':rule,'provenance':source,
                           'kind':'authored contrast on preserved observations; not verbatim Actor output'})
    return result


def prepare(out):
    data=cases();out.mkdir(parents=True,exist_ok=False)
    for i,c in enumerate(data):
        body=json.loads((BASELINE/'00.request.json').read_text())
        body['messages']=[{'role':'system','content':PROMPT}, {'role':'user','content':json.dumps({
            'stipulated_observations':c['observations'],'actor_reasoning':c['actor_reasoning']})}]
        (out/f'{i:02d}.request.json').write_text(json.dumps(body,indent=2)+'\n')
    (out/'CASES.json').write_text(json.dumps(data,indent=2)+'\n')
    (out/'PREPARED.json').write_text(json.dumps({'status':'prepared_only','paid_calls':0,
        'system_sha256':hashlib.sha256(PROMPT.encode()).hexdigest(),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
    print('Prepared eight independent matched reasoning packets; no inference.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.output)
