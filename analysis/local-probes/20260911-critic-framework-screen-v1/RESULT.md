# Framework critic diagnostic: no demonstrated improvement in substantive judgment

Completed all 12 calls on the existing Qwen3.8-27B OpenRouter endpoint, CoreWeave FP8,
reasoning off. Actual cost: **$0.0406726**. Frozen inputs still match their hashes.
Six identical evidence packets were reviewed once with each prompt. No gameplay or
Actor continuation occurred. The captured R8 context is unchanged from the earlier
review request; the remaining five cases are authored controls/contrasts. All are
development cases, not a held-out generalization estimate.

## Manual grades against the frozen criteria

| Case | Old critic | Framework critic |
|---|---|---|
| Captured R8, inconsistent reachability evidence | Output failure: exceeds word cap. Challenges collision assumptions instead of the available inconsistency; quotation uses ellipsis, and critique adds a claimed historical movement as justification. Does not demonstrate grounded detection of the target gap. | Valid empty no_concern. Avoids certifying four positions but does not identify the inconsistency. Under the frozen rubric, active Actor checking makes no_concern permissible; this is restraint, not demonstrated detection. |
| Actor already addressing inconsistent computations | Valid no_concern; appropriate restraint. | Valid no_concern; appropriate restraint. |
| Counter change promoted to threshold objective | Valid concern; detects unsupported objective. Revision asks for verified completion before committing, which risks requiring too much certainty unless commitment means the unbounded policy. Mixed correction. | Valid concern; detects unsupported objective, but invents a 64+ click commitment and proposes checking after the threshold is met. This retains the disputed threshold framing and may encourage the same costly policy. Mixed, unsuitable revision. |
| Provisional objective, bounded probe | Valid no_concern; appropriate restraint. | Valid no_concern; appropriate restraint. |
| Scoped negative observation, one LEFT probe | Correct no_concern semantically, but quote exceeds 25 words: output failure. | Valid no_concern; appropriate restraint. |
| One UP result excludes all directions | Valid concern; correctly identifies excessive scope and preserves other contexts as open. | Correct scope concern in substance, but exceeds 90 non-quote words: output failure. Correction emphasizes another UP test rather than directly preserving all untested directions. |

The raw outputs remain available even when they fail delivery validation. A useful
point inside a rejected response is not a successfully delivered intervention.
The framework prompt produced five valid responses out of six; the old prompt four
out of six. This small difference does not establish reliability. Both returned the
expected assessments on all five authored cases before interface validation; the
quality of their corrections is more discriminating than the labels.

## What changed and what did not

The new no_concern format removes unsupported approving prose by construction.
Its absence cannot count as improved detection. The substantive issue remains:
the critic can name an epistemic fault while offering a revision that still lives
inside the Actor's disputed premise. In the objective case it asks what happens
after the supposed threshold is met, although the threshold objective itself was
never established. It also describes 64+ clicks, which the Actor did not explicitly
commit to. These are failures of the proposed alternative's own evidential discipline.

The new critic took 7.99 seconds on the full captured context, compared with 4.60
seconds for the old critic. Short authored cases took 1.14–3.26 seconds with the new
prompt and 1.72–2.65 seconds with the old. These single samples differ in prompt
length, output length, and cache history; they do not isolate a prompt latency effect.

## Decision

Do not promote this candidate on these results. Framework-oriented instructions
have not yet demonstrated the desired independence. Preserve the narrow output
discipline, but examine whether the minimal revision actually withdraws the
unsupported commitment without requiring its premise to be true. The next candidate
would need to pass that test on new cases as well as this known example. Another
live endurance is not justified by improved labels alone.

Operational evidence: EXECUTION.json records 12 attempts and usage for 12 responses;
all finished normally on the required provider with zero reported reasoning tokens.
Three semantic output failures were preserved without retries. Total reservation
was $0.20895936 under the $0.25 cap. FROZEN.json binds the card, case criteria, exact
requests, order, candidate prompt and runner. Source imports were not separately
hashed, a reproduction limitation; the actual request prompts and evidence were.
