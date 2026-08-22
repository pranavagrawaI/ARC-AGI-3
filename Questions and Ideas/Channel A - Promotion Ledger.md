# Channel A — Promotion Ledger

**Admission:** preserved evidence discriminates for the exact intervention over its
live alternative. **Active WIP: 2.** Follow-ons remain blocked until an active item
passes its local acceptance test.

## Passed gate 1 — Candidate survival without authority leakage

### Supported observation

In the selected trigger trajectory, transient model
`4f78ef3b371d3e8f4d106710d19d03f94de6c60cfa399776818714fa1c5a86df`
scored 8/16 exact transitions against the evidence available at its checkpoint. It
later scored 9/17 against final evidence. Deadline rollback restored a canonical
model scoring 0/17 and left no content-identical live path to the transient body.

This is one demonstrated selection inversion. It contradicts “successful Builder
return identifies the artifact worth retaining.” It does not show that timed-out
models are usually better or safe to publish.

### Intervention

Merge the former A1, A2, and A8 into one chain:

```text
model-changing Builder tool call
  -> immutable candidate body
  -> contemporaneous evidence frontier
  -> independent exact score and first divergence
  -> timeout restores canonical workspace
  -> retained candidate remains queryable by the next Builder
  -> canonical promotion still requires the strict verifier
```

Final-evidence scoring is an audit only. Live retention order may use only evidence
available at the candidate's frontier.

### Result — passed 2026-08-22

The preserved turn-18 replay passed every requirement:

1. the deadline occurs;
2. canonical `world_model.py` is restored byte-exactly;
3. the candidate and its 8/16 contemporaneous score survive on an explicit path;
4. a fresh Builder can list, read, or select it;
5. the planner receives no authority from retention alone.

The exact 8/16 candidate is now retained, independently readable by a fresh Builder,
noncanonical, and planner-ineligible after byte-exact rollback. See
`Three Gate Replays - 2026-08-22.md`.

## Passed gate 2 — Goal-to-discriminator closure replay

### Supported observation

Among the reviewed eligible corpus, the typed-goal treatment was exposed in three
runs. Two created one goal each; both ended at revision 1 and status `proposed`, with
zero `update_goal` attempts. The third created no goal. The selected typed trajectory
never converted its candidate into a near-term test, objective-focused Builder
request, canonical `outcome()`, or plan.

This establishes a frozen lifecycle, not model incapacity and not typed-goal
uselessness.

### Current implementation

The source already implements a stronger contract than the proposed bounded action
count. In the typed-goal treatment, a new proposal immediately blocks `take_action`
until the actor:

1. requests an objective-focused Builder review citing the candidate and public
   evidence;
2. receives a host-recorded review, including an explicit unavailable result on
   timeout;
3. calls `update_goal` to select, reject, or retain the candidate.

An attempted action before either lifecycle edge is rejected. The deterministic
positive control exercises proposal, both blocked actions, Builder ownership of
`world_model.py`, review, explicit selection, and the first permitted action. The
focused goal suite passes. The frozen corpus therefore describes the earlier run, not
the current implementation.

### Result — passed 2026-08-22

The selected post-proposal checkpoint passed under the current contract:

```text
cited candidate goal
  -> immediate objective review obligation
  -> update_goal resolution obligation
  -> ledger revision/status changes before action
  -> selected/rejected/retained result cites the discriminating evidence
```

The source goal advanced from revision 1 to revision 2 with an inconclusive independent
review and a named discriminator before the preserved action was permitted. Scripted
responses settle protocol correctness, not natural model uptake or gameplay benefit.

## Rejected follow-on — Allocate the Builder by job

The selected turn-18 request was explicitly dynamics-scoped, and all 28 calls were
mechanics work. Accuracy remained zero until call 27, where it reached 8/16. Mixed jobs
therefore did not cause this timeout, and a lower simple cap would have erased the only
positive candidate. Do not implement a job allocator from this evidence.

## Passed gate 3 — Continue from the retained candidate

The pinned CoreWeave comparison completed with matched evidence, prompt, tools,
verifier, task, caps, and publication policy. Retained finished 8/16 after 14 calls;
canonical finished 0/16 after 13. Neither arm gained score. Promote explicit selection
of the higher contemporaneously verified candidate as the next Builder's starting
artifact, while keeping it noncanonical and planner-ineligible. Do not infer that a
longer generic mechanics pass will improve it.

## Closed or demoted

- **Planner reachability — CLOSED.** The source path is wired and ten focused planner
  tests pass. Runs did not plan because objectives and verified models were absent.
- **Delete unused actor tools — DEMOTED.** Zero `read_file`/`ls` calls do not mean the
  actor ignored durable evidence: it repeatedly used `grid` and `wmlib.frames()`
  through `run_python`. Tool removal needs schema-cost measurement or an ablation.
- **Persistent Builder kernel — MOVED TO SUBSTRATE TEST.** Repeated helper authoring
  is supported; “the Builder is already 71% CodeAct” and “switching is mere porting”
  are not.

## Completion order

```text
explicit retained-candidate continuation selection
  -> preserve rollback and authority separation
  -> target a named representation failure before buying more mechanics calls
  -> only then consider another paid gameplay comparison
```
