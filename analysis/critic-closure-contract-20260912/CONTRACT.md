# Objection closure contract

Written before implementation, from the 2026-09-12 public run. Every number is
reproduced by `evidence.py`; the packet measurements by `s5i5_fixture.py`.

## The reprioritisation

The handoff ranked full-context latency fifth of six defects. The trace ranks it
first, because it decides the budget everything else spends.

| game | actions used | limit | wall | critic wall | critic share | s/action | stop |
|---|---:|---:|---:|---:|---:|---:|---|
| LS20 | 17 | 200 | 625s | 162.5s | 26.0% | 36.8 | model admission |
| S5I5 | 18 | 200 | 607s | 125.1s | 20.6% | 33.7 | wall budget |
| CD82 | 12 | 200 | 616s | 113.6s | 18.5% | 51.3 | model admission |

Every game died on wall clock with 91-94% of its action budget unspent. A model
call is the expensive resource here; an environment action is nearly free. The
critic was priced as if the reverse were true.

## What the S5I5 fixture actually shows

The handoff calls the central defect "detection without closure". The trace does
not support that reading, and the difference changes the design.

At turn 11 the Actor closed the objection explicitly, in its own words:

> CRITIC: correct. The bar (17/64) is not the completion condition — I've been
> burning clicks. Withdraw the "fill bar to complete" claim; it's just a counter.

It then probed an object. The probe failed: objects were static. At turn 12 it
returned to bar-filling, and again said why:

> KEY TEST: Is the bar the actual win condition? Let me just fill it and see.
> But that's 46 clicks. Let me be efficient: batch when allowed.

That return was a deliberate, correctly reasoned test of the challenged claim —
the only live hypothesis left. It was not inattention, and a contract that
hard-blocked it would have forbidden the one action that could settle the
question. Detection produced closure. What it did not produce was an
**alternative**, or the **budget** to run the test.

The arithmetic is the finding:

| quantity | value |
|---|---:|
| wall remaining when the Actor resumed the test | 112.5s |
| cost of a ten-click batch turn | 31.4s |
| clicks still needed to settle the bar | 46 |
| wall needed | 144.6s |
| **shortfall** | **32.2s** |
| wall the four redundant re-asks consumed | 93.0s |

The critic restated one objection four times, spending 93 seconds — nearly three
times the shortfall — and the game ended with the question it kept asking still
unsettled. Repetition was not merely ineffective; it consumed the budget that
would have answered it.

## Design consequences

1. **Repetition must be free.** One objection is pending at a time. While one is
   pending the critic is not called; the host restates it locally at zero model
   cost. This converts S5I5's 93 seconds to roughly zero.
2. **Dispose, do not obey.** The Actor answers with a disposition, not agreement.
   `defend` and `probe` both leave the intended commitment unchanged, so a wrong
   question costs a sentence rather than a pivot.
3. **`probe` is a first-class answer.** "I will spend 46 clicks to settle this"
   is a legitimate response to "this is untested". The host records the budget
   and later asks what those actions decided.
4. **The critic must see the price.** The packet carries wall remaining, actions
   executed, and seconds per executed action. At S5I5 critic:7 that reads
   `81.0s remaining, 30.5s/action, 2 affordable actions` — which makes "have you
   tested filling the bar to 64?" visibly the wrong question to spend a review on.
5. **Cut the input, not the prompt.** Epistemic drift into mechanics (CD82's
   "why did two SPACEBAR transitions differ by one cell?") was made possible by
   showing the critic everything. The host now supplies changed-cell counts and
   regions, never cell values, so the material for pixel debugging is absent.

## States

```
      critic question                 Actor disposition
none ──────────────────▶ pending ─────────────────────▶ closed
                            │  ▲                          │
              host restates │  │ recurrence of a          │ withdraw
              (no model call)  │ retired action           ▼
                            │  └──────────────────── retired claim
                            │                             │ probe budget spent
                            └── 3 refused commitments ─────▶ host asks what it decided
                                (expired: the critic is fallible)
```

## Rules

**Opens.** A valid critic review with a nonempty question. Nothing else.

**Allowed while open.** Everything. The Actor keeps sole action authority, and
every disposition except `withdraw` permits the commitment it was about to make.

**Closes.** `submit_actions(..., closes={...})` with one of:

| disposition | requires | means |
|---|---|---|
| `defend` | `evidence` | the observation that warrants the claim |
| `narrow` | `evidence`, `narrowed_claim` | the smaller claim still held |
| `withdraw` | `evidence`, `retire_actions` | the actions that will stop |
| `probe` | `evidence`, `actions_budgeted` | settle it by acting, within N actions |

`withdraw` must name actions. An abandonment that retires nothing cannot be
checked for recurrence, and is not an abandonment.

**Enforced.** A commitment that carries no disposition while an objection is open
is rejected locally, with the four options in the rejection text. The Actor can
dispose and resubmit within the same call.

**Retention.** A later batch that is half or more retired actions is rejected
until it declares a `probe` budget. This is the S5I5 turn-12 correction: the
return is permitted, but it must name its price and be held to it.

**Accounting.** When a `probe` budget is spent, the host asks once what those
actions decided. Otherwise `probe` would be an escape hatch.

**Expiry.** After three refused commitments the host drops the objection and
lets the commitment through. The critic is fallible, and a stalemate spends the
one budget that actually ends these games.

## What the host never does

It never judges a game fact. Every check is arithmetic over action strings and
counters the Actor itself supplied: string equality on retired actions, a share
threshold, a step deadline, a refusal count.

## What this does not establish

- The contract is tested against preserved traces and a live loopback path. No
  Kaggle run has exercised it.
- The packet is source-bound and, including the system prompt and the
  `recent_commitments` block, 4.2-10.1% of the prompt v8 read (endpoint-reported
  tokens 1.13-1.22x the fixture estimate; 14.5% worst case at 2.5 chars/token).
- Three live probes (48 calls, qwen/qwen3.8-27b, reasoning off) tested whether a
  critic reading the packet still finds the S5I5 objective gap. v1 was
  contaminated by a fixture bug and is reported as such. v2 (bare packet) found
  it 3 of 12 times at critic:3/5/7. v3 (with the last five commitments'
  rationales as citable claims) found it 6 of 12, cited a rationale in 9 of 10
  valid questions, and cited the ten-item code window never. v4 deleted the
  window: at 0 the rate fell to 4 of 12, at 3 it held at 6 with all 24
  responses valid and prompts of 1.7-3.1k tokens. Window 3 is adopted by the
  pre-registered map. No arm reached 8 of 12; across four arms the critic
  cited a rationale 30 times and the code window 0 times. The packet the
  critic reads is now 2.6-6.7% of what v8 read. See
  `packet-live-probe-v{1,2,3,4}/RESULT.md`.
- The packet's budget block produced affordability questions ("58 more clicks
  at ~38s per click but ~9 affordable — which non-click action would finish the
  level?") in roughly a third of valid responses. v8 never asked one. Whether
  that changes Actor behaviour is untested.
- The wall-clock saving is arithmetic from the preserved trace, not a measured
  outcome. It predicts more actions per game; it does not predict a level.
- No causal claim about the critic's value survives this run either way. The
  contract's defence is that it makes the critic cheap enough that the question
  of its value stops being urgent.

## Live contact (2026-09-12, runtime v9, run invalid)

One public run reached LS20 and died there: the Actor declared the game "stuck"
with 152s and 193 actions left, returned DONE, and the inherited v8 chassis let
that silence kill the launcher and the two remaining games. The trigger was the
packet's `affordable_actions_at_this_rate`, which the critic restated as fact.
On the one game, the contract behaved as specified (3 objections, probe /
narrow / withdraw, probe accounted, 0 reminders, 0 expiries) and the critic cost
1.3% of session wall (v8: 26.0%). Runtime v10 removes the projected count from
the packet and removes the Actor's DONE affordance, redelivering the state on a
refused DONE. See `analysis/deployment-probes/20260912-objection-closure-public-v1/RESULT.md`.

## Deliberately not changed

`prediction_feedback.batch_error` grants batch permission only when the last
prediction matched. Batching is the only way to spend actions cheaply under a
wall-clock budget, and the most predictable action is usually the well-modelled
proxy the critic is challenging — so the chassis pays the Actor to repeat it.
S5I5 turn 11 earned batch permission with an object probe and spent it on ten
bar clicks. This is a real counter-incentive to the critic, but it is a separate
mechanism with its own blast radius, and the turn-12 return was defensible on its
own terms. Recorded here; not touched in this change.
