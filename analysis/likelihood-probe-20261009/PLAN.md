# Likelihood probe: plan and pre-registered readouts (written 2026-10-08 22:00Z, before any GPU result)

**Question.** Can a sampling-free score, the teacher-forced probability the model gives a good action at a fixed logged
state, serve as a hill-climbing metric? Sprints are Bernoulli-bound (`../sprint-power-20261009`). This probe has no
sampling noise by construction, so the open question is validity, not noise.

**Items.** There are 355 decision points from dfranzen-control: the last request of a turn, with images stripped. Only
control kept request logs. 67 items are on BFS-oracle levels and have an optimal-action set. 328 are on levels with a
non-move decisive action. Candidates are the legal keys plus up to 48 component clicks, always including the taken and
decisive clicks. The median is 33 candidates per item.

**Variants.** R0 has an empty think block. R1 conditions on the model's own logged reasoning. KF0 is R0 with KF's
Duck system prompt (same model; hidden score 2.96 vs 23.87). MIN0 is R0 with a one-line system prompt. Kernel
`arc3-lprobe-a1` scores R0 and R1. Kernel `arc3-lprobe-b1` scores KF0, MIN0, then R0 again, which is the session-repeat noise floor.

**Known bias.** Summed string log-prob favours short keyboard codes over click codes in mixed games. A mock with
uninformative token log-probs showed +0.43 log-lift. Read V1/V2 in the **same-kind** normalisation. V3 is paired per item, so the bias mostly cancels there.

## Readouts and what each decides
| check | pass | if it fails |
|---|---|---|
| V0 instrument | session repeat top-1 identical ≥ 95%, mean abs dP < 0.01 | the score has numeric noise; size it before using |
| V1 policy (R1) | same-kind log-lift on the taken action > +1 | the scorer is broken (R1 contains the decision itself) |
| V1 policy (R0) | log-lift > +0.5 | the intuitive choice is unrelated to the policy; R0 cannot proxy behaviour |
| V2 quality (R0) | P(opt) lift > 0 and AUC(P(opt) → run took optimal) ≥ 0.6, or AUC(P(dec) → solved ≤ 10 min) ≥ 0.6 | the probe does not see correctness |
| V3 prompts | R0 − KF0 > 0 with t ≥ 2, positive in the majority of games; MIN0 lowest | the probe cannot rank prompts the way hidden does |

**Hill-climbing candidate** = V0 + V2 + V3 all pass. Then the next step is a second known pair (e.g. lean-cap4 prompt) and a
cost check (seconds per item under arm changes). If only V1 passes, the probe measures imitation, not quality. Drop it.
