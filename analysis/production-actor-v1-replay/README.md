# Production Actor V1 replay

This package freezes six evaluator-hidden, real-workspace decision replays and
runs them through `tycho_nooa.agent.TychoNooaAgent`. The only substituted
boundary is consequence: `ActionDecision` is evaluated and never sent to ARC.

The checked-in scripted probe is an engineering proof, not evidence of Actor
cognitive quality. It proves packet integrity, evaluator isolation, production
controller reachability, and the absence of ARC actions. Its terminal outcomes
test legal transport only.

The first source-grounded limitation is explicit in every report: native
`read_file` evidence access is exactly observable, while `run_python` may read
arbitrary workspace evidence without record-level provenance. Such grounding
is reported as `unobservable`, never inferred.

## Zero-provider findings

- All six scripted production-controller paths reached an accepted terminal
  decision and executed zero ARC actions.
- The multi-transition path confirmed the Python-provenance ceiling above; no
  other recurring interface seam appeared.
- The first 60-turn shadow exposed linear model-visible history growth:
  provider requests rose from 26,240 to 107,076 bytes while exact evidence rose
  from 9,782 to 618,051 bytes.
- One generic repair now collapses old Actor events into structural facts while
  retaining the original NOOA events and exact workspace evidence. The replay
  then stabilized after warm-up: turns 6-60 stayed between 49,255 and 50,284
  request bytes with a 1.18-byte/turn fitted slope, while exact evidence kept
  growing to 618,051 bytes.

These runs use scripted historical actions. They establish reachability,
authority isolation, and resource behavior—not decision quality.
