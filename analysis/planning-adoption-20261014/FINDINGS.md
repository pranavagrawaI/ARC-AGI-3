# Live planning and probing adoption screen

The optional helper exposure failed the screen. Control completed one of four
continuations; planner, probing and combined completed zero of four each.
All sixteen episodes are audit-valid, including independent native replay.
Each treatment has four valid pairs, zero gains and one loss, so every frozen
promotion gate is `NOT_MET`. There is no promotion or submission.

| Restored checkpoint | Seed | Control | Planner | Probing | Both |
| --- | ---: | ---: | ---: | ---: | ---: |
| B1 request 73 | 1701 | 0 | 0 | 0 | 0 |
| B1 request 73 | 2903 | 0 | 0 | 0 | 0 |
| Raw control request 107 | 1701 | 1 | 0 | 0 | 0 |
| Raw control request 107 | 2903 | 0 | 0 | 0 | 0 |

One means the next level completed, confirmed by native replay. These are two
selected checkpoints of one game, not independent-game or full-game scores.

## What the traces show

Neither new helper was called in any of the twelve treatment episodes. No live
Python tool code even names `bounded_search` or `probe_transition`, and there
are no persisted helper execution records. The first wire requests contain the
search guidance in all eight assigned episodes and the probe guidance in all
eight assigned episodes. This rules out missing guidance as the explanation
for this lack of adoption.

Three B1 treatment episodes mention `bounded_search` in reasoning. They then
choose their own BFS, heap-based search, or manual simulation. For example,
`b1-1701-both`, response 1, explicitly chooses plain BFS after considering the
helper. `b1-1701-planner`, response 1, similarly considers the helper and writes
its own heap-based search. This supports failure of optional exposure to induce
use. It does not test how well the new algorithms work when invoked.

The successful control provides a useful positive trace. It predicts and tests
two real actions, detects a simulator mismatch on the second, changes its
growth/contact rule, and builds its own A* search. It validates a 35-action
model-valid route and executes it to
native level completion. That episode uses 37 new actions, 11 model calls and
18,390 generated tokens. Planning and prediction checks can occur in the
existing actor without these added helpers; this one success does not establish
their general reliability.

Other traces show concrete execution friction. In `b1-1701-planner`, custom
callbacks are rejected for retention because they depend on snippet globals;
later calls repeatedly fail on undefined `cur`, and a heap comparison fails on
dictionary states. It finishes with no new native actions. These errors and
incorrect simulators are observed failure modes, but this screen does not
assign every failed episode to one cause.

Fifteen episodes exhaust all 32,768 generated tokens. Across the matrix,
218,140 of 509,910 generated tokens, or 42.8%, occur in length-cutoff responses
with no parsed tool call. This describes a substantial observable interruption
pattern; it does not prove that those tokens were wasted or that a larger cap
would solve the games.

The successful control's last tool result includes a `ValueError` while parsing
the terminal frame after the winning action. Its action receipt and native
replay confirm completion before that error. It remains a gameplay success;
the post-completion parsing error is preserved separately.

## Evidence and cost

The private [Kaggle session](https://www.kaggle.com/code/pranavagrawai/planning-adoption-20261014)
is kernel 138688264, version 1, `COMPLETE`. There was one accepted launch and no
retry. The matrix exits normally at notebook elapsed 4733 seconds. Account
used-plus-reserved GPU time rises from 91,435.005 to 96,177.133 seconds, a
4,742.128-second delta, about 79 minutes. No active sessions remain at the final
account readback. The hard session cap was 9000 seconds.

All 225 selected downloaded files pass size/hash verification. All sixteen
restorations, declared first requests, request/response coverage and budgets,
action counts, protected-source identities and native action replays pass.
The package manifest and native engine hash match the frozen identities, and
effective serving settings match the declared stack. The matrix receipts cover
150 model calls and 75 new native actions with no unresolved request exposure.

[RESULT.json](RESULT.json) contains the independent paired audit.
[MECHANISM-EVIDENCE.json](MECHANISM-EVIDENCE.json) retains arm totals, helper-name
scans and eleven excerpts bound to raw file hashes and call numbers.
[collection.json](collection.json) records the raw artifact inventory.
[REVIEW.md](REVIEW.md) preserves qualification failures and the post-run replay
policy correction. The raw traces remain in the ignored local `output/` folder.

## Supported next question

The gap is between offering the generic functions and getting an actual
invocation. A minimal invocation template or an explicit callback handoff is a
bounded next hypothesis. First establish real helper use, then compare
completion. This result gives no basis to promote the current exposure or to
reject planning and probing as entire approaches. No second live run is part
of this experiment.
