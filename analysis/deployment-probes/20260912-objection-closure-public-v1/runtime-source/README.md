# Objection-closure runtime v9

Successor to the v8 epistemic questioner. The Actor and critic still use the same
reasoning-off Qwen3.8-27B-FP8 model with separate contexts, and only the Actor can
call tools or spend environment actions.

v8 ran validly on Kaggle on 2026-09-12 and completed zero levels. Its questions
were correct and source-bound; what failed was price and closure. All three games
died on wall clock with 91-94% of their action budgets unspent, and the critic
consumed 18.5-26% of that wall. On S5I5 one objection was restated four times for
93 seconds, and the game ended 32 seconds short of the budget needed to settle it.

v9 changes two things and keeps everything that worked.

**The critic reads a commitment packet, not the transcript.** Current commitment
and its predicted path, the newest Actor claims with stable refs, host-measured
transitions as changed-cell counts and regions, the wall and action budget with
seconds per executed action, the last five commitments' rationales as citable
claims, and the objection ledger. Live-probed on the preserved S5I5 trace across
four cards (72 calls): the critic cites rationales, never the code window, so
the window is three items; the packet is 2.6-6.7% of the prompts v8 read and
found the objective-proxy gap 6 of 12 times at the boundaries v8's critic did. Withholding
cell values is also the anti-drift mechanism: a critic that never sees a pixel
cannot spend a review asking which pixel moved, as CD82 did twice.

**One objection is pending at a time, and it is disposed of rather than obeyed.**
While an objection is open the critic is not called; the host restates it locally
at no model cost. A commitment closes it through `submit_actions(..., closes={})`
with `defend`, `narrow`, `withdraw`, or `probe`. Defending and probing leave the
commitment unchanged, `withdraw` must name the actions being retired, and a
retired action cannot return in bulk without a declared probe budget the host
later audits. Three refused commitments expire the objection: the critic is
fallible and a stalemate spends the budget that actually ends these games.

Retained from v8: Actor-only source binding, the three-field question schema,
event-driven triggers with no review-count cap, reasoning off, and invalid output
that does not consume a pending boundary.

The contract, its evidence, and the reconstructed packets are in
`analysis/critic-closure-contract-20260912/`.

This candidate has local deterministic and loopback qualification only. It has
not been run on Kaggle and has shown no gameplay benefit.
