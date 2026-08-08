# 0004 — Autonomous objective-relation acquisition

This experiment removes historical routes from the producer boundary. Generic scale enumeration
acquires LS20 objective pairs on levels 1, 3, and 4 from prefix-only observations. Level 2 retains
seven compiled scale-consistent hypotheses; local E4B must select one or abstain. A scorer-only
terminal observation later tests whether the proposed executable objective closes.

The first E4B arm safely abstained. Generic representation deduplication then identified six
hypotheses as synchronized strict subtracks of another mutable track, leaving one compiled objective
without consulting the terminal label.

This is an offline acquisition gate. It does not yet produce a route or authorize an environment
action.

The next planning gate reconstructs the acquisition checkpoint by replaying only its already-seen
prefix in the real local LS20 engine. E4B receives four chronological frames, the inferred action
model, tracked positions, and the acquired objective. It receives no candidate plan, route suffix,
or protected successor. Each submitted plan is executed from a fresh checkpoint; only actual level
completion passes.

The subsequent feedback gate keeps that historical checkpoint and the compact replay-supported
causal package, but removes full-route commitment: E4B chooses one action, receives the actual
public successor, and chooses again. This condition also failed 0/3. It is still prefix-conditioned,
not autonomous play from reset, and the intact causal package already fails; lower-information
ablations are therefore not licensed by this result.
