# Long stalls that ended in levels: what broke them, and would a context reset pay? (2026-09-30)

**Result:** no. A stall-triggered context reset is not an action-economy win on this harness. It is not a
reliable escape either. Across independent runs, what a level costs is set early, by which observation the
mind happens to frame the level around. But a reset cannot draw that luck again for less than the slow path
still has to pay. The same arithmetic undercuts wiping and "keep the notes": wiping pays full relearning, and
keeping the notes keeps the frame.

Sources: P1/P2 transcripts + events (`duck-qwen38/runs/flash-next/out-p{1,2}`), and 8 full runs on the same
25 games (kf2, c1, r1, g1, g2, p1, p2, k2).
Tools: `stalldigest.py` (per-turn digest of every >20 min stall that ended in a level, 54 stalls, `digests/`),
`compact.py` (read view), `levelcosts.py` (actions to clear per game×level per run → `levelcosts.json`),
`restartcalc.py` (fixed-cutoff restart arithmetic).

## 1. What broke the stalls (13 read in full, longest first)

| stall | actions / min | what broke it | other runs, same level | a reset would have… |
|---|---|---|---|---|
| P2 g50t L1 | 284 / 123 | built plate → gate → replay-ghost mechanic step by step | P1 39, K2 64 | destroyed load-bearing mechanics |
| P2 cn04 L2 | 75 / 111 | mechanics right early; ~1 h of bugs in its own search code (13 yields) | 39 | nothing (code bugs, not beliefs) |
| P1 m0r0 L1 | 51 / 109 | one untested RIGHT at the bridge; 8 consecutive yields (≈46 min) deliberating | 19–40 | possibly time; actions already fine |
| P2 sk48 L1 | 214 / 101 | ~7 arrangement goals refuted at ~20 actions each, timeout reset, then a clear the model did not predict | P1 **17** | maybe: wrong frame from action ~15 |
| P2 su15 L2 | 141 / 93 | merge chain 2×2→3×3→4×4 found in the 3rd timed attempt | KF 29, P1 57 | destroyed mechanics built across attempts |
| P1 wa30 L2 | 192 / 75 | theory right from L1; execution and perception misreads, timeouts | only clear | nothing |
| P2 ka59 L1 | 120 / 71 | a push experiment showed pushed boxes slide through the wall | 19–38 | destroyed the push rules it rested on |
| P2 tu93 L3 | 81 / 69 | trap rules learned by dying | 24–72 | nothing |
| P2 cd82 L1 | 58 / 67 | stamp/rotate mechanics, progressive | 36–106 | nothing |
| P1 su15 L2 | 57 / 64 | merge found on attempt 2 | | |
| P1 bp35 L1 | 35 / 60 | shifting sub-layouts, 4 yields | 22–76 | nothing |
| P1 ls20 L1 | 79 / 51 | rotation gadget + a map bug, timeout reset | 17–84 | nothing |
| P1 dc22 L2 | 67 / 48 | pivot-ride test, then a hidden button appeared; used L1's win rule | 76, 89 | would have lost the L1 rule |

Patterns:
- **Stalls end through mechanics built up over time plus one targeted experiment**, not by dropping a locked
  goal. In 9 of 13 the accumulated knowledge was load-bearing.
- **The slow minds are the tail of a distribution whose fast mode exists.** sk48 L1: P1 saw a HUD icon go
  hollow when its beam touched red (≈ action 9) and inferred "touch targets in HUD order": 17 actions. P2
  framed it as block arrangement and spent ~190 actions refining real but irrelevant mechanics. su15 L2: KF
  merged two blocks by accident at ~action 20; P2 found the merge on its 3rd attempt.
- Where actions leak: expensive goal tests (a full ~20-action configuration per hypothesis: sk48, su15 P2).
  Where time leaks: underspending on cheap tests (m0r0, 8 yields before a 1-action probe) and self-written
  code bugs (cn04).
- Harness defect seen, low impact: m0r0's carried "Goal model" kept injecting a goal refuted at 14:01 for
  12 turns ("never yet tested"), because yielded turns never rewrite the note. The model ignored it. The R
  arm's parser fixes were null on levels.

## 2. Restart arithmetic across 8 independent runs

Cost per clear with cutoff τ = E[min(X, τ)] / P(X ≤ τ). This assumes a restart is a fresh independent draw of
the whole level, which is optimistic: it pays full relearning but gets full independence.

- Most levels are 1.2–5× between the fastest and slowest mind, and restarting never beats continuing on them
  (e.g. cd82 L1 64 → ≥108, dc22 L1 91 → ≥224, wa30 L1 81 → 149, re86 L3 49 → 49 at best).
- It wins on action cost only for **sp80 L1** (152 → 54 at τ=15) and **lp85 L2** (21 → 17).
- It turns rare clears into likely clears on sb26 L2, sk48 L1, tn36 L1 and ar25 L2, at ~100–300 actions each
  (sk48 L1 192–437). That is about what the slow runs already spent.
- **The big tails are partly arm effects, not luck:** sp80 L1's fast clears are both G (7, 11; image arm; the
  next fastest is 136). sb26 L2's only clears are both P (16, 36; 0/6 elsewhere, ≈3.6 % by chance). A restart
  does not re-draw the arm.

## Decision

A reset cannot win both ways. It pays when the draw is fresh, and fresh means paying full relearning
(section 2 says that loses). It is cheap when notes are kept, and keeping notes keeps the frame, while the
section 1 reads say the frame-plus-mechanics is what broke 9 of 13 stalls. Don't build it.

The observation that could change the decision is an early framing signal. It would have to separate fast
draws from slow ones in the first ~15 actions of a level, the way P1's hollow HUD icon did on sk48. If such a
signal exists, a cheap early re-frame (not a late reset) is the candidate. Test it offline on these 8 runs
before any GPU spend.
