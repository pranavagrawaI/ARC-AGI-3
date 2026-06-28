# Demonstrated: ran a predicted experiment; blind agents plateau at 0% and he knows why

He implemented `BlindMomentumAgent` (clean, with a `compare_agents` harness extension he added himself), predicted it wouldn't help much, measured it, and articulated the structural reason: ls20 requires *sequenced revisiting* (crossing the rotator repeatedly), not coverage — so correlated random walks don't address the bottleneck. Teacher added the complementary half: the move-bar reset erases unbanked exploration, so progress requires remembering what frames revealed, i.e. perception.

**Evidence**: working code in `teach/lessons/0003-random-agent.py`, prediction stated before result, correct causal explanation after.

**Implications**: Karpathy stage-2 discipline (input-independent baseline, predict-then-measure) is demonstrably internalized — stop scaffolding it explicitly. Ready for perception (lesson 0004, built on his own 405-step recording with check-harness feedback). Pending from lesson 0003: the lives-run-out experiment is still unreported.
