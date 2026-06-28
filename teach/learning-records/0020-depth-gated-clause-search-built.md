# Lesson 0012 learns the first minimal law and fails loudly outside its grammar

The closed-alphabet reasoning layer now exhaustively searches conjunctions by
depth over confidence-gated event facts, demands consistency on trusted events,
and uses a two-part MDL proxy to prefer shorter exact clauses. Its synthetic
trace learns `contact AND target_shape=cross` at depth 2 after 56 candidates,
filters `UNCERTAIN`/`LOST`, never enumerates depth 3, and returns
`GRAMMAR_GAP` when a single conjunction cannot express the observations; live
ls20 clause induction remains blocked on trustworthy interaction events.
