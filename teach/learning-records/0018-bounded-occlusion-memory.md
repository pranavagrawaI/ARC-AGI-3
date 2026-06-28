# Bounded occlusion memory separates tracker failure from world events

Lesson 0011 now emits `OCCLUDED` only on entry, retains hidden identities for a
configurable four-frame horizon, requires stronger evidence for `REAPPEARED`,
and retires expired hypotheses as epistemic `LOST` rather than causal
`DISAPPEARED`. On the same seeded ls20 audit this removed all consecutive
occlusion re-emissions and bounded hidden age to 3/4 frames, while the remaining
233 `NEW_OBJECT` and 224 `LOST` events locate the next problem in lesson 0010's
fragmentary objectness rather than identity-memory leakage.
