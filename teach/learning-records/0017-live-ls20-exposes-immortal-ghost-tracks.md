# Live ls20 exposes immortal ghost tracks in lesson 0011

Status: superseded by LR-0018

Lesson 0011's seeded 120-action audit preserved identity and classified
`SAME`/`MOVED` correctly on all 1,888 conservative exact-appearance matches, but
failed lifecycle correctness: it emitted 4,625 `OCCLUDED` events, retained 61
IDs for a scene showing at most 19 components, and ended with 42 non-visible
tracks. Occlusion needs a bounded memory and a disconfirmation path before its
events are safe for clause search; the audit still cannot judge real
recolour/reshape/occlusion transitions, and consistently tracking lesson 0010's
colour fragments does not repair the composite-object gap.
