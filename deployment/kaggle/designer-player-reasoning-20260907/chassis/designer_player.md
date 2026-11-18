# Designer–player reasoning

Discover and satisfy the game's success condition, not merely predict motion.

At first contact, or while the objective is unclear, take two perspectives. As a
player, identify what you control, what constrains it, and what might signal
success. As a designer, ask what task would make the visible objects,
relationships, affordances and feedback meaningful. Include static patterns and
objects, not only changed pixels.

Designer intent generates hypotheses; it is not evidence. Do not dismiss a simple
objective as too easy or favor complexity as more intelligent. Consider matching,
arrangement, reachability, transformation or relational conditions only when the
observed structure supports them. These examples are not an exhaustive menu.

## One-cell pre-action workflow

When the next test is already clear, keep the mechanical preparation in one
CodeAct cell so that a failed representation check does not cost another model
turn. Use measured summaries first and the canonical helper input:

```python
state = self.latest_state()
print(self.grid_summary(state["grid_rows"]))
print(self.last_transition())
grid = self.model_grid()                 # NumPy int array; not grid_rows
# inspect or update the helper here
self.load_helpers()
print(self.preflight_model(["UP"]))      # execution check; not proof
self.set_experiment(question, hypotheses, "UP", outcomes)
self.submit_actions(["UP"], rationale="predict: ...", predicted_path=[...])
```

Do every write, load, preflight, and experiment record before
``submit_actions``: a successful submission ends the turn. Use
``changed_region(padding=1)`` or ``render_grid(..., bbox=..., max_chars=...)``
for targeted evidence; do not print the entire history by default. A ready
preflight only says that the helper executes under the runtime representation.
It does not confirm the objective, grant batching, or replace the measured
post-action correction.

Before an informative action, identify the uncertainty that would change your
next choice. When plausible explanations compete, prefer a cheap bounded test
whose observable outcomes separate them. State the measured condition and outcome
window prospectively. Do not invent alternatives to fill a format.

Distinguish failing to reach the test condition, reaching it without the predicted
outcome, and being unable to measure it. Only the second contradicts that scoped
prediction. Revise the objective, dynamics or representation responsible; preserve
observations the replacement still leaves unexplained.

When mechanics predictions improve without level progress, revisit the objective
or representation before polishing another mechanics detail. Predicting a counter
or movement does not establish either as progress.

Before reset or another costly commitment, match the remedy to the doubt:
reinspect or recompute uncertain state; test uncertain goals through contact.
Prefer a reversible discriminating action when it can resolve the doubt.

Keep records short and update only what changed. A justified exploratory action
remains available when no useful prediction can yet be made.
