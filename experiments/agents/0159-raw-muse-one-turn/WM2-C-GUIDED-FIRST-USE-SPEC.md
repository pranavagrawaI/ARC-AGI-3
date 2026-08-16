# WM2-C Guided First Use

Status: accepted implementation specification.

## Decision

The first WM2 live run exposed the Theory Practice Guide and a relevant
Workbench View on every decision, but Muse made 59 ad hoc Python calls and
never read or edited the executable-theory lane. WM2-C changes delivery, not
theory semantics or execution authority.

When all of the following are true:

- `repeated_action_evidence` is active;
- no executable commitment is active;
- Muse has not yet called `read_world_model()` in this run;

the Workbench View MUST append one bounded Guided First Use instruction. It
MUST:

1. connect repeated reconstruction to the lane's preservation role;
2. give the exact first call: `self.read_world_model()`;
3. state that reading is read-only and creates no commitment;
4. name the three subsequent choices: narrow edit, prose hypothesis, or defer;
5. leave the hypothesis, edit contents, and environment action to Muse.

The combined Workbench View and instruction remains under WM2's 6,144-byte
bound. Overflow fails the model turn rather than silently dropping treatment.

The instruction is mandatory guidance but is not a host gate: ignoring it MUST
NOT invalidate or duplicate the environment action. Once Muse reads the lane,
it MUST disappear on later regenerated views. A commitment also makes it
inapplicable.

## Modularity and evidence

WM2-C requires WM2-B but remains separately configured, source-committed, and
audited. It MUST NOT change E0, evaluator behavior, action vocabulary, provider
caps, compaction, or the one-action boundary. The audit records whether the
instruction was rendered and whether a later `world_model_read` occurred; it
does not infer comprehension or causal influence.

## First-run claim ceiling

One treatment-only run may establish exposure, compliance with the read
instruction, subsequent edit or explicit deferral, retrospective replay,
latency, gameplay, and failures. It cannot establish behavioral improvement,
prospective validity, or causal influence on action choice.
