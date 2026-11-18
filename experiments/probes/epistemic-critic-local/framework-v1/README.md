# Framework critic candidate

Status: drafted, not wired into the runner or tested with a model. Existing frozen
prompts and run artifacts are unchanged. `system.md` is the candidate instruction;
this document and evaluator diagnoses must not enter the critic's context.

The behavioral change is from assessing an Actor's explanation within its framing
to assessing the commitments and exclusions produced by that framing. Same Qwen
weights and separate context do not establish independent errors. Independence
must be measured through warranted interventions, including justified permission
to act while uncertainty remains.

The existing five-field interface remains. One deliberate semantic change removes
approving rationales from `no_concern`: its text fields are empty. This prevents an
absence-of-objection response from adding an unearned factual endorsement; it does
not by itself improve detection. Existing host source binding, output validation,
execution status, and Actor response responsibilities remain outside the prompt.

Prior evidence: `analysis/local-probes/20260911-critic-endurance-v1/RESULT.md` and
`CASE-R8.json`. The prior critic certified a computed reachability conclusion despite
a contradictory placement check in its input. The Actor repaired the search itself.
This motivates distinguishing program reports from conclusions about the world,
without adding a factual-verification or debugging subsystem.

Before another live run, freeze a diagnostic card comparing the existing prompt
with this candidate. Include the captured R8 boundary and sound controls, including
an Actor that explicitly notices the discrepancy and investigates it, a bounded
probe under uncertain goals, a properly scoped negative observation, and a justified
rejection of an alternative. Include other games and untouched commitments before
claiming generalization; success on R8 is development evidence because it informed
this prompt. Supply no later outcomes or evaluator diagnosis to either critic.

Score warranted challenge, unsupported challenge, missed consequential gaps, and
whether the proposed minimal revision preserves justified action. Empty no_concern
fields eliminate an endorsement channel, so their disappearance cannot be scored
as improved epistemic detection. Measure validity, latency, and tokens separately.
Only subsequent real Actor decisions can establish adoption and retained repair;
only suitable controls can attribute benefit to the separate role.
