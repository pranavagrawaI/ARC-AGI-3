# Operational cycle, version 1

Callable through CodeAct: `print(self.workflow_skill())`. Execute the cycle through
`self.operational_cycle(operation, payload)`. This is the sole authoritative
model-authored knowledge store. Notes, helpers and procedures remain available,
but are scratch or derived artifacts until their evidence/claims or executable
hashes are registered here. Existing tools remain available. Do not infer truth
from host acceptance: the host checks structure, references and bytes only.

1. Observe: inspect latest_state, trajectory, observations, and measurements.
   `operational_cycle('read')` returns current theory, digest, schema and available
   evidence bindings. `operational_cycle('observe', ['transition:1'])` starts a cycle.
2. Reconcile: submit `{'evidence_refs': [...], 'claim_refs': [...],
   'disposition': 'what supports, challenges or leaves claims unresolved'}`.
   On genesis, claim_refs may be empty. On later cycles use stable existing IDs.
3. Revise: submit a FULL theory object matching the returned JSON schema.
   Set schema_version=1, version=previous+1 (genesis 1), parent_sha256=last digest
   (genesis null). Keep every prior claim and evidence reference. Change claims
   with a revision entry binding their previous SHA256, reason and evidence.
   Status is your judgment, never a host-certified truth. Retain tested scope and
   open conditions. Enter observations, action semantics, objectives, uncertainties,
   experiments and procedure references. Unknown objectives are legitimate claims.
   Set current_policy=null here. Use self.operational_hash(value) for canonical hashes.
   Register executable helpers with self.register_operational_artifact(filename)
   before citing the returned path/hash in procedures. Empty procedures are allowed.
4. Identify: `select` with `{'kind':'uncertainty'|'procedure','id':...}` naming
   an active entry in that theory. Select only uncertainty relevant to the decision,
   or a procedure whose stated applicability you judge to hold.
5. Declare: `declare` with `explore`, `exploit`, or `repair`.
6. Prospective decision: `decide` with theory_sha256 (the revision's digest), mode,
   selection (same object as select), nonempty claim_refs and evidence_refs,
   actions (one action), rationale, expected_observation, revision_if_wrong.
   This creates a new canonical version containing the policy. It must precede action.
7. Prepare: `prepare` with the EXACT actions array from the decision. Then call
   self.submit_actions with that same array and your rationale. This experiment
   records a proposed action only; no game runs.
8. Reconcile result: when a new recorded observation arrives, `read`, then submit
   `result` with `{'theory': FULL_NEXT_VERSION, 'observation_id':'transition:2',
   'relation':'recorded_continuation_not_action_outcome'}`. Add the evidence, retain
   the policy, and revise only what you think it warrants. This recorded successor
   resulted from the historical action, NOT your proposed action. Never score it
   as a counterfactual test. Then begin the next cycle and propose another action.

Ordinary format errors return REPAIR and preserve the previous phase and accepted
state. At most three repair responses per arm; the fourth ends construction as
incomplete. No extra model-call allowance. Integrity/accounting/provider errors
stop the pair. Use short cells; combine operations where appropriate. No separate
consolidator exists. The same store persists across turns; `read` resumes it.

Bound helper filenames are sealed in the host-owned read-only helper directory.
To change bytes, use write_helper with a new filename, register it, and revise the
canonical procedure path/hash before selecting it. Rewriting a sealed name is rejected.
Observation APIs and the game-state producer read host-owned /evidence. Any writable
IPC copies are scratch, never authoritative evidence or valid evidence references.
