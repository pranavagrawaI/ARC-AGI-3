# Kimi K3 compatibility run: after action

## Judgment

This is a valid protocol-compatibility result. Kimi K3 passed the Native termination prerequisite and failed the sterile-CodeAct termination prerequisite. Because eligibility required both arms to pass, the pairing is not eligible for the cognitive parity experiment under `parity-compat-v2`.

Six provider calls cost USD 0.01710285. Every call used the committed `moonshotai/kimi-k3` model through the exact DigitalOcean endpoint. The ladder stopped after the matched Gate 0 pair; no evidence, representation, cognition, or live-ARC gate ran.

## Native arm

Kimi reached the correct typed answer on call three. Its first return omitted the `task_kind` discriminator. Its second added the discriminator but omitted `evidence_refs`. Its third corrected both fields and passed. This establishes Native protocol compatibility within the precommitted three-call ceiling, although only at the boundary.

## CodeAct arm

Kimi did not terminate validly within three calls:

1. It used `execute_python` to create a local dictionary with the wrong evidence field name instead of returning immediately.
2. It emitted plain assistant text (`result`) rather than exactly one tool call, which the controller rejected.
3. It called `return_result`, but again used `evidence` instead of the required `task_kind` and `evidence_refs` fields.

The failure is therefore protocol formation, not evidence retrieval, grid representation, or ARC reasoning. The evidence capability was disabled and no evidence query was needed.

## Consequence

Do not advance Kimi K3 to the twelve-packet Native-versus-CodeAct screen under this frozen protocol. A different CodeAct schema presentation could be tested only as a new intervention with a new compatibility package; it must not be treated as a retry of this result.
