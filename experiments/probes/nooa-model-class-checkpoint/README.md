# NOOA model-class checkpoint probe

## Question

At the same archived contradiction checkpoint, does GPT-5.6-sol revise the active
`UP`/moving-ceiling theory where Gemma repeats it?

This is a one-call-per-model shadow probe. It performs **no environment actions** and
is not a full NOOA run.

## Frozen checkpoint

The model-visible transcript is the preserved arm-C input immediately before Gemma
submitted another five `UP` actions at environment step 26, CodeAct round 2:

`../nooa-gemma-epistemic-context/results/arm-C/20260730_093940_ls20_memory_contradiction_first_gemma-modelrun-fp4_cap44/agent_logs/nemo/team_leader/messages/step_026_round_02_user.md`

The complete transcript is wrapped identically for both calls. No tools are exposed;
each model must return one structured shadow decision.

## Bounds

- exactly one request per model;
- order: Gemma, then GPT-5.6-sol;
- zero retries and zero redirects;
- no tools and no environment actions;
- maximum 3,000 output/reasoning tokens per request;
- reasoning enabled on both routes;
- Gemma route: `google/gemma-4-31b-it`, `modelrun/fp4`, no fallback;
- Sol route: `openai/gpt-5.6-sol`, `openai`, no fallback;
- projected catalog upper bound: approximately USD 0.22 for both calls together;
  this is not a provider-side financial cap.

## Interpretation

- Sol revises while Gemma repeats: candidate evidence that the original soft NOOA
  contract is model-class dependent.
- Both repeat: candidate evidence that transcript ordering/evidence presentation is
  the stronger limitation.
- Both revise: the prior live failure more likely depends on CodeAct timing,
  tool-state visibility, or the weak host comparator.

One response per model establishes only a checkpoint-local behavioral difference.

## Observed result

The original pair stopped fail-closed after Gemma consumed all 3,000 completion
tokens as reasoning and emitted no structured final. That call cost `$0.0064064`.
Its preserved reasoning nevertheless rejected another `UP` macro and ended at a
one-action `DOWN` probe.

The user then explicitly authorized the still-unspent Sol call as a replacement
comparison. Sol returned a valid structured decision after 657 reasoning tokens:

- status: `ABANDON`;
- contradiction: the two-pixel changes were not the predicted player/ceiling movement;
- next action: one `RIGHT` probe;
- cost: `$0.136285`.

Total observed cost was `$0.1426914`.

Both models operationally stopped repeating `UP`. Sol was much more concise,
explicit, and schema-compliant. Therefore this checkpoint does **not** show that
Gemma cannot revise while Sol can. It shifts the leading explanation toward the
live CodeAct loop: observation/action ordering, accumulated self-conditioning, and
the host's weak `diff_pixels > 0` success proxy. See `audit.json` for the bounded
interpretation and `run/` plus `replacement-sol/` for raw evidence.
