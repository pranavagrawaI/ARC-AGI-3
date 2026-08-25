# Paid production Actor shadow v1 — stopped short run

## Judgment

The short feasibility gate **failed** and the twenty-turn phase did not run.
This is a valid bounded refusal, not a provider or route failure: all 12
dispatched calls returned from the frozen DeepInfra / DeepSeek V4 Flash route,
were usage-accounted, and cost $0.008028504 in total. No ARC action was
executed.

The Actor spent eleven provider calls investigating the first observation before
returning `UP`. The recorded trajectory's actual first action was
`CLICK(row=16, col=19)`, so the next historical observation was already
counterfactual to the shadow decision. On turn two, the model explicitly noticed
that mismatch, resumed investigation, and the next reconstructed request reached
a conservative input bound of 69,147 against the frozen 65,536 ceiling. The
reference monitor rejected it before dispatch and stopped the authorization.

## Observed causal chain

1. Calls 1–11 inspected files and repeatedly analyzed the initial 64x64 grid.
2. Call 12 returned the first terminal action, `UP`.
3. Shadow replay supplied the recorded post-`CLICK(16,19)` observation because it
   has no valid counterfactual environment.
4. The Actor noticed `UP` versus recorded `CLICK` and investigated the apparent
   inconsistency.
5. The following request was refused before dispatch because its serialized
   payload was 68,635 bytes (69,147 with protocol reserve).

## Resource record

- provider calls planned/received/route-validated/accounted: 12/12/12/12
- prompt/completion tokens: 199,838 / 5,302
- cached prompt tokens: 139,264
- reasoning tokens: 3,364
- actual spend: $0.008028504
- largest dispatched request: 63,145 bytes
- refused request: 68,635 bytes; bound 69,147
- validated shadow decisions: 1 of 6
- environment actions: 0

## Consequence

Do not raise the cap and rerun this treatment. Two mechanisms are now confounded
but both are production-relevant: the Actor lacks a stopping discipline on the
first decision, and multi-turn off-policy replay becomes causally false after its
first action divergence. The next experiment must either use checkpoints that
end at one decision, or use an environment/model capable of producing the
consequence of the Actor's proposed action. A long historical replay cannot be
interpreted as Actor continuity after divergence.
