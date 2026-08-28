# Unified Tycho Nooa one-action contact

Status: explicitly authorized by Pranav in the consolidation implementation request.
This is a diagnostic integration contact, not candidate promotion evidence.

## Question

Can the unified `trigger/control` incumbent cross the real path

```text
arc-host -> worker -> persistent Actor -> one ARC action -> accepted successor -> finish
```

without a protocol, sandbox, schema, provider, timeout, or trace failure?

## Bound source

- Worktree: `/home/hipra/projects/arc-agi-3-tycho-nooa-unified`
- Branch: `refactor/tycho-nooa-unified`
- Fixed point: `fa190780ac20dd50b3b45a9ec7b63abbfda8bd79`
- Recovery record: `/home/hipra/archives/one-solver-consolidation-20260828/unified-live-source-final/RECORD.json`
- Recovery-record SHA-256: `071c916994c8e340170e5f10b4606795a934bc9db01c7a7c7bdef43b5b796055`
- Tracked-patch SHA-256: `ee5e53450c274674b0caabb510e2879f6a6f07b90f517c7e026e7a84da9490cd`
- Untracked-manifest SHA-256: `125dd17339d4312e353b2e1242b9734fc89929f52fbe8da18fcbf0f1e22beee7`

The referenced environment file supplies credentials and route configuration but
is not copied, printed, or committed. Explicit shell values select the Solver
mode, treatment, and sandbox because the host loads dotenv with `override=False`.

## Frozen run

- Game: `ar25`
- Seed: `0`
- Mode: `trigger`
- Treatment: `control`
- Sandbox: `host`
- ARC action cap: `1`
- Stop-after-levels: `1`
- Worker timeout: `900` seconds
- Provider-call bounds: existing Solver bounds only, at most 40 Builder tool
  rounds plus 25 Actor tool rounds and the final action demand for this decision.
- Output: `analysis/unified-micro-v1/runs/trigger-control-ar25-s0-1a/`

## Required evidence

- host summary and transcript;
- worker stdout/stderr;
- `tool-trace.jsonl` and `llm-trace.jsonl`;
- decision checkpoint for every provider call;
- ready metadata naming mode, treatment, and sandbox;
- one accepted action and its host-confirmed successor;
- finish acknowledgement;
- observed provider/model/call count/token use/cost when reported.

## Result map

- One accepted action, accepted successor, clean finish, and complete traces:
  retain the unified runtime as integration-valid for one action only.
- Provider/model failure before an action:
  preserve the run and diagnose transport; make no Solver claim.
- Protocol, sandbox, schema, or trace failure:
  stop. Repair offline before another live call.
- More than one ARC action, route drift, wrong mode/treatment, or source drift:
  invalidate and preserve.

The run cannot establish gameplay benefit, transfer, decision-discipline effect,
or promotion-grade empirical evidence.
