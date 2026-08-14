# 0158 LS20 long-run attempt v20

## Result

The fresh run executed one action and then stopped during the second-cycle
commitment after Qwen exhausted all 12 CodeAct iterations without submitting
`submit_probe`.

- Run: `artifacts/live-ls20/20260812T143857Z-qwen36-autonomous-44`
- Run ID: `8745c70e14f94b08aa716d0547e3aa52`
- Actions: 1
- Levels completed: 0
- Provider calls: 34 successful calls
- Counted usage: 241,548 provider-reported input plus output tokens
- Transport: no 429, no ARC timeout
- Scorecard: terminal, non-resumable

The first claim and commitment were accepted; action 1 produced a public
successor and a contradicted exact-replacement verdict. The autonomous reset
worked and the second claim was accepted. In commitment phase, Qwen repeatedly
re-inspected the same region and history. Despite the prompt naming and
exemplifying every constructor, it spent late iterations listing or testing
them. On iteration 12 it successfully called `primary_unchanged()` but still
did not submit the commitment.

All provider responses completed successfully. There was no output truncation,
transport failure, IPC rejection, missing function, or hidden schema. The
separate `classification-correction.json` therefore identifies
`second_cycle_commitment_codeact_iteration_exhaustion`: a genuine agentic
completion failure under the frozen 12-iteration interface.

The executed 20-file candidate was hash-checked and snapshotted, and all
failure-packet checksums pass.
