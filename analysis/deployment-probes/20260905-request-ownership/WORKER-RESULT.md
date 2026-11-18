# Worker result: retained answer, redundant state wakeup

The real notebook worker retained the held operation and delivered its answer.
It then started another generation for the same already-submitted state. A
separate six-line dispatcher candidate removes that reproduced redundant wakeup.
It does not change NOOA shielding, SGLang, reasoning limits, or the queued Kaggle
notebook. No inference-speed or gameplay gain is claimed.

## The actual boundary

In this join, `WAIT` is the result of `submit_actions`, after it has written an
action batch. It is not a short timeout that detaches a waiter from a pending
model operation:

1. `HostSession.decide` enters `_wait_batch`, polling the action file every 20 ms.
   It retains the same stack frame and monotonic deadline throughout that wait.
2. The headless dispatcher directly awaits `agent.handle(notification)`.
3. The controlled HTTP response calls the real CodeAct `execute_python` tool,
   which calls `self.submit_actions(["UP"], rationale="predict: move upward")`.
4. `submit_actions` writes the action and signals a WAIT result. The host returns
   the decision; the headless dispatcher can then wait on its notification queue.

There is no production pair of short WAIT callers between which to inject model
completion. We tested the reachable receive gap: allow the action-producing
operation to finish and return WAIT before the driving host attaches its protocol
receiver. The buffered decision remains deliverable once. This does not qualify
a detached-future interface that this worker does not implement.

## Decisive observations

| Control | Original notebook worker | Isolated dispatcher candidate |
|---|---|---|
| Hold HTTP response over five host polls | One operation, one unchanged deadline, one admitted HTTP request | Same |
| Release response after the polls | One UP action delivered, then WAIT | Same |
| Complete before host attaches its receiver | The original decision is retained and delivered once | Same |
| Process queued state after WAIT, before any successor state | Second HTTP request admitted for already-submitted turn 0 | Notification skipped; no second HTTP request |
| Supply legitimate successor turn | Not needed to reproduce the baseline defect | One new request and one action for turn 1 |

The first notification was the startup `user_messages` kickoff. The tailer also
queued state 0 on `game_states` while that first generation was running. After
the first action and WAIT, the original dispatcher consumed this queued state
and invoked `handle` again. The second HTTP admission was observed with exactly
one state line and one submitted action line, before any successor was supplied.

This is redundant generation after a successful decision. It is not a dropped
provider result, an SDK retry, or demonstrated SGLang orphan work. The baseline's
second response was held until trial cleanup; no second model-produced action
was executed in this control.

## Candidate and verification

`candidate/headless.py` adds one condition before invoking `handle`: when a
notification contains only `game_states` and the latest turn already has a
submitted action batch, consume the notification without another generation.
Explicit user notifications, including mixed state/user notifications, retain
their existing behavior. The next unsubmitted state remains eligible.

This is an isolated candidate based on the notebook's original headless source.
It has not been copied into `deployment/kaggle`, repackaged, or promoted. The
condition is not an action-authority guard and makes no supersession guarantee.
It specifically addresses the state-only wakeup observed here; it is not a
general deduplicator for every event or startup ordering.

Real-worker evidence:

- `worker-trial-e6_fs4pa/repeated-host-polls/result.json`: original worker,
  retained request/answer plus observed redundant post-WAIT HTTP admission.
  Its legacy outcome string reports retention passing; the extra admission is
  recorded explicitly in its events and total count.
- `worker-trial-x5vcdg1w/baseline-completion-gap/result.json`: original worker,
  completion before receiver attachment, followed by the redundant request.
- `worker-trial-ro9prz8v/candidate-repeated-polls/result.json`: candidate;
  stale notification skipped, then a legitimate successor served. Two requests
  and two action batches across turns 0 and 1.
- `worker-trial-ro9prz8v/candidate-completion-gap/result.json`: same candidate
  result with receiver attachment delayed until after completion.
- `candidate-controls.json`: execution of the candidate dispatcher with fake
  notifications confirms explicit user and mixed notifications are handled,
  stale state-only notification is skipped, and successor state is handled.
  This small branch control is additional evidence, not a real-worker replay.

The process trials use the actual worker, actual headless dispatcher, actual
MdArcSolverAgent/CodeAct tool execution, and the verified NOOA wheel. Only the
HTTP endpoint supplies scripted model answers. Source-line observers record
host polls, task/frame identity, action entry and WAIT return without replacing
those operations. Explicit HTTP admission/release barriers choose the ordering.
Ten-millisecond observer polls await those events; sleeps do not select the race.
Each worker has an independent 65-second watchdog process targeting only its
newly created process group. Trials close their workers and HTTP endpoints.

Source identity is retained beside each trial. Worker/headless/chassis files are
extracted as data from the local reasoning notebook's embedded ZIP, never by
executing notebook cells. The original worker hash is
`1a2e7c39a58b3bba38a8bfc90be4adbe5f80faffa7a5635d4324606e82e8eb7d`;
headless hash is
`169791aa6b162dca9166179bb88043dbfbeb5025838bfcb8081acbb61f1c84f3`.
The current loose worker file differs: it already publishes state in `observe`.
That independent change was not substituted into the baseline.

The NOOA wheel is loaded directly from the previously hash-verified v11 artifact.
LiteLLM uses cached 1.97.0; other dependencies come from the local deployment venv.
The protocol classes come from the repository's `solvers/tycho_nooa/src`.
Thus this is a local process/path qualification, not a full reconstruction of the
remote environment. Synthetic grid, endpoint/model alias and a 40-second worker
watchdog allowance are probe controls; neither timeout fired in successful cases.
No claim is made about numerical inference or timing under production load.

The first two setup attempts failed before any HTTP admission: the driver used
an obsolete observation field, then `previous_result` instead of the protocol's
`previous_decision`. Those attempts remain under `worker-trial-7g9njg8y` and
`worker-trial-e1bfpy_n`; neither is behavioral evidence. The driver now constructs
observations through the real protocol projector. Final scripts pass Ruff.

Run from the repository root:

```bash
deployment/kaggle/.venv/bin/python analysis/deployment-probes/20260905-request-ownership/worker_probe.py
python3 analysis/deployment-probes/20260905-request-ownership/candidate_controls.py
```

## Claim boundary and package promotion

The selected dispatcher is now packaged separately under
`deployment/kaggle/s0-dispatcher-admission-candidate/`. Its `qualification.json`
records successful replay from the delivered notebook's embedded ZIP, including
executing worker/dispatcher source hashes and compiled-code identity. Repeated
polls, receive-gap completion with an identical-grid successor, and a legitimate
two-call live decision all pass. Only `headless.py` changes inside the archive;
engine, manifest and reasoning cells remain unchanged. This closes the local
package gate, not the remaining integrated lifecycle gates.

Stop investigating ordinary host polling as a mechanism that cancels the model
consumer on this path. The successful answer was retained. The reproduced
extra work comes from the next stale notification, and the isolated candidate
passes that specific before/after test.

In-flight supersession and interrupted shutdown are still unqualified. The
worker processes protocol messages serially, so `finish`/`close` cannot simply
be assumed to revoke an in-progress `decide` at send time. The next local test
must use the actual outer LocalSolver termination path and distinguish sending
a stop command from its effective boundary. This probe used a protocol driver,
not the complete MyAgent/LocalSolver parent. It does not test an obsolete action
racing with successor authority.

Target-side retirement and integrated performance remain unqualified. The
independent isolated FP8/MTP probe is tracked in
`analysis/deployment-probes/20260905-isolated-engine-v1/`; it does not depend on
repeating this local component investigation.
