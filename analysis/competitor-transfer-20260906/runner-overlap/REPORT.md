# Pinned runner overlap: direct CPU-only reproduction

The packaged runner permits overlapping model HTTP requests from independent game sessions despite its C1 semaphore. This is an observation from real packaged worker/headless + actual NOOA, not inference from the Swarm design. It invalidates the stronger interpretation of the previous fake-worker `CONFIRMED_SERIALIZATION` result.

## Mechanism

All relative source paths below are under `deployment/kaggle/q6-summary-boundary-20260906/`.

- `shared_local_solver.py:258-260`: admission wraps sending a turn and receiving its decision only.
- `shared_local_solver.py:281-282`: observe sends its result and receives acknowledgement outside admission.
- `nooa_worker.py:220-235`: on batch exhaustion, observe increments the NOOA turn and publishes the successor immediately.
- `headless.py:49-68`: state notification triggers actual `agent.handle(notification)` autonomously.
- `headless.py:107`: each headless process builds the configured shared-endpoint LLM client.
- `chassis/arc_llm.py:106-116`: CompletionClient uses the forwarded base URL/model.
- `nooa_worker.py:126-178`: start launches the headless process but does NOT publish the initial observation. First decide writes the initial state at lines 199-203. Thus the demonstrated escape is successor publication, not initial startup.
- Downloaded official `ARC-AGI-3-Agents/agents/swarm.py:77-104` constructs independent game agents and starts one thread per agent. Its official game execution was not invoked by this probe.

## Experiment

`probe.py` loads the packaged LocalSolver without modifying its semaphore. It starts two real packaged workers, which launch real packaged headless actors and the installed NOOA CodeAct runtime. Two distinct synthetic public observations enter separate sessions. All requests target the same loopback HTTP endpoint and model `shared_actor_probe`; an inert `probe_owner` field identifies ownership. The scripted server returns valid `execute_python` tool calls invoking actual `self.submit_actions`.

1. A.decide returns its first action.
2. A.observe receives a distinct successor grid; A sends its second completion request without a second A.decide call.
3. The HTTP server holds A's second request until B's first request arrives.
4. B.decide enters the semaphore and sends its request while A is still in flight.
5. Both HTTP requests complete; B returns its action, and separate IPC action files contain A's turns 0/1 and B's turn 0.

`RESULT.json` records peak HTTP in-flight = 2 and no second A.decide. `events.json` records barrier order. `request-actor_a-2.json` and `request-actor_b-1.json` preserve exact rendered requests. The final result identifies the corresponding session directories; earlier exploratory sessions are also retained. Any requests during process shutdown are outside the assertion window; reported counts are snapshotted at the assertion.

Run locally:

```bash
/home/pranav/projects/arc-agi-3-reference/nooa/.venv/bin/python analysis/competitor-transfer-20260906/runner-overlap/probe.py
```

The installed NOOA checkout HEAD during the experiment is `2d3f6ebc4d12d48da5b82a21bc4fadd16f3bc32c`; this uses the existing local runtime, not a recreated Kaggle container. Worker startup verifies every packaged executable file against CHASSIS.json. Shared LocalSolver SHA256 is recorded in RESULT.json and matches the package manifest.

## What this does and does not establish

It establishes actual concurrent completion HTTP requests can reach one serving endpoint from the packaged NOOA join, preserving distinct session state in this probe. Therefore batching feasibility cannot be dismissed because a semaphore is present, and C1 cannot be assumed for measurement baselines.

It does not measure SGLang admission, actual GPU batch size, throughput, latency, cache behavior, long contexts, deadline correctness, or the official scored runner end to end. Scripted responses are compatibility/control-plane evidence, not policy quality. No GPU or external API was used. No deployment source was changed.

Next experiment should instrument actual request-level in-flight counts and server running-batch behavior. A genuine C1 control requires request-boundary admission (or an isolated single-session benchmark), not reliance on LocalSolver.decide's lock.
