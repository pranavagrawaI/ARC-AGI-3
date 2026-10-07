# MTP-depth serving sweep: experiment card, frozen 2026-10-06

Question: on the dfranzen M2 incumbent server (Qwen3.8-Flash-Next Intel W4A16, albucino MTP drafter, Pennyroyal SGLang d00d88e, one RTX PRO 6000), with production client admission S=10 and realistic mid-game traffic, which speculative depth M (`--speculative-num-steps`, `num_draft_tokens = M+1`) maximizes end-to-end emitted completion tok/s U?

Origin: pre-mortem of the expert-routing proposal (board 504 #539) and a five-round blind grill with Pranav, 2026-10-05/06. Expert-routing telemetry, server cap > 10, Attack A and routing replay are deferred to a later tree (see "Deferred").

## Contrast

Five launches in one Kaggle session, in this order: M=3 (A) -> M=1 -> M=2 -> M=4 -> M=3 (B). Every launch uses the incumbent server flags unchanged except `speculative_num_steps` / `speculative_num_draft_tokens`. In particular `max_running_requests=10`, `cuda_graph_max_bs_decode=10`, fp8_e4m3 KV, `mem_fraction_static=0.96`, chunked prefill 8192, Mamba cache 60, `speculative_eagle_topk=1`, accept thresholds 1.0, `speculative_token_map=hot_tokens_64k.pt`.

## Workload

- Closed-loop replay of the 2026-10-01 control run (`analysis/m2-winners-20261001/runs/out-control/*_requests.jsonl`).
- Start point: each game's first request whose reconstructed send time is at run-minute >= 40. Amended 2026-10-06: on the run clock, all 25 games are still live at minute 40. The earlier '~21' came from per-game clocks, but games were admitted 0-30 min after run start. Games that run out of requests during replay simply finish. At the checkpoint, 10 hold slots and 15 wait, in production's order.
- Admission copies dfranzen's `_PriorityGate`: S=10 slots. A game keeps its slot across tool gaps and hands it over only where the logged prefix breaks (a trim). Waiting games are served FIFO.
- Free generation from each recorded prefix, with `max_new_tokens` = recorded completion length. A natural EOS ends the request early. Early-stop rate is reported per M.
- The tool gap after each response is gap-shape x alpha. The shape comes from transcript turn intervals minus modelled server time. alpha is one scalar, fitted offline by discrete-event simulation on the serve.log curves and frozen before any GPU time.

## Measurement

- Each launch has a warm-up excluded by a state criterion: every admitted game has completed at least one replayed request, and occupancy has settled.
- One 4-minute measured block per launch, split into 4 x 60 s sub-windows. Scored on achieved occupancy.
- Reported per launch: U, accept length, early-stop rate, mean running R, P(R>=9), cached-token fraction (diagnostic against production).
- Optional graphs-on `nsys --cuda-graph-trace=node` profile. Amended 2026-10-06: it runs as a sixth launch (PROFILE_M3) after M3B, not inside launch 1. The server must be started under `nsys launch` for nsys to attach, and that would make 3A differ from 3B and contaminate the drift D. The 30 h quota makes the extra ~20 min affordable. If nsys is absent, the profile is deferred, with no substitute profiler. It is the first thing cut.

## Validity gate (launch 1, M=3)

- Frozen intent: replay occupancy reproduces production.
- Amended 2026-10-06 after measurement: production itself failed the originally stated band (mean R in [9.0, 10.0] and P(R>=9) >= 0.80). Over run-minutes 30-60, production's serve.log gives mean R 8.97 and P(R>=9) 0.72.
- Gate: mean R within 8.97 +/- 0.40 and P(R>=9) within 0.72 +/- 0.10.
- A miss marks the workload invalid. No refit of alpha on GPU.

## Decision rule (pre-registered)

For candidate m:
- Delta_A = U_m / U_3A - 1 and Delta_B = U_m / U_3B - 1.
- D = |U_3B / U_3A - 1|, the launch drift.
- W = the largest coefficient of variation across the 60 s sub-windows of the candidate, 3A and 3B blocks.

Clean win: min(Delta_A, Delta_B) > D + W. If a candidate fails only because D is large, the result is unresolved, not rejected; launch replicates can be bought later. This is an empirical envelope, not a confidence interval.

The Q7 audit (Q7-AUDIT.md) classifies M in {1..4} as distribution-preserving. So a winner is adopted after one short live harness qualification with no scored arm. Operational qualification means: no crashes, no cache pathology, unchanged context behaviour, and early-stop rate not systematically different by M.

## Operations

- M=1 or M=4 failing to launch: one attempt, logged, then the next launch. No retry. Baseline B is protected.
- Cut order if late: profile -> shorten non-baseline blocks -> never cut M=1 or 3B.
- Hard session cap set from the live quota read before the push.
- Nothing launches while another experiment holds both GPU slots (wave-2 holds them on 2026-10-06). An experiment whose trigger fires (e.g. a WIN that unlocks a conditional pair) goes first.

## Pre-launch gate (all must pass)

1. Mock-server end-to-end replay. It must show: ~21 workers, checkpoint selection, closed loop, applied gaps, slot hold and handover, no cancelled in-flight requests, usage and occupancy recorded, clean stage termination, partial results surviving failure.
2. A timed dry run of the five-launch orchestration state machine.
3. Q7 code audit: done, see Q7-AUDIT.md.
4. alpha frozen.
5. Request-log dataset hash: local = uploaded private dataset = Kaggle-session readback.

## Deferred (next tree, not this card)

- Expert-path kill rule: routed experts >= 50% of the step.
- Server cap > 10, and with it Attack A (affinity batching).
- Routing replay.
- Streams adoption rule at a matched 6,000 s deadline vs the control's 100 levels: >= 119 adopt; 90-118 adopt on mechanism; 83-89 no; <= 82 reject.
- A speculation-off empirical sanity test.
- A graphs-off attribution profile.
- More launch replicates for small effects.

## Implementation evidence (2026-10-06)

- Prompt fidelity: logged messages with `_arc3_control` stripped, rendered through the model's own chat template, match the recorded text tokens at a median ratio of 0.9998 over 200 requests (tokcheck.py).
- alpha = 0.523 (alpha.json). The server rates are time-weighted from serve.log, and token conservation was checked. One scalar matches both targets: mean R 8.97 and P(R>=9) 0.73.
- Predicted replay cache fraction without splicing generated outputs: 0.923, against 0.935 in production. No splice.
- Mock rehearsals, all in /tmp and reproducible with sweep.py --mock:
  - null case: KEEP M=3, D 0.4%, W 1.2%;
  - planted +10% on M2: CLEAN WIN, and planted -8% on M4: CLEAN LOSS;
  - M4 launch failure: recorded, no retry;
  - injected request errors: handled;
  - short budget: cut order applied.
- Notebook dry run (dryrun_notebook.py): readback hashes match, flags are substituted correctly for all five launches, 0 errors.
- Dataset upload readback: 36/36 files match (evidence/dataset-readback.json).
- Launch: systemd unit `mtp-depth-launch-watch`. Log: evidence/launch-watch.log.

## Budget note

The design was cut to fit an assumed ~1.65 GPU-h. The Kaggle UI shows a 30 h weekly cap, and the API's 21,600 s figure is wrong (see memory). Budget no longer forces the cuts. The frozen design stands; adding replicates is a separate decision.
