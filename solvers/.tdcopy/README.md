# Tycho/Nooa on dfranzen's runtime

A local experimental notebook has been built from the exact `dfranzen-m2-control.ipynb` in `analysis/m2-winners-20261001/runs/control`. It keeps the donor's model/serving setup and replaces the benchmark solver with canonical Tycho/Nooa through a protocol-v3 adapter. The canonical solver and donor source files are unchanged; candidate-only changes are generated in `runtime/` and recorded in `canonical-candidate.patch`.

This is an executable integration candidate, not a demonstrated improvement. Twenty local tests passed before q5, and its CPU setup rehearsal passed. GPU q5 then failed before gameplay because the runner's game-name fallback accessed a label absent from real GameAPI objects. See `experiment/gpu-q5/RESULT.md`. The fallback is fixed, the small-set regression now uses real GameAPI objects, and a new runner regression exercises actual offline game actions. The distinct private source-v4 bundle is uploaded. CPU q6 passed the packaged real-GameAPI rehearsal and receipt audit; GPU q6 completed under its separate frozen one-hour card: mean score 30.23, 437 actions, no game crashes. Both admitted Builder attempts timed out without an accepted model. Its host deadline-accounting defect is now repaired in the editable candidate; 30 local tests and packaging verification pass. See `experiment/gpu-q6/{RESULT.md,BEHAVIOR.md}` and `evidence/deadline-accounting/REVIEW.md`. Frozen q4/q5 evidence remains unchanged. See `experiment/REVIEW.md` for reproduced defects/fixes and `experiment/PRE-MORTEM.md` for experiment risks. An earlier test accidentally reached an inherited live endpoint; the preserved incident is described below.

## Component decisions

| Component | Owner and decision | Reason |
| --- | --- | --- |
| Weights, quantization, drafter, Pennyroyal/SGLang setup, server flags, monitoring | Keep the donor notebook cells verbatim | Preserve its engineering baseline before changing reasoning |
| Games, deadlines, benchmark accounting, diagnostics | TAAF remains the host | One authoritative engine action and receipt per acknowledged Tycho decision |
| Actor, Actor-controlled Builder, Scribe, world model and goal gates | Canonical Tycho/Nooa, `orchestrator/control` | Preserve role ownership, evidence discipline and guarded solving |
| Action protocol, legal actions, CLICK coordinates, batches, successor continuity | Canonical protocol-v3 worker and transport checks | Keep action commitments grounded in host transitions |
| Python execution and isolation | Tycho gated executor and Kaggle hardened host | Retain workspace semantics and credential separation; fix the uv interpreter mount in the candidate snapshot |
| Useful donor computation | Preload `dfr.segment(grid)` in native Python; install NumPy/SciPy | Reuse segmentation, adjacency, containment, shape hashes and rotation signatures without adopting Duck's solving policy |
| Reusable model-written functions | Keep workspace files and imports across calls | Whole Python modules can be retained; Duck's selective function extraction is unnecessary here |
| Board presentation | Canonical numeric grid plus donor 10x PNG, frame diff and terminal image helpers | Preserve exact state access while adding the donor's visual input |
| Context retention | Candidate provider projection using donor estimators; 128 Ki context, 12 Ki output, 58 Ki block drain | Avoid rewriting old rendered turns every few actions; count images and tool schemas in the input estimate |
| Game admission | Donor priority gate, holding game slots across calls and yielding after an observed context eviction | Reuse its scheduling machinery; count Actor, Builder and Scribe tokens |
| Sampling and thinking | Donor temperature .7, top-p .95, top-k 20; thinking enabled; replay `reasoning_content` | Match the donor's request conventions through NOOA |
| Request evidence | Canonical exact-request checkpoints plus projection log and host action receipts | Preserve what the provider saw and what the engine actually did |
| Duck automatic death resets, no-op/death refusal, section-memory parser | Leave out of this candidate | These alter the solving policy; importing them would make a reasoning comparison harder to interpret |
| Duck Python/action tool and solver loop | Replace with Tycho's native tools and worker | Give each responsibility one owner |
| Warmup RESET and competition path | Offline qualification only; no warmup action | The candidate waits for initial server readiness within its deadline and rejects competition reruns |
| Duck viewer | Replace with pointers to hybrid artifacts | Its expected trace schema differs from this worker |

The path is `donor notebook -> isolated runner -> TAAF game -> protocol-v3 worker -> Tycho Actor/Builder/Scribe -> native tools -> host action -> canonical observation acknowledgement`.

The adapter adds process transport and role accounting to the donor's orchestration. It does not simplify Tycho's reasoning architecture. The potentially expensive interaction is Builder/Scribe calls disrupting useful cache reuse: longer retained history alone does not guarantee faster inference. Fresh Builder calls, role changes, template formatting, and projection-anchor resets can change prefixes. The donor scheduler's pace reference is calibrated on Duck, not Tycho, and backend request priority hints are not currently forwarded. These are measured uncertainties for GPU qualification, not established gains.

## Artifacts and reproduction

- `kernel/tycho-dfranzen.ipynb`: generated Kaggle notebook.
- `kernel/kernel-metadata.json`: private candidate metadata retaining donor hardware, container, model and dataset references.
- `src/hybrid/`: editable adapter source. `runtime/` is generated output; edit source and rebuild.
- `manifest.json`: exact parent notebook, source provenance, dependency manifest and payload hashes.
- `source-bundle/`: private hash-bound runtime payload, loaded by the notebook to stay below Kaggle's 1 MB source limit.
- `canonical-candidate.patch`: all candidate changes to canonical modules.
- `runtime-deps/`: three additional wheels and their hash manifest, with private dataset metadata. Upload status is recorded in `experiment/dependency-upload.log`.
- `evidence/`: qualification summaries and preserved raw request/tool/action/workspace evidence under `fixtures/`.

From the repository root, using the already prepared canonical environment:

```bash
python3 solvers/tycho_dfranzen/build.py
solvers/tycho_nooa/.venv/bin/python solvers/tycho_dfranzen/verify.py
solvers/tycho_nooa/.venv/bin/python solvers/tycho_dfranzen/tests/qualify.py
```

The local environment was created with the canonical frozen `uv.lock` on Python 3.12.12. Tests additionally need `nbformat`, `imageio==2.37.3`, `imageio-ffmpeg==0.6.0`, and `scipy==1.18.1`; `nbformat` is verification-only. The extra dependency wheels can be reproduced using `python -m pip download --only-binary=:all: --no-deps` with those three exact runtime versions. `runtime-deps/hybrid-deps-manifest.json` binds their actual bytes.

Kaggle needs the exact `pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11` dataset and the prepared extra-dependencies and `pranavagrawai/tycho-dfranzen-source-v3` datasets attached in addition to the original donor inputs. The installation cell verifies every referenced wheel and extracts the hash-bound source payload, creates an isolated venv, and matches the competition's installed ARC package versions from its bundled wheels. The local v11 directory contains manifests but no wheel files, so the Kaggle offline installation has not been executed locally. Verification found no installed-package version differences against its pinned v11 dependency manifest.

`HYBRID_RUN_MODE` defaults to `smoke`: exactly sb26, ls20 and ar25, one pass, 16 actions and 600 seconds per game. `small` uses the same three games, one pass, 2400 seconds/game and no action-count cap. `public25` retains donor game list/budgets and is outside the current experiment plan. Notebook metadata enables a GPU: running setup cells starts the donor server. The local reproduction commands do not execute those cells.

## What was checked

1. TAAF runs the real worker and Python sandbox, preserves a saved helper across turns, executes two expected actions, acknowledges terminal/level evidence, and runs the Scribe.
2. The native Python tool computes the expected segmentation of a small independent grid and imports SciPy.
3. A loopback OpenAI-compatible server exercises orchestrator/control, executes two Actor-requested Builder passes with a wrong prospective prediction followed by repair and a correct prospective prediction, records Actor/Builder/Scribe calls, verifies output/sampling/thinking settings, images, initial readiness, reasoning replay and cached-token accounting.
4. Provider projection keeps whole user-led exchanges and tool pairs, retains its selected head on the next request, and bounds input estimates. Image accounting uses dimensions instead of base64 string length.
5. Cancellation during a blocked worker call reaps that process and marks both the active and queued games cancelled before returning.
6. The generated runner unpickles its benchmark/target, runs end to end in a separate process and preserves a won fixture result.

7. A worker error preserves completed token accounting and causes zero engine actions.

8. Slow successful and HTTP-error bodies cannot trickle past the host deadline; the owned worker is reaped, no engine action occurs, and unresolved usage is not treated as measured zero.

These checks are eight test methods; the first two items share an end-to-end test and the projection/image checks are separate methods. Notebook schema, source-size limit, payload hashes, untouched donor cells, source provenance, hardware/model references and extra wheel hashes are verified separately. Fixture wins are transport qualification, not ARC capability evidence. GPU startup, offline Kaggle dependency resolution, hardened-host isolation on Kaggle, real multi-game throughput, prefix-cache efficacy and ARC scores remain unqualified. Token totals include completed responses from all three roles, including totals reported on worker errors; interrupted calls without a completed response can consume unreported server work.

The next discriminating measurement is this candidate versus the unchanged donor control on the same small game set and GPU/time/token budget, retaining per-role tokens, action latency, cache tokens, projection resets, evictions, levels and complete receipts. A pass through the three-game smoke establishes runtime compatibility; it does not select a winner. Only a matched comparison can establish whether Tycho's extra reasoning machinery earns its cost.

## Qualification incident

The first test reused a transport whitelist that omitted the scripted-model selector. Inherited endpoint configuration reached a live `openai/qwen/qwen3.8-27b` model: four completed requests reported a total cost of $0.0081258972. The exact test and worker processes were stopped. The candidate now forwards the selector, and qualification removes inherited model endpoint configuration and credentials before importing the solver. Subsequent tests use scripted or loopback-only responses. `evidence/qualification-incident.json` and the preserved raw incident directory record what happened; no GPU/Kaggle run was launched.

See `NOTICE.md` and `LICENSE-APACHE-2.0` for source attribution.

## v2 qualification

The original candidate is preserved in `revisions/local-v1`. The current candidate uses Actor-controlled Builder calls to match the policy associated with the cited Tycho result. A stronger test exposed that a winning fixture could hide rejected Builder calls: its 60-second deadline left no time after the canonical 300-second Actor reserve. The fixture now grants the intended decision budget and verifies executable prediction and repair before host actions. It is scripted execution evidence, not learned reasoning evidence.

Slow HTTP success/error tests exposed a ten-second cleanup grace after request expiry. The candidate now bounds both the pipe close and process close grace to one second. Failure logs are retained. Execution receipts record elapsed time, worker exit, and possible unresolved provider usage. See `experiment/PLAN.md` for the confirmed objective and ten-GPU-hour campaign budget.

Kaggle rejected the embedded payload because its notebook source exceeded 1 MB. The loader now reads the exact compressed payload from a private dataset and verifies its SHA256 before extracting any files. All runtime source hashes remained identical across this packaging change. Attempts q1–q3 and their rejections are preserved; q4 was accepted, with compatibility and scores still pending.
