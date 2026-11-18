# MTP stochastic native-renormalization qualification

Question: Does the pinned native renormalization route enable the actual stochastic
MTP workload offline, and what paired Actor latency/output behavior follows?
Posture: repair qualification followed by matched-workload measurement.
Contrast: baseline auto FP8 versus shallow EAGLE1/topk1/drafttokens2. Same sealed14
inputs and model generation defaults: temperature1, top-k20, top-p0.95. Native
renormalization overlay installed in both arms; ordinary sampler remains PyTorch.
No compiler, wheel, checkpoint, reasoning or acceptance-setting substitution.
Prediction: existing precompiled kernels avoid the failed FlashInfer sampling JIT.
The synthetic1.59x speedup is not a prediction for this corpus.

Measure: preflight sampling-patch.json and sampling.json bind the source overlay,
versions, numerical errors and real GPU. Require PASS with JIT disabled. Retain
stochastic-preflight-request/response per arm; MTP spec_verify_ct must be positive.
Retain per-case request/response, latency, generated-token counts, tool errors,
actual server prompt IDs, generation config, server logs and lifecycle. All14
cases remain in results, including invalid outputs. Fixed1024-token controls remain
separate native diagnostics without chat grammar. No gameplay/tool execution.

Cap: one user-launched private RTX PRO6000 allocation <=7800s (2.17GPU-hours).
Kernel preflight600s, replay runner6400s/outer6600s. At most34 explicit model calls
(28natural,2excluded warmups,4fixed controls), plus engine startup warmups.
Natural output limits remain omitted; overall and900s per-request deadlines bound
cost. Zero retries, external model calls, actions or submissions. No assistant push.

Stop: wrong source/version/hardware/model identity, numerical kernel failure,
missing stochastic MTP evidence, engine exit, HTTP/deadline failure, prompt drift
or failed containment. Preserve original failure separately from cleanup errors.
Do not start a replacement server after unresolved cleanup. Invalid tool outputs
are recorded outcomes, not reasons to abort before the other arm is measured.

Invalid if: sampling parameters were weakened, native route not selected, MTP not
exercised, prompt identities differ, or accounting/timing/lifecycle evidence missing.
Local kernel validation does not establish full EAGLE or rejected hybrid-state
correctness, equivalence to FlashInfer in every numerical edge case, or gameplay.

Result map: kernel and MTP path pass, acceptable reviewed outputs and >10% paired
natural-latency improvement -> select for integrated qualification. Runtime/quality
failure -> retain baseline, inspect evidence, no automatic relaunch. Mixed/uncertain
latency -> retain baseline pending interpretation. Fixed arm order and one sample
per case limit claims; no selective removal of invalid or longer outputs.

Outputs: /kaggle/working/mtp-sampling-preflight and
/kaggle/working/mtp-sampling-repair-20260906. manifest.json binds notebook, corpus
and overlay installer. Original failed notebooks and their cards remain unchanged.
