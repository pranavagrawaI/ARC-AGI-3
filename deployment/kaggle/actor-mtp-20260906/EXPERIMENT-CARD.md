# Actor replay — sealed capture, user launch only

Question: Does shallow native MTP improve latency on these actual early Actor
requests without worsening tool output behavior?

Contrast: baseline auto FP8 versus native EAGLE/MTP, steps1/topk1/drafttokens2.
Same checkpoint, wheel, attention, sampler, generation defaults, complete messages
and tool declarations. No host changes, retries, tools or gameplay.

Evidence: all 14 request/response hashes reconcile with capture events. Keep three
original unknown-tool outcomes labelled. Coverage: 10.6k–21.3k prompt tokens,
early/tool-continuation, auto tool choice, thinking disabled. No forced-action,
~102k compression or long-history claims.

Measures: paired natural latency, output counts, tool/schema errors, finish reasons,
raw responses and diagnostic token IDs. Separate fixed native 1024-token controls,
twice per arm, measure decode without grammar. Cold cache before measured calls.
Record server settings, speculation and lifecycle. Acceptance length includes
target/bonus tokens; accept_length/2 is not drafted-token acceptance rate.

Cap: one private RTX PRO 6000 allocation <=7200s (2 GPU-hours), runner6400s,
outer runner6600s. Two server starts, 34 explicit requests: 28 natural, two64-token
warmups, four1024-token controls. Engine startup warmup additional. Natural bodies
omit output limits: no invented aggregate token cap. Per-call timeout900s and
overall deadline bound cost. Zero automatic retries, game actions, submissions
or paid external calls. At most one user push; no assistant launch.

Stop: hardware/corpus/default/manifest drift, HTTP failure, missing prompt IDs,
server failure, deadline or failed teardown. Preserve partial outputs; no new
server after unresolved cleanup. Invalid tool responses are measured failures,
not grounds to stop the arm. A completed run can still fail quality review.

Selection: >10% paired natural-latency improvement with acceptable reviewed outputs
selects MTP for further qualification, not deployment. Report all cases and validity
breakdown; do not select only successful or shorter outputs. Mixed results retain
baseline pending interpretation. One sample per request and baseline-first order
limit precision. This is one failed decision, not independent games.

Artifacts: /kaggle/working/actor-mtp-20260906; manifest binds notebook, runner and
sealed corpus. Startup differences and text divergences are not automatically
benign. Hybrid rejected-state and closed-loop correctness remain separate gates.
