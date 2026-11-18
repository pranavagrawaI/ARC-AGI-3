# Contingent controller contrast

Question: With routine reasoning off, does the bounded contingent interface change useful closed-loop investigation against the same solver without it?

Contrast: `off` = the interface is absent from the tool surface, the prompt, the decision packet and the dispatcher; `on` = it is available. Same FP8 checkpoint, sampling, tools, deadlines, action gates, 16,384-token ceiling, discovery arm A, optional world model, and no MTP.

Measure: valid actions and transitions; model calls, prompt/cache/completion tokens, model/tool wall time; contingent proposals, selected branches and stop reasons; levels; helper/procedure artifacts; exact request and lifecycle reconciliation. TTFT remains explicitly unavailable on non-streaming requests.

Cap: bounded public ls20 only, twenty environment actions per arm, 3,000 seconds of admission per arm under a 3,600-second launcher timeout, two arms, no submission or automatic promotion. The time budget is set from v1's measured 104.6 s/action so that the action budget, not the clock, is what stops an arm.

Stop: any source, factor, reasoning, lifecycle, accounting, timeout, or containment failure stops before the next arm.

Invalid if: arm identity, reasoning-off state, request provenance, or terminal ownership cannot be established. `test_contingent_arm.py` establishes arm identity locally against the real worker; the run must still show it in the live request record.

Result map: advance a repeatably better arm to a larger matched target run; choose the simpler arm when practically tied; retain `off` on no benefit; report a run where no plan was ever proposed as uninformative, not as a null; diagnose technical failures without silently relaunching.

Outputs: packaged notebook, exact source/manifest, raw IPC/server evidence, assessments, and post-run disposition.
