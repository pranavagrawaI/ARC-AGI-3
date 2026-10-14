# Planning and prospective rule checks

Intent: improve conversion of learned rules into completed play, without training or changing the incumbent.

First resolve whether the generic helper retains the earlier native-valid recovery and whether other frozen learned models return misleading routes. Reuse the existing m2_search implementation through a frozen sparse overlay; never edit its active workspace. Test all six saved B1 continuation models, using their actual final states. They share one game and selected historical contexts: this is a retrospective diagnostic, not a held-out-game generalization test. Use the same action order, state key, goal and model for zero-heuristic versus guided search. Record a 20-second cooperative search limit, 300000 expansions, 500000 stored records, depth 40 and a 30-second external process deadline per arm. Engine code may only restore prefixes and validate returned routes. No reference route or reference simulator enters search.

Then implement prediction-before-action as one explicit sandbox operation. The agent supplies observation, prediction and state-key callbacks. Persist the prediction through the existing host-owned saved-data transport before one real action. Preserve mismatches, refusals and interrupted/unknown outcomes. No callback may execute an action during observation, prediction or comparison. A match covers only the agent's declared state representation.

Implement four switchable variants: control, planner, probing, both. Qualify actual ToolAgent/sandbox behavior with local recorded or fixture replies, fixed budgets, no external inference. Keep capability exposure and instructions aligned. Freeze an executable continuation runner and paired receipt audit; do not infer adoption from analyst-written callbacks.

Only if local evidence survives should a bounded inference comparison test adoption and completion. GPU execution needs a separate immutable card with account admission, source hashes, total token/time ceilings and no automatic retry. This implementation does not submit or promote candidates.

# Pre-mortem

Failure scene: the helper passes toy tests and a selected route, but spends the live budget planning through wrong rules and reduces completions.

- Observed simulator errors -> model-valid plans fail native dynamics -> false success. Validate every returned route stepwise, stop at first mismatch, report FOUND_IN_MODEL separately from native completion.
- Selected s5i5 recovery -> tuned adapter/heuristic -> apparent generality. Freeze six-model census before results; label shared game/selected contexts explicitly. Different-game adoption remains untested.
- Callback takes actions or hangs -> speculative mutation or unbounded compute. Sandbox guards prevent actions during callbacks; enclosing subprocess deadline kills hangs. Search limits are cooperative and remain distinct from graph exhaustion.
- Probe records after acting or callback mutates prediction -> retrofitted evidence. Deep-copy prediction and observation, commit prediction through host IPC before action, and exercise timeout/order behavior in the real subprocess.
- More probes -> fewer productive actions -> worse score. Live comparisons must share total token/time budgets and score completions, not only agreement counts. CPU fixtures establish wiring only.
- Baseline reconstruction changes prompt/helper visibility -> invalid paired comparison. Restore the original archived source first; apply only named variant changes at branch time. Record both restoration and intervention identities.
- Concurrent untracked work -> accidental edits/commit ownership. Work in new isolated directories, record borrowed source hashes, verify the parent, stage an explicit task allowlist.

Residual risks accepted for this local stage: analyst adapters, one game, selected checkpoints, no evidence of model adoption. These prohibit promotion but do not prevent interface/transport qualification.
