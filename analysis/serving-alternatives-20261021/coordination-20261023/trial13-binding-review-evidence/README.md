# Independent trial13 binding/reducer witness — CPU archives only

Candidate freeze: `5ac295607eae8c7d0e71a42e023612c5e19ac88255f18d927fe39a87863da2ad`. Reducer SHA: `9f6795dc0cfc197199ed1c906294015c02c8ad3779ea557e8d7dae3590c375ee`. No candidate files changed. No network, SDK, launcher, model, or GPU execution.

The baseline copies the real captured `qualified-r4/cases/cpu-two-loads` lifecycle. It uses an explicitly synthetic accepted-identity envelope and the actual unassigned source notebook; these are reducer fixtures, not launch authority. CPU lifecycle skips GPU compatibility, so `query-compatibility.json` explicitly wraps its captured CPU pre-release GPU/apps command records. Original raw artifacts remain hash checked. The envelope freezes references to every original 487 local and 251 external input binding.

`SUMMARY.json` preserves the initial incomplete compatibility fixture, which correctly failed. `COMPLETE-FIXTURE-SUMMARY.json` records the completed baseline and mutations. Baseline and every mutated archive qualify without issues under the original reducer.

- `release_false`: only release.json pass becomes false and error identifies the injected failure. Reducer emits qualified status alongside release_pass=false.
- `fresh_after_original_release_deadline`: release.json absolute_deadline and corresponding release_start/release_result event deadlines move to 1ms before fresh.sample.start. Fresh admission remains otherwise unchanged.
- `boundary_oom`: only last host boundary oom_kill increases by one; observer series unchanged.
- `nonterminal_readback`: readback says RUNNING/reserved6300. This is a collection/slot-closure reporting gap, not a separate demonstrated runtime false pass.

Each case retains archive ZIP, extracted members, CRC/hash envelope, readback, and full derived output. Unified JSON diffs show exact before/after changes. `recheck.py` preserves the executed fixture-completion script. To replay existing archives without changes or APIs:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 analysis/serving-alternatives-20261021/coordination-20261023/trial13-binding-review-evidence/reproduce.py
```

The reproduction creates a new exclusive timestamped replays directory. It imports the unchanged current reducer and redirects only its input-root constant to the explicitly synthetic envelope. It does not patch any validation function. Exact raw-reference verification and all original frozen source/reference checks run.
