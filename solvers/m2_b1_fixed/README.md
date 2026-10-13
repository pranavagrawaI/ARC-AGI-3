# M2 B1 bugfix successor

Local successor of the preserved `../m2_b1`. **CPU and packaged-sandbox qualified; not deployed or gameplay-qualified.** The model, serving, context limits, opening pin, history retention and action policy are inherited. No improvement mechanisms from the Chen survey are added.

Two reproduced defects are repaired:

- Frame instructions claimed only five fields existed and denied raw numeric-grid access while the sandbox already exposed `.grid`, `.rows`, `.counts` and `.cells(color)`. Both system and turn instructions now describe the real API. The regression gate checks the assembled system prompt and exercises these fields in the actual sandbox subprocess.
- B1's eviction counter missed commit-time assistant-turn/token eviction while the pin was active, and missed overflow fallback eviction. Counting now occurs once at each outer boundary; internal commit-time request trimming is not a synthetic request. The system message is excluded from commit comparisons because it is intentionally not persisted. A failed commit releases counter suppression.

Counter schema **2** adds `request_calls`, `commit_calls`, `overflow_calls`, corresponding `_trims` and `_messages_dropped`, and `schema_version`. `requests`/`pinned_requests` count request preparation only. `trims` and `trims_with_pin` now include all three phases; they are not request-only rates and must not be compared directly with archived schema-1 totals. A trim means at least one message was removed; image-only changes are still represented by the existing eviction flags, not these message-removal counters. `fallbacks` retains the original request-pin reassembly meaning.

The old `trims=0` observation is not evidence of no eviction. Historical artifacts and reports are preserved; the existing failure audit already corrected that interpretation.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 /home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12 solvers/m2_b1_fixed/qualify.py
```

This rebuilds `kernel/m2-b1-fixed.ipynb`, runs inherited behavior/receipt/evidence gates, checks exact overlay installation and drift rejection, executes the bugfix regression against the installed package, replays the opening pin on real archived requests, and compares the preserved B1 parent against this successor on 435 request/commit/overflow outcomes. All compared retained-message hashes and cache-eviction flags match. The meaningful regression fails before the repair (`evidence/bugfix-before.log`) and passes afterward. Protected parent file hashes are verified at the end.

The parent comparison uses 108 synthetic fixtures across pin sizes, token budgets, turn caps, drain settings and level boundaries, plus 37 archived B1 requests, each through all three entry points. It establishes retained-message behavior for those cases, not gameplay efficacy.

## Repair plan and review

Scope: correct API descriptions and measurement at existing lifecycle boundaries. Preserve the original B1 source, notebook, manifest and evidence. Main failure paths considered were changed pin/retention behavior, nested double counting, counting system-message omission as eviction, suppressed counting after exceptions, and shipping a stale notebook. Parent comparisons, focused regression checks and installed-package execution cover those paths. The inherited gates preserve sandbox receipt semantics, evidence persistence, ordering and tool pairing.

The harness interpreter lacks Coverage.py/Radon, and no CRAP gate exists for this source tree. `tests/measure_change_risk.py` records a bounded stdlib-trace statement-line estimate under the same inherited gate before/after; its formula and branch convention are explicit, with no added threshold or dependency. The counter has more necessary bookkeeping branches; this is not claimed as a complexity reduction. Functions were reviewed directly rather than split to improve a metric.

Generated game simulators, invalid geometry assumptions, depth-limited searches interpreted as goal rejection, and long reasoning cutoffs remain separate diagnosed failures. Changing the general planning or reasoning policy requires a distinct implementation/evaluation. These two harness repairs do not establish that those failures are solved.
