# M2 F1: friction bundle on B1-fixed

F1 is B1-fixed plus two friction fixes. B1-fixed is `solvers/m2_b1_fixed`: B1, the notebook drawn as hidden D5 (26.96), with its false frame-API instruction repaired. It was added on 2026-10-17, before launch, on the notebook-mining report. It comes from the plan agreed on 2026-10-16: "start with friction removal", where pieces enter on mechanism plus no-harm. `build.py` derives everything from the frozen B1-fixed notebook. Its install cell is untouched; its sandbox is byte-identical to B1's. F1 adds one hash-checked patch cell after it and changes three config lines.

| Piece | Change | Evidence (B1 pair-1, two 25-game runs) |
|---|---|---|
| **preload** | The sandbox preloads `bisect, collections, copy, functools, heapq, itertools, json, math, statistics`. These are names the allowlist already permits. | 7 and 1 NameErrors on these module names (`collections`, `copy`, `itertools`, `json`); each one loses a whole snippet. |
| **out16k** | `LOCAL_ANALYZER_MAX_OUTPUT` 12K→16K. The analyzer window grows (116+12)K→(116+16)K and the server context (116+12+8)K→(116+16+8)K. The reply reserve equals the max output, so the input budget stays 116K. | 28 and 24 responses ended at exactly 12,288 tokens. In production 30 of 31 cutoffs carry no tool call (constraint-audit-20261013), and cutoffs hold ~7% of response tokens. |

Expected size: small. Both pieces together touch about 1% of tool calls and about 7% of tokens, and the token figure only matters if longer replies finish with a call. That's why this is a mechanism-plus-no-harm screen, not a score claim.

## Considered and left out

- **Retaining snippet literals** (`CROSS = (...)` read by a later function). It would remove ~11 NameErrors per run plus some retention rejections. Left out because it means reworking the hardened retention module, and a constant kept from an earlier level could silently mislead a later one.
- **StaleStateActionError** (~30 per run). This is B1's deliberate guard that stops a batch after a no-op move. It's protection, not friction.
- **Tool timeouts.** Only 2–3 per run at the 60 s ceiling.

## Verify

```bash
PY=/home/pranav/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/bin/python3.12
PYTHONDONTWRITEBYTECODE=1 $PY -B solvers/m2_f1/build.py
PYTHONDONTWRITEBYTECODE=1 $PY -B -m unittest solvers/m2_f1/tests/test_f1.py
```

`test_f1.py` runs the notebook's actual install cell against a copy of the B1-fixed source, then exercises the real sandbox subprocess. It checks forgotten imports, retention of functions that use preloaded modules (with and without explicit imports), rebinding of a preloaded name, an unchanged allowlist, and refusal of a non-B1 sandbox. B1-fixed's own suite (`solvers/m2_b1_fixed/tests`, 29 tests) also passes on the F1 tree when `M2_SOURCE_ROOT` points at it. On plain B1, the forgotten-import case fails with NameError.

`incumbent/check.py` needs `--without c14 --without server-fail-fast --without reap448`, because the B1 notebook drawn as D5 never carried them.

## Screen

`experiment/screen-1` (card inside) holds the f1 and b1ctl arms, 60 min of play each. They're launched once by the systemd user timer `m2-f1-screen1` after the 2026-10-17 quota reset, then collected and analyzed with `analyze.py`. The analyzer was checked on real B1 output (MECHANISM_FAIL, as expected without F1) and on a synthetic passing screen (PASS).
