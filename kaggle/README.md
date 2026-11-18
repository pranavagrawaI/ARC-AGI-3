# Reference arms on Kaggle

Two Kaggle notebooks that run the two reference ARC-AGI-3 harnesses — upstream
**NOOA** and upstream **Tycho** — on the 25 public games, served by a local
`Qwen/Qwen3.8-27B-FP8` with the network disabled.

They exist to answer one question neither paper answers: **what do these harnesses
score with an open-weight 27B?** Tycho's published numbers are on Claude Opus 4.8
and GPT-5.6 Sol; NOOA's are on GPT-5.5 and GPT-5.6 Sol. The one external datapoint
for a local 27B is ARC Prize's milestone-1 winner, which reported that gains came
from multimodality and better base models rather than hand-built tools.

Both arms are the baseline against which `arc-agi-3/solvers/tycho_nooa` is judged.

## What is measured, and by what

Both arms sit behind the **same** official `MyAgent` seam, on the **same** serving
route, with the **same** context and decode bounds, and are scored by the **same**
instrument: the official ARC-AGI-3 recordings under `RECORDINGS_DIR`. Only the
harness differs. A difference in the result is therefore a difference between the
harnesses rather than between two measuring rigs.

```
Kaggle kernel (RTX PRO 6000, offline)
  └─ SGLang serving Qwen/Qwen3.8-27B-FP8            ← identical for both arms
     └─ official ARC-AGI-3-Agents launcher, offline ← identical for both arms
        └─ MyAgent  →  LocalSolver (protocol v3)    ← identical for both arms
           └─ arm worker                            ← the only difference
              ├─ nooa-upstream : ipc/states.jsonl ↔ ipc/actions.jsonl
              │                  → pristine MdArcSolverAgent (mdfiles)
              └─ tycho-upstream: proxy Arcade
                                 → upstream run_env → TychoAgent (single)
```

Phase A only. Each notebook refuses competition-rerun mode and creates no
submission.

## Layout

| Path | What it is |
|---|---|
| `common/build_notebook.py` | Builds either arm's notebook from one base and one route |
| `common/reference_probe.py` | Runs the official launcher and summarizes the recordings |
| `common/route/` | The qualified R8-L3B serving route, adapter, and environment manifest |
| `nooa-upstream/upstream/` | Pristine `nooa-pin` solver source and 51-line skill |
| `nooa-upstream/worker/` | The harness-role worker and headless launcher |
| `tycho-upstream/worker/` | The proxy Arcade worker |
| `*/SOURCES.json` | Pins, hashes, and every departure from the reference setup |
| `tests/` | Local join tests — no GPU, no model, no Kaggle |
| `notebooks/` | Generated notebooks and kernel metadata |

## Provenance

Both upstream packages come from the offline wheelhouse
`pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11`, and both were verified
byte-identical to the pristine reference checkouts:

| Distribution | Version | Checked against |
|---|---|---|
| `arc-agi-3-tycho` | 0.1.0 | `arc-agi-3-reference/tycho/tycho` — no differences |
| `nooa` | 0.0.1.dev95 | `nooa-pin/src/nooa` — no differences |

The Tycho arm ships **no** Tycho source and applies **no** deltas. The NOOA arm
ships pristine `solver_agent.py` and `SKILL.md` unchanged, and applies exactly two
deltas to `arc_llm.py`, both marked in place and listed in `SOURCES.json`.

Each arm's `SOURCES.json` also lists what the deployment forces away from the
reference setup. The two that matter most:

* **NOOA runs the `mdfiles` variant, not `memory`.** The memory subsystem needs an
  embeddings gateway and the `nooa-memory` distribution; an offline session has
  neither. The paper measures `mdfiles` at 38.4 RHAE against 50.2 for `memory` on
  GPT-5.5, so this arm is handicapped by about 12 RHAE relative to NOOA's headline.
* **Tycho's per-level baselines are synthetic.** The official host owns the real
  action budget, so this affects Tycho's internal budget and its own scorer, not
  the measured outcome.

## Build and run

```bash
python common/build_notebook.py --arm nooa-upstream
python common/build_notebook.py --arm tycho-upstream

kaggle kernels push -p notebooks/nooa-upstream  --timeout 28800 --accelerator NvidiaRtxPro6000
kaggle kernels push -p notebooks/tycho-upstream --timeout 28800 --accelerator NvidiaRtxPro6000
```

## Local tests

Both joins run end to end on CPU against a fake OpenAI-compatible gateway, using a
venv built from the same wheelhouse the kernel installs:

```bash
uv venv --python 3.12 refvenv
uv pip install --python refvenv/bin/python --offline --no-index \
    --find-links <wheelhouse> <wheelhouse>/arc_agi_3_tycho_nooa-0.1.0-py3-none-any.whl \
                              <wheelhouse>/arc_agi_3_tycho-0.1.0-py3-none-any.whl
refvenv/bin/python tests/test_tycho_join.py
refvenv/bin/python tests/test_nooa_join.py
```

Both passed before the first push. The Tycho run printed:

```
[tycho vision] model=qwen3.8-27b-fp8 render_scale=32 lossless_cells=True
  (qwen3.x patch16×merge2=32px/token; scale=32 → 1 token/cell, no downscale)
```

Tycho's own per-model vision table resolves this checkpoint as the one family
whose encoder recovers the 64×64 grid exactly, and renders accordingly. The NOOA
arm sends its grid as a PNG too (`visual: additive`, upstream's competition
setting). Both reference harnesses use the visual channel that
`solvers/tycho_nooa` does not have.

## What a result here can and cannot say

It can say what each reference harness does with this model on these games, on one
run each, measured by the official recordings.

It cannot say what either would score in its own setup: NOOA is missing its memory
variant, Tycho is on `single` rather than `orchestrator`, both are on the public 25
rather than the hidden set, and neither runs its own concurrent fleet. It is not a
leaderboard score and produces no submission.
