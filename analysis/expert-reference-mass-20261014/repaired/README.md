# Locally verified repair; GPU comparison pending

The original screen reused a process-wide scheduler across sequential arms. Its clock intentionally takes the earliest session start/deadline, so a new benchmark object retained the control's expired clock. The original launcher also suppressed successful unittest stderr, losing the detailed GPU qualification report.

Each repaired gameplay arm uses a new interpreter. It reproduces the notebook's explicit source import precedence, rejects an initialized gate before play, and saves/checks the actual scheduling clock after play. Parent containment owns both model and gameplay process groups. Routing qualification saves stdout, stderr and status before rejecting a failure or missing CUDA graph marker. Routing mathematics and model settings are unchanged.

`test_repair.py` reconstructs Duck commit `7652836056c59e044f093e3c13ed7438c814169e` and applies the exact frozen c14 patch. It imports the full real gate module and executes the actual `_HarnessGameSession.play` and `_queue_position` methods; only game/model/artifact IO is substituted. AST extraction of those methods avoids an unavailable `arcengine` dependency on the CPU host. The earlier fake-module fixture was a process-separation check, not a reproduction of the defect.

The real session reproduction gives clock `[100,1000]` for both sequential sessions even when the second starts at1500. For an identical level1 snapshot with10 actions and10000 tokens, stale-clock priority is131; fresh-clock priority is919. Separate repaired child processes both use fresh900-second clocks. This establishes a scheduling effect, not its contribution to the original14-level completion difference.

Five repair tests pass: original stale-clock reproduction; fresh real-module sessions; preinitialized-gate rejection; both owned process groups reaped; actual launcher report retention for successful, failed and missing-graph tests. Routing checks: six pass, one CUDA-only test skips locally. Rebuilt notebook schema, embedded-source compilation, asset hashes and incumbent checks pass. Evidence is in `evidence/diagnosis-replay.json`, `diagnosis-tests.log`, `routing-tests-after.log` and `LOCAL-QUALIFICATION.json`.

Reproduce locally without GPU inference:

```bash
python3 -B analysis/expert-reference-mass-20261014/repaired/test_repair.py
uv run --with nbformat --with ipython python -B analysis/expert-reference-mass-20261014/repaired/build.py
REFERENCE_WHEEL=/tmp/a5b-wheel/sglang-0.5.19+gd00d88efc8d6-cp312-cp312-linux_x86_64.whl OMP_NUM_THREADS=2 analysis/expert-reference-mass-20261014/.venv/bin/python -B analysis/expert-reference-mass-20261014/repaired/test_routing.py
```

The Duck archive location can be overridden with `REFERENCE_DUCK_REPO`; the numerical oracle needs CPU Torch and the exact wheel. No GPU time was spent on this repair. Full Kaggle startup, benchmark serialization and gameplay remain untested in the repaired notebook. The original v1 notebook/card/launch/result and raw evidence are preserved. A clean comparison requires a new bounded authorization: the original card allowed one accepted launch and explicitly prohibited automatic retries. Do not run the original v1 `readout.py` unchanged on a successor: its validity flags are specific to v1.
