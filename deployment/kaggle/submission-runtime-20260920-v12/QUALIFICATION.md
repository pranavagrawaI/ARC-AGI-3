# Local qualification

Runtime contract: `f8fa9c62d63b814b965861646aebbe82f57b3a0b190101ec6b042d6369a78627`.

- Full runtime suite: 187 passed (`pytest .` in this directory).
- `verify_candidate.py`: PASS, 0 network calls, slug `pranavagrawai/tycho-nooa-v12-sb26-su15`.
- `freeze_candidate.py --check` and `prepare_candidate.py --check` match.

These are deterministic and captured-CLI checks. They do not establish Qwen
play quality, GPU behavior, or Kaggle performance. The discriminating run is
authorized by `LAUNCH.json` and `EXPERIMENT-CARD.md`.
