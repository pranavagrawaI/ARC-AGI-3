# Local qualification

Runtime contract: `eff3cc37598dc8c31279919ccdc3dc8d1e22ec23c8512772670085886f789cda`.

- Full runtime suite: 188 passed (`pytest .` in this directory).
- `verify_candidate.py`: PASS, 0 network calls, slug `pranavagrawai/tycho-nooa-v12-sb26-critic-off`.
- `freeze_candidate.py --check` and `prepare_candidate.py --check` match.

These are deterministic and captured-CLI checks. They do not establish Qwen
play quality, GPU behavior, or Kaggle performance. The critic-off sb26 run is
authorized by `LAUNCH.json` and `EXPERIMENT-CARD.md`.
