# Fresh Actor + critic competition candidate — T3 local qualification

Candidate: `pranavagrawai/arc3-ac-competition-20260917-v1`.
Package: `../../deployment/kaggle/submission-20260917-actor-critic-competition-v1/`.
The title equals the installed Kaggle SDK-normalized slug. Metadata inherits the verified R9 datasets, pinned container image, private/GPU settings and sole competition source.

The notebook materializes an allowlisted runtime derived from v11 and the v15 official Agent.main chassis. Commit mode writes a dummy parquet without bootstrap or model startup. Competition rerun uses the pinned competition mount and `http://gateway:8001`; only the explicitly selected local fixture accepts a loopback gateway/model and disposable root. Actor and critic retain the v11 execution path. A shared per-game ledger counts Actor (including summarizer) and critic admissions across solver episodes; caps are 12 and 4, respectively. A reserved critic review can run after Actor admission is exhausted. The existing outer supervisor reconciles persisted request evidence with server request counts.

Time allocation in seconds:

| Component | Cap |
|---|---:|
| Provider maximum | 32400 |
| Provider overhead uncertainty | 900 |
| Final output | 300 |
| Offline bootstrap | 1800 |
| Model startup including warmup | 1200 |
| Outer cleanup | 180 |
| Gameplay across 110 games | 28020 |

Gameplay includes a final 180-second reserve and, per remaining game, 30 seconds cleanup, two 90-second request tails and a 15-second tool margin. Initial Actor admission is about 28.09 seconds per game; unused time rolls forward. The second request tail funds a critic followed by an already-admitted Actor continuation. These are bounded feasibility allocations, not measured latency predictions. Production rejects an impossible all-game envelope before model startup. The provider startup reserve is an explicit uncertainty margin, not proof of provider billing or timeout behavior.

`RESULT.md` and `checks/final-checks.json` report final counts and exact hashes. Raw failed attempts are retained separately. The local dependency installation is confined to `checks/local-deps` and `checks/local-deps-cp312` (pandas 3.0.3 and pyarrow 23.0.1); no existing environment was modified.

Read-only gate command (expected exit 2/HOLD until independent review and authorization exist):

```sh
python3 -B analysis/competition-submission-20260917-actor-critic-v1/launch_control.py check --run-dir /home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus/analysis/competition-submission-20260917-actor-critic-v1
```

Do not rerun `freeze.py` after independent review; it authors a new manifest. Reconciliation recomputes all source, donor, metadata, notebook and embedded-asset bindings. `authorization.json` and `final-review.json` remain outside the manifest and must bind its exact hash plus card/notebook hashes and identity. A future authorized `push` consumes authorization atomically before the provider call, including failed or uncertain invocations. No automatic competition submission is implemented. Exact-version readback/retrieval uses `exact_version.py` with the installed Kaggle SDK interpreter.

`REVIEW-REQUEST.md` names the exact T4 review surface and remaining remote boundaries. There is no self-approval. No GPU, paid model call, kernel push or competition submission occurred in T3.
