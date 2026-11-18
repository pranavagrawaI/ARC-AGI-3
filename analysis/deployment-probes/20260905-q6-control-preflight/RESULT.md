# Kaggle implementation result — 2026-09-05

Status: package complete and locally verified; GPU launch pending explicit user answer. No kernel pushed, no competition submission, no commit.

Delivered `deployment/kaggle/q6-control-preflight`: private kernel metadata, byte-identical qualified control notebook, hash-bound launch argv/card, drift verifier, offline actual-CLI request check and run/download/scoring instructions. Candidate contract remains 1d65e3669ff7ace45eb271aaf943dff35fc279a6ebeba95912e89884d3a05459. Existing q6 candidate and unrelated worktree changes were preserved.

The run is ls20 only, 200 decisions, treatment off, thinking disabled, 4096 output tokens/request; startup 1200 s, gameplay launcher 7200 s, server session 14400 s. One push, no automatic retry. The requested RTX Pro 6000 is explicit in launch argv. A read-only live quota check reported 30.00 GPU-hours remaining, refresh 2026-09-12; this observation must be refreshed before launch.

Verification:

- `deployment/kaggle/.venv/bin/python -m pytest deployment/kaggle/tests/test_q6_preflight.py deployment/kaggle/tests/test_q6_candidate.py -q`: 4 passed in 0.08 s. Checks frozen-control identity, card drift rejection, complete candidate payload, and control/treatment difference.
- `/home/pranav/.local/share/uv/tools/kaggle/bin/python deployment/kaggle/q6-control-preflight/verify_request.py`: PASS; actual Kaggle CLI 2.2.4 request construction, transport replaced, zero network calls. Private=true, timeout=14400, machine_shape=NvidiaRtxPro6000. See REQUEST-CHECK.json.
- Ruff passed on new package Python and test files; notebook cells compile via the frozen builder's validation.
- Initial verification failed when Python 3.12 and the system Python rebuilt compressed ZIP payloads differently. Reusing the qualified notebook by fixed SHA-256 removes cross-interpreter byte drift. The final artifact is byte-identical to q6 control.ipynb.

The preflight card fixes how validity, >=3 evaluable stalls, hold evidence and runtime determine the next decision. This single control run cannot establish treatment efficacy and will not count as a later arm replicate. Actual accelerator allocation, model behavior and output completeness remain untested until execution.

Next: resolve the pending launch-scope question. If authorized, recheck hashes and quota, execute LAUNCH.json's argv exactly once, save the response, monitor terminal status, download all output and score it with the frozen q6 scorer. Never retry an ambiguous push without reconciling remote status. README.md contains the exact commands. A run failure requires a new preflight identity/card.
