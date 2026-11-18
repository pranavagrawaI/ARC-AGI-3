# q6 summarization repair — local qualification

Implemented and locally qualified `deployment/kaggle/q6-summary-boundary-20260906`. All 51 tests pass: 15 summarization-boundary cases and 36 inherited host/prediction checks. The successor notebook has not been pushed. No paid model calls were made.

Contract: `24c291bc6956a96dce361a52a713c0b3c4e4208da7e464d735d2b310834976a9`.
Notebook SHA256: `35ec3048a4eff4e100f5f140ba28fc1cde0d1da3635e53d5db5fd7932e2a6f7a`.

## Change

The guard now validates notes in the original history, binds that history to the background summarizer call with a ContextVar, and verifies the actual serialized Task at the final request boundary using NOOA's formatter. It preserves the witness across structured-output retries and resets it when the call ends. Completed summarizer calls remain recorded and unexpected reasoning still aborts.

Actor validation selects active PythonOutput events and binds each readback to its event tag. A summary mentioning an archived tool ID no longer implies that the original output is still visible. An intact copy in summary prose cannot satisfy a changed Actor event. Legitimately omitted history is recorded as UNOBSERVED, not as verified readback.

Only `chassis/solver_agent.py` and `chassis/integrity_guard.py` changed inside the runtime payload. The prediction workflow, summarizer prompt, inference settings, and gameplay caps remain inherited. The failed parent payload and downloaded raw evidence are unchanged.

## Evidence

- `REGRESSION-BEFORE.txt`: all five printed-note compression lifecycle variants fail on the frozen original package with the original integrity error.
- `TESTS-FINAL.txt`: 51 passed on the repaired extracted payload, including those five variants.
- `test_summary_boundary.py`: actual CodeAct note write/read, scheduled summarization, application of the resulting Summary at BeforeTurn, and resumed Actor request. Also covers the exact captured Kaggle request; Unicode, quotes, backslashes, triple quotes, and Markdown event-like headings inside notes; altered source history and final request; intact decoys; retries; repeated compression; absent history; and unbound summarizer calls.
- Existing checks retain Actor/summarizer accounting, reasoning-token rejection, host lifecycle/clean deadlines, full-grid motion detection, and prediction evaluation budgets.
- `PACKAGE-CHECK.json`: rebuilt notebook/extracted source match the frozen manifest.
- `TRANSPORT-CHECK.json`: Kaggle request construction passes, private RTX PRO 6000 target, zero transport calls.
- `LOCAL-QUALIFICATION.json`: source hashes, test counts, and runtime provenance. Final tests use Python 3.12.14 and installed NOOA 0.0.1.dev95, revision `55a4b57872b49bfe9698cda6fc8e1f3b0bdd723e`, matching the v11 lock; 208 installed Python files match their distribution RECORD hashes. Initial exploratory tests used the editable reference checkout; those were superseded because its Task docstring rendering differs from Kaggle.

The existing `set_dynamic()` deprecation warning remains; there were no failing checks.

## Reproduce

```bash
python3 deployment/kaggle/q6-summary-boundary-20260906/prepare.py --check
deployment/kaggle/.venv/bin/python -m pytest -q \
  analysis/deployment-probes/20260906-q6-summary-boundary/test_payload.py \
  analysis/deployment-probes/20260906-q6-summary-boundary/test_summary_boundary.py \
  analysis/deployment-probes/20260906-q6-summary-boundary/test_join_payload.py \
  analysis/deployment-probes/20260906-q6-summary-boundary/test_deadline.py \
  analysis/deployment-probes/20260906-q6-summary-boundary/test_feedback_budget.py \
  deployment/kaggle/q6-summary-boundary-20260906/test_prediction_feedback.py
```

## Limits

Model responses are scripted. The tests establish that intact history reaches the model boundary, corrupted readbacks are rejected, and the Actor can continue after compression. They do not establish that generated summaries preserve experimental distinctions, that gameplay improves, or that a long GPU run passes. The existing summarizer prompt was not changed, so this is an infrastructure repair rather than a claim of better summarization quality.

The built successor retains the prospective 100-action / 7200-second gameplay admission / 10800-second Kaggle session limits. No launch or automatic retry was performed.
