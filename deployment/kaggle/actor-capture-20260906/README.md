# Actor request collection notebook

Upload `capture.ipynb` with the adjacent `kernel-metadata.json`. Attach competition
`arc-prize-2026-arc-agi-3`, keep the four declared datasets, select RTX PRO 6000,
disable Internet and use a 3600-second session limit. No assistant upload occurred.

The hardware-qualified q6 quick-host notebook is frozen as the base. Its complete
solver ZIP, engine installation, manifest checks, reasoning-off setting, 4096
output-token ceiling, single-game ls20 and 10-action cap remain unchanged.
Only the launcher's model URL is routed through a loopback recorder on port30001;
SGLang still serves port30000. No MTP or tuning changes are included.

The recorder saves actual outgoing request bodies before forwarding identical
bytes, plus paired upstream responses and event hashes. Repeated attempts retain
distinct identities. It stores no Authorization headers. First request also
captures resolved server_info; full config and generation_config are retained.
These are collection timings, not clean engine benchmarks.

After the run download `/kaggle/working/actor-capture-20260906/` and the inherited
`nooa-join-public3/` evidence. Run `inspect_capture.py /path/to/capture-directory`
to list request shapes, actual token counts, tool-result presence and tool_choice.
Reconcile against model_calls/server logs before selecting a corpus. Never infer
that a request is forced-action merely because it exposes one tool. A 10-action
run need not reach long-history or forced-action paths. This q6 donor is
reasoning-off; a later thinking-on workload needs its own capture, not relabelling.

Local tests run HTTP traffic through the exact notebook-embedded recorder and
check payload/response bytes, repeated admissions, secret-header exclusion,
pre-forward disk failure, streaming rejection, and preservation of source cells.
They do not prove GPU execution or game-result quality.

Build/check: `python3 build.py --check`.
Tests: `../.venv/bin/python test_capture.py`.
The optional `verify_push.py` exercises installed Kaggle request construction
with transport mocked; it never uploads. The assistant leaves actual upload to
the user. Old failed runs and previous candidate packages are unchanged.
