# Authorized one-attempt Marlin replay launch

User authority: "do the launch setup, then then launch it." Recorded in AUTHORITY.json. This authorizes the existing single replay qualification under the one-hour card. No serving comparison or dataset upload is included. The target reduction contrast and numerical/performance gates remain unchanged. A corrected identity fixes the two inherited evidence paths; the predecessor's41 prepared artifacts are preserved.

The adapters reuse the qualified package, dependency lock, original replay dataset/readback and metadata. They check typed used+reserved quota plus3600 seconds against108000 seconds, require no competing or unknown running/queued kernel, and reject an already-existing remote slug. A live read-only collector is armed before the final repeated admission check. The exclusive receipt is created immediately before the only SDK push, which includes a3600-second timeout. Any save error or incomplete acceptance consumes the local attempt identity and cannot trigger another push.

Remote notebook cells and hardware/model/dataset metadata are checked against the sealed source. Collection checks the accepted current version before and after explicit streamed pagination: the SDK's status/output methods ignore a version suffix, and its high-level output helper recursively fetches pages itself. The adapter uses native page requests to detect repeated tokens and prevents later-run output mixing. Signed output URLs and authentication values are never written to receipts. Existing OAuth credentials are read; equivalent refresh tokens stay in process memory so credential-cache writes do not touch the user's protected home directory.

The current session gives Git metadata read-only access. New adapters are identified by the exclusive launch-freeze.json SHA256 seal instead of changing Git refs or the previously committed experiment. No candidate, cap, metric or experiment-card bytes are changed.

Commands, using the installed SDK interpreter:

```bash
python3 -B analysis/marlin-reduction-replay-v2-20261010/seal_launch.py
~/.local/share/uv/tools/kaggle/bin/python -B analysis/marlin-reduction-replay-v2-20261010/execute.py
```

The collector is a detached, explicitly read-only child with an exclusive lock, a checked PID/readiness receipt, and48-hour deadline. It polls every120 seconds after acceptance, streams all outputs, hashes them, and invokes audit_capture.py. If the host terminates the collector, resume only collect.py; that command has no push or relaunch path. A saved acceptance-unresolved receipt needs read-only remote reconciliation rather than rerunning execute.py.

Environment failure: DNS resolution for www.kaggle.com currently fails in this session's restricted network; no browser provider is available. Admission must fail closed before the push receipt. The shared board is readable through a hash-stable copy of its clean database; direct SQLite access needs journal-file writes outside the allowed roots. No write to the shared board or Git metadata was attempted. The network observation is a launch-access failure, not a GPU attempt or a treatment result. Authority persists; another user approval is not needed when access is restored, provided no push has been attempted and the seal still verifies.
