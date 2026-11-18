"""Local-only reproduction: does the frozen gate accept C6R1-REVIEW.json? No network, no credential, no claim."""
import json, sys, datetime
from pathlib import Path
ROOT = Path("/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus")
CAND = ROOT / "analysis/operational-theory-t1r7-20260917"
STAGE = Path(sys.argv[1]); tmp = STAGE / "gate-repro"
sys.path.insert(0, str(CAND))
import run_pair
from identity import sha, verify
from transport import Stop
meta = CAND / "metadata/fixed-endpoints.raw"   # preserved local bytes, not a fresh fetch
meta_sha = run_pair.metadata_check(meta)        # local parse only
auth = tmp / "probe-auth.json"                  # PROBE ONLY: authorizes nothing, used only to isolate the review gate
auth.write_text(json.dumps({"role": "user", "decision": "AUTHORIZE_REPLACEMENT_PAIR", "task": "operational-theory-T1R7", "manifest_sha256": verify(), "metadata_sha256": meta_sha, "probe": "gate-isolation only; not an authorization"}))
out = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "network_calls": 0, "generation_calls": 0, "review_sha256": sha(CAND / "C6R1-REVIEW.json"), "review_task": json.loads((CAND / "C6R1-REVIEW.json").read_text())["task"]}
try:
    run_pair.gates(CAND / "C6R1-REVIEW.json", auth, meta)
    out["frozen_gate_with_c6r1_review"] = "ACCEPTED"
except Stop as e:
    out["frozen_gate_with_c6r1_review"] = "REFUSED: " + str(e)
# control: same review bytes with only task rewritten (in tmp, never in source) -> isolates the single failing field
probe_review = tmp / "probe-review-task-rewritten.json"
r = json.loads((CAND / "C6R1-REVIEW.json").read_text()); r["task"] = "operational-theory-T1R7"; probe_review.write_text(json.dumps(r))
try:
    run_pair.gates(probe_review, auth, meta); out["frozen_gate_with_task_rewritten_probe"] = "ACCEPTED (probe only; this file is not the reviewer's record and authorizes nothing)"
except Stop as e:
    out["frozen_gate_with_task_rewritten_probe"] = "REFUSED: " + str(e)
out["gate_source"] = 'run_pair.gates: record.get("task") != "operational-theory-T1R7" -> Stop("missing exact-byte T6 review or replacement-pair authorization")'
(STAGE / "GATE-REPRO.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
print(json.dumps(out, indent=2, sort_keys=True))
