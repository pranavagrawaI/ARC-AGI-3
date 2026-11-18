"""C7 HOLD: preserve station-1 evidence, verify source unchanged, seal read-only. Zero network, zero generation."""
import datetime, hashlib, json, os, shutil, stat, sys
from pathlib import Path
ROOT = Path("/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus")
CAND = ROOT / "analysis/operational-theory-t1r7-20260917"
STAGE = Path(sys.argv[1])
sys.path.insert(0, str(CAND))
from identity import verify, sha
now = datetime.datetime.now(datetime.timezone.utc)
OUT = ROOT / "analysis" / f"operational-theory-replacement-{now:%Y%m%dT%H%M%SZ}-c7-hold"
assert not OUT.exists()
OUT.mkdir()
# probe files: record hashes, then discard (a task-rewritten copy of the reviewer's record must not persist as a look-alike review)
probes = {p.name: sha(p) for p in (STAGE / "gate-repro").iterdir()}
shutil.rmtree(STAGE / "gate-repro")
for name in ("USER-TASK.txt", "PREFLIGHT.json", "GATE-REPRO.json", "SOURCE-BEFORE.json", "preflight-tests.txt"):
    shutil.copy2(STAGE / name, OUT / name)
(OUT / "operator").mkdir()
for p in (STAGE / "operator").glob("*.py"):
    shutil.copy2(p, OUT / "operator" / p.name)
# source unchanged
before = json.loads((OUT / "SOURCE-BEFORE.json").read_text()); counts = {}
for name, expected in before.items():
    d = ROOT / name
    actual = {str(f.relative_to(d)): sha(f) for f in sorted(d.rglob("*")) if f.is_file() and not f.is_symlink()}
    assert actual == expected, name
    counts[name] = len(actual)
manifest_sha = verify()
assert manifest_sha == "539540436f6805b17a05ee92a7eda22b9eee85dc8a56ae254b9591bee7417c62"
assert not (ROOT / "analysis/operational-theory-replacement-claims").exists()
assert not list((ROOT / "analysis").glob("operational-theory-replacement-[0-9]*Z"))
pre = json.loads((OUT / "PREFLIGHT.json").read_text()); repro = json.loads((OUT / "GATE-REPRO.json").read_text())
result = {
    "task": "C7", "status": "BLOCKED", "stage": "preflight station 1 (identity/review/chain/launch gate), before metadata lookup",
    "generation_calls": 0, "network_calls": 0, "metadata_get_requests": 0, "provider_retries": 0,
    "model": "qwen/qwen3.8-27b", "endpoint": "reka/fp8", "credential_loaded": False,
    "authorization_written": False, "authorization_sha256": None, "claim_created": False, "consumed_record": False,
    "requests": {"control": 0, "treatment": 0, "total": 0},
    "usage": {"tokens": {"input": 0, "output": 0, "total": 0}, "usd": "0", "unresolved_requests": 0, "unresolved_exposure_usd": "0"},
    "control_metrics": None, "treatment_metrics": None, "delta": None,
    "validity_classification": "NOT_RUN: blocked before any generation; no result to classify",
    "blocker": {
        "id": "C7-B1",
        "finding": "The frozen driver's release gate (run_pair.gates) refuses the exact-byte independent review C6R1-REVIEW.json: the record's task field is \"C6R1\", while the gate and EXPERIMENT-CARD.md require task == \"operational-theory-T1R7\" for the review input. All other review fields (role, decision, manifest_sha256, card_sha256) match.",
        "reproduction": repro,
        "isolated_field": {"file": "analysis/operational-theory-t1r7-20260917/C6R1-REVIEW.json", "pointer": "/task", "actual": "C6R1", "required": "operational-theory-T1R7"},
        "precedent": "T2R3-REVIEW.json (launched under T3) carried task operational-theory-T1R3; C6R1's own card_exactness finding quotes the gate's task binding",
        "why_not_repaired": "The brief forbids modifying the frozen candidate/review and any manual repair or gate bypass; the review record is owned by the independent reviewer, not the launch operator. Rewriting its task field or monkeypatching gates() would be authoring approval.",
        "not_a_machinery_defect": "The gate behaved as frozen and reviewed; the defect is in the form of the review record. C6R1's substantive PASS is unaffected.",
        "discarded_probe_files_sha256": probes,
    },
    "preflight_passed_before_blocker": {
        "card_sha256": pre["expected"]["card_sha256"], "manifest_sha256": pre["expected"]["manifest_sha256"], "review_sha256": pre["expected"]["review_sha256"],
        "predecessor_chain": pre["chain"]["sequence"], "source_read_only_no_pycache": True, "prior_launches_for_manifest": [],
        "local_gate": pre["local_gate"]["tail"], "sandbox_tools": pre["sandbox_tools"],
    },
    "source_identity_unchanged": {"manifest_sha256_after": manifest_sha, "inventory_files_unchanged": counts},
    "release_consumed": False,
    "handoff": {
        "to": "independent C8 audit; then the C6R1 reviewer",
        "minimal_unblock": "The reviewer re-issues the PASS as a new record (e.g. C6R2-REVIEW.json, excluded from identity like C6R1) with task == \"operational-theory-T1R7\" and the same role/decision/manifest_sha256/card_sha256, or the user/reviewer decides on a new candidate identity whose gate accepts the reviewer's task label. The operator does not choose between these.",
        "then": "A fresh C7-style launch: station 1 again, fresh metadata GET, new manifest-bound user authorization, one pair. Nothing from this hold is consumed or reusable as authority.",
        "kept_out_of_this_hold": "No fresh OpenRouter metadata was fetched: the brief orders the hash/scope gate before the metadata lookup, and any metadata fetched now would not be the one a future authorization must bind.",
    },
    "scientific_scope": "none; no construction result, no gameplay",
    "at_utc": now.isoformat(),
}
(OUT / "RESULT.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
(OUT / "REPORT.md").write_text(f"""# C7: HOLD at preflight station 1 — frozen gate refuses the C6R1 review's task label

Status: **BLOCKED**, generation_calls = 0, network_calls = 0, credential never loaded,
no authorization written, no claim created, no release consumed, source identity unchanged
(manifest `{manifest_sha}`).

What passed before the block: card/manifest/review hashes exact; C6R1 decision PASS with
`findings: []` and `handoff.c7_authorized: true`; predecessor chain t1r6→…→t1 resolves (7 nodes,
0 errors); candidate read-only with no pycache and no non-manifest files beyond the four review/
manifest records; no prior replacement claim/output/CONSUMED for this manifest; offline release
and chain tests `{pre["local_gate"]["tail"]}`; bwrap/unshare present.

The block (C7-B1): `run_pair.gates()` requires the review record's `task` to equal
`operational-theory-T1R7` (EXPERIMENT-CARD.md: "Review input has role independent-reviewer,
decision PASS, task operational-theory-T1R7, and this manifest hash"). C6R1-REVIEW.json carries
`task: "C6R1"`. Reproduced locally against the frozen driver with preserved metadata bytes and a
throwaway gate-isolation probe (no network, no credential, no claim): the exact-byte review is
`REFUSED: missing exact-byte T6 review or replacement-pair authorization`; the same bytes with
only `/task` rewritten are accepted. The probe copies were hashed and discarded so no look-alike
review persists; `GATE-REPRO.json` and `operator/gate_repro.py` preserve the reproduction.

Why the operator did not fix it: the brief forbids modifying the frozen candidate/review and any
manual repair or bypass, and the card assigns review records to the independent reviewer. A
launch operator rewriting the reviewer's task label — or patching the gate — would be authoring
approval. This is a form defect in the review record, not a machinery defect and not a
substantive review failure.

Not done, deliberately: no fresh OpenRouter metadata GET. The brief orders the identity/scope
gate before the metadata lookup, and any metadata fetched now could not be the fresh bytes a
future authorization must hash-bind.

Handoff: independent C8 audit of this hold; then the C6R1 reviewer re-issues the PASS with the
gate's task label (a new excluded record such as C6R2-REVIEW.json with identical role/decision/
manifest/card hashes), after which a fresh C7-style launch runs station 1 again, the metadata
lookup, a new manifest-bound user authorization, and one pair. Nothing here is reusable as
authority. No construction or gameplay result exists.
""")
# seal
files = {}
for p in sorted(OUT.rglob("*")):
    if p.is_file():
        raw = p.read_bytes(); files[str(p.relative_to(OUT))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
am = {"task": "C7", "status": "BLOCKED", "generation_calls": 0, "source_manifest_sha256": manifest_sha, "files": files, "credential_scan": "no credential was ever loaded in this session; .env never read", "provider_retries": 0, "self_excluded": "ARTIFACT-MANIFEST.json"}
with (OUT / "ARTIFACT-MANIFEST.json").open("x") as f: f.write(json.dumps(am, sort_keys=True, indent=2) + "\n")
for p in OUT.rglob("*"):
    if p.is_file(): p.chmod(stat.S_IMODE(p.stat().st_mode) & ~0o222)
for p in sorted((p for p in OUT.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True): p.chmod(stat.S_IMODE(p.stat().st_mode) & ~0o222)
OUT.chmod(stat.S_IMODE(OUT.stat().st_mode) & ~0o222)
print(json.dumps({"launch_dir": str(OUT.relative_to(ROOT)), "artifact_manifest_sha256": hashlib.sha256((OUT / "ARTIFACT-MANIFEST.json").read_bytes()).hexdigest(), "result_sha256": files["RESULT.json"]["sha256"], "files": len(files), "source_manifest_after": manifest_sha}, indent=2))
