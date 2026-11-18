"""C7R1 station 1: identity, C6R2/C6R1 reviews, chain, read-only source, C7 HOLD integrity, no prior claim,
real frozen gate exercised with C6R2 against preserved metadata bytes, local gate tests. Zero network, zero generation."""
import datetime, json, os, re, shutil, stat, subprocess, sys
from pathlib import Path

ROOT = Path("/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus")
CAND = ROOT / "analysis/operational-theory-t1r7-20260917"
HOLD = ROOT / "analysis/operational-theory-replacement-20260917T174546Z-c7-hold"
STAGE = Path(sys.argv[1])
EXPECTED_MANIFEST = "539540436f6805b17a05ee92a7eda22b9eee85dc8a56ae254b9591bee7417c62"
EXPECTED_CARD = "a6eadd19317b95cc5e5067c0f231c74422945757be3f59b3adcd1df6e8f023db"
EXPECTED_REVIEW = "0f3a8a53df820de5bf5eebf088034384b6632a3f8c0d5a568954806d0e0b4938"      # C6R2, gate-compatible
EXPECTED_SUBSTANTIVE = "e710afb7d9a3920d70be631b3e5d6b893c43a148eac25ee4e6898dd0f0879998"  # C6R1, substantive
GATE_TASK = "operational-theory-T1R7"
sys.path.insert(0, str(CAND))
from identity import verify, sha  # noqa: E402
from chain import validate_chain  # noqa: E402
import run_pair  # noqa: E402
from transport import Stop  # noqa: E402

record = {"task": "C7R1", "station": "preflight", "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "network_calls": 0, "generation_calls": 0}
# 1. hashes
assert verify() == EXPECTED_MANIFEST, "manifest drift"
assert sha(CAND / "EXPERIMENT-CARD.md") == EXPECTED_CARD, "card drift"
review = CAND / "C6R2-REVIEW.json"
substantive = CAND / "C6R1-REVIEW.json"
assert sha(review) == EXPECTED_REVIEW, "C6R2 review drift"
assert sha(substantive) == EXPECTED_SUBSTANTIVE, "C6R1 review drift"
r2 = json.loads(review.read_text()); r1 = json.loads(substantive.read_text())
assert r2["role"] == "independent-reviewer" and r2["decision"] == "PASS" and r2["task"] == GATE_TASK
assert r2["manifest_sha256"] == EXPECTED_MANIFEST and r2["card_sha256"] == EXPECTED_CARD
assert r2["findings"] == [] and r2["handoff"]["c7r1_authorized"] is True and r2["launch_performed"] is False
assert r2["substantive_review"]["sha256"] == EXPECTED_SUBSTANTIVE and r2["substantive_review"]["decision"] == "PASS"
assert r1["role"] == "independent-reviewer" and r1["decision"] == "PASS" and r1["manifest_sha256"] == EXPECTED_MANIFEST and r1["card_sha256"] == EXPECTED_CARD and r1["findings"] == []
record["reviews"] = {"gate_compatible": {"path": str(review.relative_to(ROOT)), "sha256": EXPECTED_REVIEW, "task": r2["task"]},
                     "substantive": {"path": str(substantive.relative_to(ROOT)), "sha256": EXPECTED_SUBSTANTIVE, "task": r1["task"], "incorporated_by_c6r2": True}}
# 2. chain
ok, errors, sequence = validate_chain(CAND / "PREDECESSOR.json", ROOT)
assert ok, errors
record["chain"] = {"sequence": sequence, "errors": errors, "nodes": len(sequence)}
# 3. source read-only, no generated artifacts, no pycache
writable = []; pycache = []; extra = []
manifest = json.loads((CAND / "MANIFEST.json").read_text())
allowed = {"MANIFEST.json", "C6R1-REVIEW.json", "C6R2-REVIEW.json", "T6R1-REVIEW.json", "T6-REVIEW.json"}
for p in CAND.rglob("*"):
    if p.is_symlink(): writable.append(str(p) + " (symlink)"); continue
    if stat.S_IMODE(p.stat().st_mode) & 0o222: writable.append(str(p.relative_to(CAND)))
    if "__pycache__" in p.parts or p.suffix == ".pyc": pycache.append(str(p.relative_to(CAND)))
    if p.is_file() and str(p.relative_to(ROOT)) not in manifest["files"] and p.name not in allowed: extra.append(str(p.relative_to(CAND)))
assert not writable, writable
assert not pycache, pycache
assert not extra, extra
assert not (CAND / "live").exists()
record["source"] = {"read_only": True, "pycache": [], "non_manifest_files": sorted(allowed & {p.name for p in CAND.iterdir()})}
# 4. C7 HOLD preserved and recorded zero generation / no consumption
hold_result = json.loads((HOLD / "RESULT.json").read_text())
hold_manifest = json.loads((HOLD / "ARTIFACT-MANIFEST.json").read_text())
assert hold_result["status"] == "BLOCKED" and hold_result["generation_calls"] == 0 and hold_result["network_calls"] == 0
assert hold_result["claim_created"] is False and hold_result["consumed_record"] is False and hold_result["authorization_written"] is False and hold_result["release_consumed"] is False
for name, entry in hold_manifest["files"].items():
    assert sha(HOLD / name) == entry["sha256"], "C7 HOLD artifact drift: " + name
assert not any(stat.S_IMODE(p.stat().st_mode) & 0o222 for p in HOLD.rglob("*"))
record["c7_hold"] = {"path": str(HOLD.relative_to(ROOT)), "status": hold_result["status"], "generation_calls": 0, "claim_created": False, "sealed_files_verified": len(hold_manifest["files"]), "preserved_unmodified": True}
# 5. no prior replacement launch / claim / consumed for this manifest (the C7 HOLD is not a launch)
claims = ROOT / "analysis/operational-theory-replacement-claims"
prior = []
if claims.exists(): prior += [str(p) for p in claims.iterdir()]
prior += [str(p) for p in (ROOT / "analysis").iterdir() if re.fullmatch(r"operational-theory-replacement-[0-9]{8}T[0-9]{6}Z", p.name)]
for d in (ROOT / "analysis").glob("operational-theory*"):
    if d == HOLD: continue
    for f in d.rglob("*.json"):
        if any(part in ("worker-source", "actor", "canonical") for part in f.parts): continue
        if "AUTHORIZ" in f.name.upper() or f.name in ("CONSUMED.json", "RESULT.json", "STOP.json"):
            try:
                if EXPECTED_MANIFEST in f.read_text() and "REVIEW" not in f.name.upper(): prior.append(str(f))
            except OSError: pass
assert not prior, prior
record["prior_launches_for_manifest"] = []
# 6. exercise the REAL frozen gate with C6R2 against preserved (not fresh) metadata bytes. Probe authorization
#    exists only to isolate the review half; it is hashed then discarded and authorizes nothing.
tmp = STAGE / "gate-probe"; tmp.mkdir()
meta = CAND / "metadata/fixed-endpoints.raw"
meta_sha = run_pair.metadata_check(meta)
probe_auth = tmp / "probe-auth.json"
probe_auth.write_text(json.dumps({"role": "user", "decision": "AUTHORIZE_REPLACEMENT_PAIR", "task": GATE_TASK, "manifest_sha256": EXPECTED_MANIFEST, "metadata_sha256": meta_sha, "probe": "gate-isolation only; not an authorization"}))
probe = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "network_calls": 0, "generation_calls": 0, "metadata_bytes": "preserved candidate metadata/fixed-endpoints.raw (local parse only)", "preserved_metadata_sha256": meta_sha}
try:
    got = run_pair.gates(review, probe_auth, meta)
    probe["frozen_gate_with_c6r2_review"] = "ACCEPTED"; assert got == (EXPECTED_MANIFEST, meta_sha)
except Stop as e:
    probe["frozen_gate_with_c6r2_review"] = "REFUSED: " + str(e)
try:
    run_pair.gates(substantive, probe_auth, meta); probe["frozen_gate_with_c6r1_review"] = "ACCEPTED"
except Stop as e:
    probe["frozen_gate_with_c6r1_review"] = "REFUSED: " + str(e)
probe["discarded_probe_files_sha256"] = {p.name: sha(p) for p in tmp.iterdir()}
shutil.rmtree(tmp)
assert probe["frozen_gate_with_c6r2_review"] == "ACCEPTED", probe
assert probe["frozen_gate_with_c6r1_review"].startswith("REFUSED"), probe
(STAGE / "GATE-PROBE.json").write_text(json.dumps(probe, indent=2, sort_keys=True) + "\n")
record["gate_probe"] = {"c6r2": probe["frozen_gate_with_c6r2_review"], "c6r1": probe["frozen_gate_with_c6r1_review"]}
# 7. inventories of candidate + every chain node + launch/audit dirs + C7 HOLD (SOURCE-BEFORE)
dirs = [CAND]
node = json.loads((CAND / "PREDECESSOR.json").read_text())
while node:
    dirs.append(ROOT / node["identity"])
    node = node.get("earlier_chain")
dirs += [ROOT / "analysis/operational-theory-t3-20260917T134947Z", ROOT / "analysis/operational-theory-t4-audit-20260917", HOLD]
inventory = {}
for d in dirs:
    assert d.is_dir(), d
    inventory[str(d.relative_to(ROOT))] = {str(f.relative_to(d)): sha(f) for f in sorted(d.rglob("*")) if f.is_file() and not f.is_symlink()}
(STAGE / "SOURCE-BEFORE.json").write_text(json.dumps(inventory, sort_keys=True, indent=2) + "\n")
record["source_before_dirs"] = {k: len(v) for k, v in inventory.items()}
# 8. local launch gate: offline release/chain tests, no cache writes
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
env.pop("OPENROUTER_API_KEY", None)
cmd = [str(ROOT / "deployment/kaggle/.venv/bin/python"), "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(STAGE), str(CAND / "test_release.py"), str(CAND / "test_chain.py")]
proc = subprocess.run(cmd, cwd=str(STAGE), env=env, capture_output=True, text=True, timeout=600)
(STAGE / "preflight-tests.txt").write_text(proc.stdout + proc.stderr)
assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
record["local_gate"] = {"command": " ".join(cmd[2:]), "returncode": 0, "tail": proc.stdout.strip().splitlines()[-1]}
# 9. tooling / credential plumbing presence (never read)
record["sandbox_tools"] = {"bwrap": bool(shutil.which("bwrap")), "unshare": bool(shutil.which("unshare"))}
record["credential_present_in_environment"] = bool(os.environ.get("OPENROUTER_API_KEY"))
record["credential_plumbing"] = "ROOT/.env OPENROUTER_API_KEY via python-dotenv, loaded only into the frozen driver subprocess environment at launch (T3 precedent); never printed or persisted"
record["credential_file_present"] = (ROOT / ".env").is_file()
record["expected"] = {"manifest_sha256": EXPECTED_MANIFEST, "card_sha256": EXPECTED_CARD, "review_sha256": EXPECTED_REVIEW, "substantive_review_sha256": EXPECTED_SUBSTANTIVE}
record["user_task_sha256"] = sha(STAGE / "USER-TASK.txt")
record["status"] = "GO"
(STAGE / "PREFLIGHT.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
