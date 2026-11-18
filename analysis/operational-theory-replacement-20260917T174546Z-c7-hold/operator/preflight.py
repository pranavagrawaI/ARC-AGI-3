"""C7 station 1: identity, review, chain, read-only source, no prior claim, local gate. Zero network."""
import datetime, json, os, stat, subprocess, sys
from pathlib import Path

ROOT = Path("/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus")
CAND = ROOT / "analysis/operational-theory-t1r7-20260917"
STAGE = Path(sys.argv[1])
EXPECTED_MANIFEST = "539540436f6805b17a05ee92a7eda22b9eee85dc8a56ae254b9591bee7417c62"
EXPECTED_CARD = "a6eadd19317b95cc5e5067c0f231c74422945757be3f59b3adcd1df6e8f023db"
EXPECTED_REVIEW = "e710afb7d9a3920d70be631b3e5d6b893c43a148eac25ee4e6898dd0f0879998"
sys.path.insert(0, str(CAND))
from identity import verify, sha  # noqa: E402
from chain import validate_chain  # noqa: E402

record = {"station": "preflight", "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "network_calls": 0, "generation_calls": 0}
# 1. hashes
assert verify() == EXPECTED_MANIFEST, "manifest drift"
assert sha(CAND / "EXPERIMENT-CARD.md") == EXPECTED_CARD, "card drift"
review = CAND / "C6R1-REVIEW.json"
assert sha(review) == EXPECTED_REVIEW, "review drift"
r = json.loads(review.read_text())
assert r["role"] == "independent-reviewer" and r["decision"] == "PASS" and r["task"] == "C6R1"
assert r["manifest_sha256"] == EXPECTED_MANIFEST and r["card_sha256"] == EXPECTED_CARD
assert r["handoff"]["c7_authorized"] is True and r["findings"] == []
# run_pair.gates requires record["task"] == "operational-theory-T1R7"; check what the review carries
record["review_task_field"] = r["task"]
record["review_gate_task_match"] = r.get("task") == "operational-theory-T1R7"
# 2. chain
ok, errors, sequence = validate_chain(CAND / "PREDECESSOR.json", ROOT)
assert ok, errors
record["chain"] = {"sequence": sequence, "errors": errors, "nodes": len(sequence)}
# 3. source read-only, no generated artifacts, no pycache
writable = []; pycache = []; extra = []
manifest = json.loads((CAND / "MANIFEST.json").read_text())
allowed = {"MANIFEST.json", "C6R1-REVIEW.json", "T6R1-REVIEW.json", "T6-REVIEW.json"}
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
# 4. no prior C7 launch / claim / consumed for this manifest
claims = ROOT / "analysis/operational-theory-replacement-claims"
prior = []
if claims.exists(): prior += [str(p) for p in claims.iterdir()]
prior += [str(p) for p in (ROOT / "analysis").glob("operational-theory-replacement-*")]
for d in (ROOT / "analysis").glob("operational-theory*"):
    for f in d.rglob("*.json"):
        if any(part in ("worker-source", "actor", "canonical") for part in f.parts): continue
        if "AUTHORIZ" in f.name.upper() or f.name in ("CONSUMED.json", "RESULT.json", "STOP.json"):
            try:
                if EXPECTED_MANIFEST in f.read_text() and "REVIEW" not in f.name.upper(): prior.append(str(f))
            except OSError: pass
assert not prior, prior
record["prior_launches_for_manifest"] = []
# 5. inventories of candidate + every chain node + launch/audit dirs (SOURCE-BEFORE)
dirs = [CAND]
node = json.loads((CAND / "PREDECESSOR.json").read_text())
while node:
    dirs.append(ROOT / node["identity"])
    node = node.get("earlier_chain")
dirs += [ROOT / "analysis/operational-theory-t3-20260917T134947Z", ROOT / "analysis/operational-theory-t4-audit-20260917"]
inventory = {}
for d in dirs:
    assert d.is_dir(), d
    inventory[str(d.relative_to(ROOT))] = {str(f.relative_to(d)): sha(f) for f in sorted(d.rglob("*")) if f.is_file() and not f.is_symlink()}
(STAGE / "SOURCE-BEFORE.json").write_text(json.dumps(inventory, sort_keys=True, indent=2) + "\n")
record["source_before_dirs"] = {k: len(v) for k, v in inventory.items()}
# 6. local launch gate: offline release/chain tests, no cache writes
env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
env.pop("OPENROUTER_API_KEY", None)
cmd = [str(ROOT / "deployment/kaggle/.venv/bin/python"), "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider", "--rootdir", str(STAGE), str(CAND / "test_release.py"), str(CAND / "test_chain.py")]
proc = subprocess.run(cmd, cwd=str(STAGE), env=env, capture_output=True, text=True, timeout=600)
(STAGE / "preflight-tests.txt").write_text(proc.stdout + proc.stderr)
assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-2000:]
record["local_gate"] = {"command": " ".join(cmd[2:]), "returncode": 0, "tail": proc.stdout.strip().splitlines()[-1]}
# 7. tooling
record["sandbox_tools"] = {"bwrap": bool(__import__("shutil").which("bwrap")), "unshare": bool(__import__("shutil").which("unshare"))}
record["credential_present_in_environment"] = bool(os.environ.get("OPENROUTER_API_KEY"))
record["credential_plumbing"] = "ROOT/.env OPENROUTER_API_KEY via python-dotenv at launch (T3 precedent); never printed or persisted"
record["expected"] = {"manifest_sha256": EXPECTED_MANIFEST, "card_sha256": EXPECTED_CARD, "review_sha256": EXPECTED_REVIEW}
record["user_task_sha256"] = sha(STAGE / "USER-TASK.txt")
record["status"] = "GO"
(STAGE / "PREFLIGHT.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
print(json.dumps(record, indent=2, sort_keys=True))
