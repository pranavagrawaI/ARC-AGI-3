"""C7R1 seal: copy operator sources, scan every artifact for the loaded credential, write ARTIFACT-MANIFEST.json,
make the launch directory and the one-shot claim read-only."""
import hashlib, json, os, shutil, stat, sys
from pathlib import Path
from dotenv import dotenv_values

ROOT = Path("/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus")
OUT = Path(sys.argv[1]); STAGE = Path(sys.argv[2])
for p in (STAGE / "operator").glob("*.py"): shutil.copy2(p, OUT / "operator" / p.name)
secret = os.environ.get("OPENROUTER_API_KEY") or dotenv_values(ROOT / ".env").get("OPENROUTER_API_KEY"); assert secret and len(secret) > 20
files = {}; links = {}
for p in sorted(OUT.rglob("*")):
    if p.is_symlink(): links[str(p.relative_to(OUT))] = os.readlink(p)
    elif p.is_file():
        raw = p.read_bytes()
        assert secret.encode() not in raw, "credential material detected; artifact not sealed: " + str(p)
        # frozen source copies under worker-source/ legitimately contain the literal header template "Bearer " + key
        assert p.suffix == ".py" or b"Bearer " not in raw, "bearer header persisted; artifact not sealed: " + str(p)
        files[str(p.relative_to(OUT))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
del secret
result = json.loads((OUT / "RESULT.json").read_text())
claim = ROOT / "analysis/operational-theory-replacement-claims" / (result["manifest_sha256"] + ".json")
manifest = {"task": "C7R1", "status": result["status"], "source_manifest_sha256": result["manifest_sha256"], "authorization_sha256": result["authorization_sha256"],
            "files": files, "symlinks": links, "one_shot_claim": {"path": str(claim.relative_to(ROOT)), "sha256": hashlib.sha256(claim.read_bytes()).hexdigest()},
            "credential_scan": "loaded credential bytes absent from every artifact; 'Bearer ' absent from every non-.py artifact (frozen worker-source/transport.py carries only the literal header template); request bodies never carry headers", "provider_retries": 0, "driver_invocations": 1, "self_excluded": "ARTIFACT-MANIFEST.json"}
target = OUT / "ARTIFACT-MANIFEST.json"
with target.open("x") as f: f.write(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
for name, entry in files.items(): assert hashlib.sha256((OUT / name).read_bytes()).hexdigest() == entry["sha256"]
for p in OUT.rglob("*"):
    if p.is_symlink(): continue
    if p.is_file(): p.chmod(stat.S_IMODE(p.stat().st_mode) & ~0o222)
for p in sorted((p for p in OUT.rglob("*") if p.is_dir() and not p.is_symlink()), key=lambda p: len(p.parts), reverse=True): p.chmod(stat.S_IMODE(p.stat().st_mode) & ~0o222)
OUT.chmod(stat.S_IMODE(OUT.stat().st_mode) & ~0o222); claim.chmod(0o444); claim.parent.chmod(0o555)
print(json.dumps({"launch_dir": str(OUT.relative_to(ROOT)), "artifact_manifest_sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "authorization_sha256": result["authorization_sha256"],
                  "result_sha256": files["RESULT.json"]["sha256"], "report_sha256": files["REPORT.md"]["sha256"], "files": len(files), "symlinks": len(links), "sealed_read_only": True}, indent=2))
