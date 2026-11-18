"""C7R1 stations 3-4 and the single launch: credential via existing plumbing into the driver subprocess only;
manifest/metadata/review-bound user authorization written before any generation; frozen gates confirmed in-process;
the frozen run_pair.py invoked exactly once as its own CLI; staged pre-launch evidence copied into the fresh output
directory as soon as the driver has claimed it. No retry, no fallback, no prompt or driver alteration."""
import datetime, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT = Path("/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus")
CAND = ROOT / "analysis/operational-theory-t1r7-20260917"
STAGE = Path(sys.argv[1])
sys.path.insert(0, str(CAND))
from identity import verify, sha  # noqa: E402
import run_pair  # noqa: E402
from transport import MODEL, ENDPOINT, exclusive  # noqa: E402

pre = json.loads((STAGE / "PREFLIGHT.json").read_text()); res = json.loads((STAGE / "MODEL-RESOLUTION.json").read_text())
MANIFEST = pre["expected"]["manifest_sha256"]; CARD = pre["expected"]["card_sha256"]
review = CAND / "C6R2-REVIEW.json"; substantive = CAND / "C6R1-REVIEW.json"
assert pre["status"] == "GO" and verify() == MANIFEST and sha(CAND / "EXPERIMENT-CARD.md") == CARD
assert sha(review) == pre["expected"]["review_sha256"] and sha(substantive) == pre["expected"]["substantive_review_sha256"]
assert res["status"] == "RESOLVED" and res["exact_match_count"] == 1 and res["pinned_endpoint_count"] == 1
assert run_pair.metadata_check(STAGE / "endpoints.raw") == res["metadata_sha256"]
assert sha(STAGE / "models.raw") == res["models_sha256"]
assert not (ROOT / "analysis/operational-theory-replacement-claims").exists()

# Station 3: existing credential plumbing (ROOT/.env via python-dotenv). The value goes only into the driver's env.
child_env = {"PATH": "/usr/bin:/bin", "HOME": os.environ.get("HOME", "/tmp"), "LANG": os.environ.get("LANG", "C.UTF-8"), "PYTHONDONTWRITEBYTECODE": "1"}
if os.environ.get("OPENROUTER_API_KEY"):
    child_env["OPENROUTER_API_KEY"] = os.environ["OPENROUTER_API_KEY"]; plumbing = "process environment"
else:
    from dotenv import dotenv_values
    supplied = dotenv_values(ROOT / ".env").get("OPENROUTER_API_KEY")
    if not supplied: raise RuntimeError("existing credential plumbing unavailable; no generation")
    child_env["OPENROUTER_API_KEY"] = supplied; plumbing = "ROOT/.env via python-dotenv"; del supplied
secret_digest = hashlib.sha256(child_env["OPENROUTER_API_KEY"].encode()).hexdigest()[:16]  # for leak scan only; never the value

# Station 4: user authorization, bound to this request and to every hash the frozen gate and the card require.
auth = {
    "role": "user", "decision": "AUTHORIZE_REPLACEMENT_PAIR", "task": "operational-theory-T1R7", "record_label": "C7R1-AUTHORIZATION",
    "authorized_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "authority": "User's explicit request to execute TASK C7R1 exactly: one authorized paid replacement pair after the gate-form fix; preserved USER-TASK.txt",
    "user_task_sha256": pre["user_task_sha256"],
    "candidate": "analysis/operational-theory-t1r7-20260917", "manifest_sha256": MANIFEST, "card_sha256": CARD,
    "review_sha256": pre["expected"]["review_sha256"], "review_path": "analysis/operational-theory-t1r7-20260917/C6R2-REVIEW.json",
    "substantive_review_sha256": pre["expected"]["substantive_review_sha256"], "substantive_review_path": "analysis/operational-theory-t1r7-20260917/C6R1-REVIEW.json",
    "metadata_sha256": res["metadata_sha256"], "models_metadata_sha256": res["models_sha256"], "metadata_fetched_at_utc": res["at_utc"],
    "model_id": MODEL, "canonical_model_slug": res["resolved_model"]["canonical_slug"], "endpoint": ENDPOINT, "quantization": "fp8", "provider": "Reka",
    "prices_per_token_usd": {"prompt": "0.000000214", "completion": "0.00000255"}, "context_length": 262144, "provider_max_completion_tokens": 131072,
    "paired_launches": 1, "order": ["control", "treatment"],
    "requests_per_arm": 6, "requests_total": 12, "completion_tokens_per_request": 4096, "completion_tokens_per_arm": 24576, "completion_tokens_total": 49152,
    "prompt_tokens_per_arm": 600000, "prompt_tokens_total": 1200000, "pair_cost_ceiling_usd": "2", "pair_wall_seconds": 480, "arm_wall_seconds": 240,
    "request_absolute_seconds": 45, "broker_operations_per_arm": 120, "prepared_proposals_per_arm": 2, "executed_game_actions": 0, "format_repair_responses_per_arm": 3,
    "retries": 0, "fallbacks": False, "reasoning": "disabled", "prompt_alteration": False, "manual_repair": False,
    "unknown_usage_policy": "stop further calls; retain unresolved exposure; never zero",
    "blocked_c7_hold": "analysis/operational-theory-replacement-20260917T174546Z-c7-hold (0 generation calls; consumed no pair authorization)",
    "launch_operator_sha256": sha(Path(__file__)), "preflight_sha256": sha(STAGE / "PREFLIGHT.json"), "model_resolution_sha256": sha(STAGE / "MODEL-RESOLUTION.json"),
}
exclusive(STAGE / "C7R1-AUTHORIZATION.json", auth)
gate = run_pair.gates(review, STAGE / "C7R1-AUTHORIZATION.json", STAGE / "endpoints.raw")   # frozen gates, in-process, no side effects
assert gate == (MANIFEST, res["metadata_sha256"]), gate

now = datetime.datetime.now(datetime.timezone.utc)
OUT = ROOT / "analysis" / f"operational-theory-replacement-{now:%Y%m%dT%H%M%SZ}"
assert not OUT.exists()
exclusive(STAGE / "LAUNCH-START.json", {"at_utc": now.isoformat(), "output": str(OUT.relative_to(ROOT)), "authorization_sha256": sha(STAGE / "C7R1-AUTHORIZATION.json"),
                                        "frozen_gates_in_process": "ACCEPTED", "source_verified": verify(), "credential_plumbing": plumbing,
                                        "driver": "analysis/operational-theory-t1r7-20260917/run_pair.py", "driver_sha256": sha(CAND / "run_pair.py"),
                                        "invocation": "frozen CLI in a subprocess; no HERE relocation, no monkeypatch, no argv injection"})
cmd = [str(ROOT / "deployment/kaggle/.venv/bin/python"), "-B", str(CAND / "run_pair.py"), "--execute", "--output", str(OUT),
       "--review", str(review), "--authorization", str(STAGE / "C7R1-AUTHORIZATION.json"), "--metadata", str(STAGE / "endpoints.raw")]
console = (STAGE / "launch-console.txt").open("wb")
proc = subprocess.Popen(cmd, cwd=str(ROOT), env=child_env, stdin=subprocess.DEVNULL, stdout=console, stderr=subprocess.STDOUT)
del child_env
# Incremental preservation: as soon as the driver has claimed the fresh output, copy staged pre-launch evidence beside live/.
copied = False; started = time.monotonic()
STAGED = ["USER-TASK.txt", "PREFLIGHT.json", "GATE-PROBE.json", "SOURCE-BEFORE.json", "preflight-tests.txt", "models.raw", "models.receipt.json",
          "endpoints.raw", "endpoints.receipt.json", "MODEL-RESOLUTION.json", "C7R1-AUTHORIZATION.json", "LAUNCH-START.json"]
while proc.poll() is None:
    if not copied and (OUT / "live/CONSUMED.json").exists():
        (OUT / "operator").mkdir(exist_ok=True)
        for name in STAGED: shutil.copy2(STAGE / name, OUT / name)
        for p in (STAGE / "operator").glob("*.py"): shutil.copy2(p, OUT / "operator" / p.name)
        copied = True
    time.sleep(0.2)
    if time.monotonic() - started > 900: proc.kill(); break
code = proc.wait(); console.close()
raw = (STAGE / "launch-console.txt").read_bytes()
assert secret_digest not in hashlib.sha256(raw).hexdigest() and b"Bearer " not in raw
end = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "exit_code": code, "elapsed_seconds": time.monotonic() - started,
       "output_claimed": (OUT / "live/CONSUMED.json").exists(), "staged_evidence_copied": copied, "source_manifest_after": verify(), "driver_invocations": 1}
exclusive(STAGE / "LAUNCH-END.json", end)
if OUT.exists():
    if not copied:
        (OUT / "operator").mkdir(exist_ok=True)
        for name in STAGED: shutil.copy2(STAGE / name, OUT / name)
        for p in (STAGE / "operator").glob("*.py"): shutil.copy2(p, OUT / "operator" / p.name)
    shutil.copy2(STAGE / "LAUNCH-END.json", OUT / "LAUNCH-END.json"); shutil.copy2(STAGE / "launch-console.txt", OUT / "launch-console.txt")
print(json.dumps(end, indent=2)); print("OUTPUT", OUT)
raise SystemExit(code)
