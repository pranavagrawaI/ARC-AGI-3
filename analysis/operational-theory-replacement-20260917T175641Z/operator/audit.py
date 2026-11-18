"""C7R1 post-run: mechanical scoring from raw artifacts only. Parity/declaration, usage/cost reconciliation, worker
reaping, orphan scan, source unchanged, first-missing-stage localisation with raw pointers, common rubric left
unresolved for independent C8. Writes PARITY/INTEGRITY/TIMESTAMPS/WORKFLOW-REVIEW/FIRST-BOUNDARY/RESULT/REPORT."""
import copy, datetime, json, re, sys
from decimal import Decimal
from pathlib import Path

ROOT = Path("/home/pranav/.superset/worktrees/b420fea5-43e0-4bc6-a3d0-0629350d400b/tall-humerus")
CAND = ROOT / "analysis/operational-theory-t1r7-20260917"
OUT = Path(sys.argv[1])
sys.path.insert(0, str(CAND))
from identity import verify, sha  # noqa: E402
from run_pair import reconcile  # noqa: E402
from transport import encode, digest, MODEL, ENDPOINT  # noqa: E402


def save(name, value):
    (OUT / name).write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


start = json.loads((OUT / "LAUNCH-START.json").read_text()); end = json.loads((OUT / "LAUNCH-END.json").read_text())
auth = json.loads((OUT / "C7R1-AUTHORIZATION.json").read_text()); pre = json.loads((OUT / "PREFLIGHT.json").read_text())
MANIFEST = auth["manifest_sha256"]
assert end["exit_code"] == 0 and end["driver_invocations"] == 1
consumed = json.loads((OUT / "live/CONSUMED.json").read_text())
claim_path = ROOT / "analysis/operational-theory-replacement-claims" / (MANIFEST + ".json")
claim = json.loads(claim_path.read_text())
assert consumed == claim and claim["output"] == str(OUT)
assert claim["review_sha256"] == auth["review_sha256"] == sha(CAND / "C6R2-REVIEW.json")
assert claim["authorization_sha256"] == sha(OUT / "C7R1-AUTHORIZATION.json") == start["authorization_sha256"]
assert claim["metadata_sha256"] == auth["metadata_sha256"] == sha(OUT / "endpoints.raw")
driver_result = json.loads((OUT / "live/RESULT.json").read_text()); assert driver_result["status"] == "pair_completed"
assert not (OUT / "live/STOP.json").exists()
ledger = json.loads((OUT / "live/ledger.json").read_text()); account = reconcile(OUT / "live/ledger.json")
assert account["unresolved_requests"] == 0 and account["unresolved_exposure_usd"] == "0" and not ledger["stopped"]
assert json.loads((OUT / "live/ACCOUNTING.json").read_text()) == account
results = {a: json.loads((OUT / "live" / a / "RESULT.json").read_text()) for a in ("control", "treatment")}
first = {}; all_nonmessage = []; timestamps = []; per_call = {}; cached = {}
for arm in results:
    assert not (OUT / "live" / arm / "FAILURE.json").exists()
    paths = sorted((OUT / "live" / f"{arm}-raw").glob("*.request.json"))
    assert len(paths) == ledger["arms"][arm]["requests"] <= 6
    assert len(list((OUT / "live" / f"{arm}-raw").glob("*.receipt.json"))) == len(paths) == len(list((OUT / "live" / f"{arm}-raw").glob("*.response.raw")))
    per_call[arm] = []; cached[arm] = 0
    for p in paths:
        stem = p.name.split(".")[0]; r = json.loads(p.read_bytes()); receipt = json.loads(p.with_name(stem + ".receipt.json").read_text()); response = json.loads(p.with_name(stem + ".response.raw").read_bytes())
        assert sha(p) == receipt["request_sha256"] and sha(p.with_name(stem + ".response.raw")) == receipt["response_sha256"]
        assert receipt["arm"] == arm and receipt["ordinal"] == int(stem) and receipt["status"] == "receipted" and receipt.get("stop") is None
        assert response["usage"]["prompt_tokens"] == receipt["usage"]["prompt_tokens"] and response["usage"]["completion_tokens"] == receipt["usage"]["completion_tokens"]
        assert Decimal(str(response["usage"]["cost"])) == Decimal(receipt["actual_cost_usd"])
        assert response["model"] in (MODEL, auth["canonical_model_slug"]) and response["provider"] == "Reka"
        meta = response["openrouter_metadata"]; sel = [e for e in meta["endpoints"]["available"] if e.get("selected")]
        assert meta["attempt"] == 1 and meta["requested"] == MODEL and meta["strategy"] == "direct" and len(sel) == 1 and sel[0]["provider"] == "Reka"
        assert receipt["reasoning_tokens"] == 0 and receipt["reasoning_payload_present"] is False and receipt["finish_reason"] in ("tool_calls", "stop")
        assert r["model"] == MODEL and r["provider"]["only"] == [ENDPOINT] and r["provider"]["allow_fallbacks"] is False and r["provider"]["quantizations"] == ["fp8"]
        assert r["reasoning"] == {"enabled": False, "exclude": False} and r["max_tokens"] == 4096 and r["stream"] is False
        assert "Authorization" not in r and "headers" not in r
        assert receipt["latency_seconds"] <= 45
        all_nonmessage.append({k: v for k, v in r.items() if k != "messages"})
        if stem == "001": first[arm] = copy.deepcopy(r)
        cached[arm] += (response["usage"].get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        msg = response["choices"][0]["message"]; calls = msg.get("tool_calls") or []
        per_call[arm].append({"ordinal": int(stem), "prompt_tokens": receipt["usage"]["prompt_tokens"], "completion_tokens": receipt["usage"]["completion_tokens"], "cost_usd": receipt["actual_cost_usd"], "latency_seconds": round(receipt["latency_seconds"], 3), "finish_reason": receipt["finish_reason"], "tool_calls": [c["function"]["name"] for c in calls], "content_chars": len(msg.get("content") or "")})
        for f in (p, p.with_name(stem + ".response.raw"), p.with_name(stem + ".receipt.json")):
            timestamps.append({"path": str(f.relative_to(OUT)), "filesystem_persisted_mtime_ns": f.stat().st_mtime_ns})
    sup = json.loads((OUT / "live" / arm / "SUPERVISION.json").read_text()); assert sup["reaped"] is True
    assert len(json.loads((OUT / "live" / arm / "EVIDENCE.json").read_text())) == 1   # only transition 1 ever revealed
    assert results[arm]["proposals"] == [] and results[arm]["metrics"]["construction_completion"] is False
    hostops = [json.loads(l) for l in (OUT / "live" / arm / "host-events.jsonl").read_text().splitlines()]
    assert len(hostops) <= 120 and sum(e["op"] == "model" for e in hostops) == len(paths) and not any(e["op"] == "proposed" for e in hostops)
    per_call[arm + "_host_ops"] = {"total": len(hostops), "by_op": {op: sum(e["op"] == op for e in hostops) for op in sorted({e["op"] for e in hostops})}}
# Parity: initial request equal after removing only the declared treatment block; all 12 nonmessage parameter sets identical
shared = first["treatment"]; marker = "\n\n<operational_cycle>"; content = shared["messages"][0]["content"]
assert content.count(marker) == 1 and content.endswith("</operational_cycle>"); shared["messages"][0]["content"] = content.split(marker)[0]
assert shared == first["control"] and all(x == all_nonmessage[0] for x in all_nonmessage)
shared_sha = digest(encode(shared)); assert ledger["arms"]["control"]["initial_shared_sha256"] == ledger["arms"]["treatment"]["initial_shared_sha256"]
assert sha(OUT / "live/control-raw/001.request.json") != sha(OUT / "live/treatment-raw/001.request.json")
treatment_block_chars = len(content) - len(shared["messages"][0]["content"])
save("PARITY.json", {"initial_request_equal_after_removing_only_treatment_block": True, "shared_request_sha256_recomputed": shared_sha, "ledger_initial_shared_sha256": ledger["arms"]["control"]["initial_shared_sha256"], "declared_treatment_block": "<operational_cycle>...</operational_cycle> appended to system message", "treatment_block_chars": treatment_block_chars, "all_12_nonmessage_parameter_sets_identical": True, "pinned_routing_on_every_request": {"model": MODEL, "provider.only": [ENDPOINT], "allow_fallbacks": False, "quantizations": ["fp8"], "reasoning": {"enabled": False, "exclude": False}, "max_tokens": 4096}, "later_model_authored_messages_expected_to_differ": True})
save("TIMESTAMPS.json", {"request_timing": "Per-request receipt latency_seconds plus filesystem persistence timestamps; UTC launch start/end in LAUNCH-START/LAUNCH-END. Filesystem timestamps are not exact dispatch times.", "launch_start_utc": start["at_utc"], "launch_end_utc": end["at_utc"], "artifacts": timestamps})
# Source unchanged; workers reaped; no orphan bwrap
counts = {}
for name, expected in json.loads((OUT / "SOURCE-BEFORE.json").read_text()).items():
    folder = ROOT / name; actual = {str(f.relative_to(folder)): sha(f) for f in sorted(folder.rglob("*")) if f.is_file() and not f.is_symlink()}
    assert actual == expected, name; counts[name] = len(actual)
assert verify() == MANIFEST == end["source_manifest_after"]
orphans = []
for proc in Path("/proc").iterdir():
    if not proc.name.isdigit(): continue
    try:
        args = (proc / "cmdline").read_bytes()
        if b"bwrap\x00" in args and str(OUT).encode() in args: orphans.append(int(proc.name))
    except OSError: pass
assert not orphans
save("INTEGRITY.json", {"source_manifest_sha256": verify(), "all_candidate_predecessor_launch_audit_and_c7_hold_inventory_files_unchanged": counts, "matching_bwrap_processes": orphans, "both_workers_reaped": True, "receipt_counts_match_ledger": True, "raw_usage_costs_match_receipts": True, "request_and_response_hashes_match_receipts": True, "every_response_routed_reka_direct_attempt_1": True, "reasoning_tokens_all_zero": True, "secrets_in_request_bodies_persisted": False, "one_shot_claim": {"path": str(claim_path.relative_to(ROOT)), "sha256": sha(claim_path)}, "consumed_equals_claim": True, "stop_json": False, "failure_json": False, "per_call": per_call, "cached_prompt_tokens_reported": cached})
# First missing workflow stage, localised mechanically from the raw request bodies (PythonOutput blocks the model saw)
def outputs(arm):
    r = json.loads((OUT / "live" / f"{arm}-raw/006.request.json").read_bytes())
    rows = []
    for i, m in enumerate(r["messages"]):
        c = m.get("content") or ""
        if m["role"] == "user" and "PythonOutput(" in c:
            tag = re.search(r'<sys tag="(\d+)">', c); status = re.search(r"ResultStatus\.(\w+)", c)
            err = re.search(r"\n(\w+Error: .*?)(?:'''|\n)", c, re.S) or re.search(r"(CELL_STATE_REPAIR: [^\n]*?)(?:'''|\n)", c)
            rows.append({"message_index": i, "sys_tag": tag.group(1) if tag else None, "status": status.group(1) if status else None, "error": err.group(1) if err else None})
    return rows
cells = {arm: outputs(arm) for arm in results}
boundary = {
    "stage_reached": {arm: "observe/analyze (native grid inspection); no reconcile, no theory, no prepared proposal, no submit_actions" for arm in results},
    "cell_outcomes_seen_by_model": cells,
    "raw_pointers": {arm: f"live/{arm}-raw/006.request.json (messages[*] role=user containing PythonOutput; sha256 {sha(OUT / 'live' / f'{arm}-raw/006.request.json')})" for arm in results},
    "mechanical_observation": "In both arms every non-prefill cell that touched the grid through the facade either errored or was refused: `self.grid_array(...)` returns a PublicObject wrapper (cell_state._sanitize wraps any non-primitive, including numpy.ndarray), so `g == c` is a plain bool (np.where -> 'Calling nonzero on 0d arrays'), `int(element)` and `np.asarray(g, dtype=int)` raise TypeError('... not PublicObject'), and `(g == c).any` raises AttributeError('bool'). `self.model_grid()` and `self.trajectory()` are advertised by the common system prompt/skill but are not in ActorFacade's approved set -> AttributeError('Actor capability is not public'); control call 6 was refused by CELL_STATE_REPAIR (private runtime name). Only grid_summary()/observations() and the treatment's workflow_skill()/operational_cycle('read') succeeded.",
    "shared_across_arms": True,
    "candidate_source_pointers": ["analysis/operational-theory-t1r7-20260917/cell_state.py: ActorFacade.__getattribute__ approved set; _sanitize -> PublicObject", "analysis/operational-theory-t1r7-20260917/actor.py (facade installation)"],
    "why_qualification_missed_it": "Local qualification and the exact T3 replay exercised schema reads, underscore bindings and facade traversal refusals; no fixture executed numpy arithmetic on a facade-returned grid or called model_grid()/trajectory() through the facade. This run is the first live exercise of the facade against the Actor's own advertised numpy workflow.",
    "classification_offered_to_c8": "Not one of the card's listed invalidity conditions (both arms received the identical declared namespace contract; prefix/tool surface/routing/accounting parity hold). It is a shared construction-machinery boundary that ended both arms before the first prospective proposal; card result map 'Both incomplete: preserve first missing workflow stage, with no follow-up spend.' Independent C8 decides whether the facade contradicting the common prompt's advertised capabilities makes the comparison uninterpretable rather than merely both-incomplete.",
    "not_a_gameplay_or_efficacy_finding": True,
}
save("FIRST-BOUNDARY.json", boundary)
# Common rubric: unresolved/absent only; do not score treatment schema metrics as a comparative effect
rubric = {}
for arm in results:
    asset = OUT / "live" / arm / "actor/team_nemo/shared/knowledge"
    refs = [{"path": str(f.relative_to(OUT)), "sha256": sha(f)} for f in sorted(asset.glob("*.md"))] if asset.exists() else []
    canon = OUT / "live" / arm / "canonical"
    canonical = {str(f.relative_to(OUT)): sha(f) for f in sorted(canon.rglob("*")) if f.is_file()} if canon.exists() else None
    rubric[arm] = {"entered_observation": "present in executed analysis (grid_summary/observations output); canonical entry absent" if arm == "treatment" else "present in executed analysis (grid_summary/observations output)", "retained_claim_with_tested_scope_and_open_conditions": "unresolved; inspect native notes" if refs else "absent published asset", "explicit_reconciliation": "absent", "decision_relevant_uncertainty_or_procedure": "unresolved; inspect native notes" if refs else "absent published asset", "declared_mode": "absent explicit accepted mode", "complete_prospective_record": "absent; zero proposals", "post_observation_revision": "absent; second recorded transition not revealed", "reusable_next_policy_without_transcript": "unresolved; notes alone not scored sufficient" if refs else "absent published asset", "raw_native_assets": refs, "canonical_files": canonical, "common_workflow_completion": False, "counterevidence_opportunity": "second recorded continuation not reached", "review_status": "independent C8 adjudication pending"}
save("WORKFLOW-REVIEW.json", rubric)
tokens = {"input": sum(a["prompt_tokens"] for a in account["arms"].values()), "output": sum(a["completion_tokens"] for a in account["arms"].values())}; tokens["total"] = sum(tokens.values())
c, t = results["control"]["metrics"], results["treatment"]["metrics"]
score = {
    "task": "C7R1", "status": "COMPLETE", "operational_status": "one bounded pair completed; both constructions incomplete at the six-request cap; zero prospective proposals in either arm",
    "model": MODEL, "endpoint": ENDPOINT, "provider": "Reka", "quantization": "fp8", "canonical_model_slug": auth["canonical_model_slug"],
    "launch_dir": str(OUT.relative_to(ROOT)), "authorization_sha256": start["authorization_sha256"], "review_sha256": auth["review_sha256"], "substantive_review_sha256": auth["substantive_review_sha256"],
    "manifest_sha256": MANIFEST, "card_sha256": auth["card_sha256"], "metadata_sha256": auth["metadata_sha256"], "models_metadata_sha256": auth["models_metadata_sha256"], "metadata_get_requests": 2,
    "requests": {a: account["arms"][a]["requests"] for a in results} | {"total": account["requests"]},
    "usage": {"tokens": tokens, "per_arm": {a: {"input": account["arms"][a]["prompt_tokens"], "output": account["arms"][a]["completion_tokens"]} for a in results}, "usd": account["total_cost_usd"], "per_arm_usd": {a: str(sum((Decimal(x["cost_usd"]) for x in per_call[a]), Decimal(0))) for a in results}, "unresolved_requests": 0, "unresolved_exposure_usd": "0", "reasoning_tokens": sum(a["reasoning_tokens"] for a in account["arms"].values()), "cost_ceiling_usd": "2"},
    "elapsed": {"pair_seconds": driver_result["elapsed_seconds"], "control_seconds": results["control"]["elapsed_seconds"], "treatment_seconds": results["treatment"]["elapsed_seconds"], "caps": {"arm": 240, "pair": 480}},
    "control_metrics": c, "treatment_metrics": t, "arms": results,
    "delta": "Both arms: 0 accepted prospective proposals, construction_completion=false, common workflow incomplete at the observe/analyze stage; treatment canonical cycle idle (0 theory versions, 0 decisions). No demonstrated construction advantage in either direction; no efficacy inference.",
    "validity": {"invalidity_conditions_checked": {"unreported_intervention": False, "prefix_or_tool_surface_or_system_prefix_differs": False, "future_or_other_arm_evidence_leak": False, "frozen_source_or_card_changed": False, "canonical_binding_failure": "not reached", "provider_identity_reasoning_accounting_proof_missing": False, "undeclared_request": False, "cap_exceeded": False, "output_promoted_after_failed_worker": False, "containment_absent": False, "recorded_continuation_scored_as_outcome": False, "fixture_outcome_as_evidence": False, "missing_independent_review_or_new_release": False}, "operator_classification": "VALID-AS-EXECUTED, result-map row 'both incomplete'; first missing workflow stage preserved in FIRST-BOUNDARY.json", "boundary_finding_for_c8": "shared ActorFacade/PublicObject boundary defeats the Actor's advertised numpy grid workflow and hides model_grid()/trajectory(); see FIRST-BOUNDARY.json", "final_classification_owner": "independent C8 audit"},
    "source_identity_unchanged": True, "provider_retries": 0, "reruns": 0, "manual_repairs": 0, "prompt_alterations": 0, "gameplay": "not run", "executed_game_actions": 0,
    "scientific_scope": "recorded-prefix construction completion only; distinguishes construction machinery from gameplay competence; no gameplay, truth or efficacy claim",
    "c7_hold_preserved": pre["c7_hold"], "c8_audit": "pending", "at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
save("RESULT.json", score)
rows = "\n".join(f"| {a.capitalize()} | {account['arms'][a]['requests']} | {account['arms'][a]['prompt_tokens']:,} | {account['arms'][a]['completion_tokens']:,} | {score['usage']['per_arm_usd'][a]} | 0 | No |" for a in results)
(OUT / "REPORT.md").write_text(f"""# C7R1: the one authorized replacement pair ran; both constructions incomplete

Exact OpenRouter model `{MODEL}`, singleton endpoint `{ENDPOINT}` (Reka, fp8). Fresh public metadata
(2 unauthenticated GETs, `endpoints.raw` sha256 `{auth['metadata_sha256']}`) matched the pinned
provider/model/quantization/context/max-completion/prices/parameters with zero drift, and the frozen
`run_pair.gates()` accepted C6R2-REVIEW.json (`{auth['review_sha256'][:16]}…`, task
`operational-theory-T1R7`) plus the manifest-bound user authorization written before dispatch
(`{start['authorization_sha256']}`). The frozen driver was invoked once, as its own CLI, in a
subprocess; no HERE relocation, monkeypatch, retry, fallback, prompt alteration or manual repair.

| Arm | Requests | Input tokens | Output tokens | USD | Accepted proposals | Construction complete |
|---|---:|---:|---:|---:|---:|---|
{rows}

Total: **{tokens['input']:,} input + {tokens['output']:,} output = {tokens['total']:,} tokens**, **${account['total_cost_usd']}** of the $2 ceiling.
All twelve receipts are resolved (request/response hashes, usage and `usage.cost` match the raw bodies);
reasoning tokens and unresolved exposure are zero; every response routed `Reka` direct on attempt 1.
The pair took {driver_result['elapsed_seconds']:.1f} s ({results['control']['elapsed_seconds']:.1f} s + {results['treatment']['elapsed_seconds']:.1f} s of the 240 s/arm caps). Each arm ended at its
six-request ceiling with zero `submit_actions`, so the recorded second transition was never revealed.

What the raw bodies show (FIRST-BOUNDARY.json, pointers into `live/*-raw/006.request.json`): both
arms spent their budget trying to analyze the grid with numpy, as the common system prompt instructs,
and both hit the same wall. `self.grid_array()` comes back through the facade as an opaque
`PublicObject`, so `g == c` is a bare bool, `int(cell)` and `np.asarray(g, dtype=int)` raise
TypeError, and `.any` fails; `self.model_grid()` and `self.trajectory()`, both advertised in the
shared prompt, are not in the facade's approved set ("Actor capability is not public"). Control saw
4 errored cells and one CELL_STATE_REPAIR refusal; treatment read the workflow skill and
`operational_cycle('read')` (its canonical cycle stayed `idle`, 0 theory versions) and then errored on
the same grid path twice. This boundary is identical in both arms, so it is not one of the card's
invalidity conditions; it is the first missing workflow stage under the result-map row "both
incomplete". Whether a facade that contradicts the Actor's own advertised capabilities makes the
comparison uninterpretable is C8's call, not the operator's.

`PARITY.json`: exact first-request equality after removing only the declared `<operational_cycle>`
block ({treatment_block_chars} chars), identical non-message parameters on all twelve requests.
`INTEGRITY.json`: receipts, both workers reaped, no matching bwrap processes, every candidate /
predecessor / T3 / T4 / C7-HOLD inventory file unchanged, manifest `{MANIFEST[:16]}…` after.
`WORKFLOW-REVIEW.json` keeps the eight-item common rubric unresolved/absent for independent C8; the
treatment-only schema metrics are not a comparative statistic. No gameplay was executed; nothing here
speaks to gameplay competence or efficacy. No further launch or replication is authorized.
""")
print(json.dumps({k: score[k] for k in ("status", "requests", "usage", "elapsed", "delta")}, indent=2))
