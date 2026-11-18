#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from constructor import construction_messages
from real_trace import load_case
from sandbox_eval import evaluate_candidate
from request_contract import validate_request

HERE = Path(__file__).resolve().parent
REFERENCE = Path("/home/pranav/projects/arc-agi-3-reference/model_construction_probe")


def run(command: list[str], cwd: Path) -> dict:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    checks = []
    checks.append({"name": "reference_sha256", **run(["sha256sum", "-c", "SHA256SUMS"], REFERENCE)})
    checks.append({
        "name": "reference_tests",
        **run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], REFERENCE),
    })
    checks.append({
        "name": "integration_tests",
        **run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], HERE),
    })
    checks.append({"name": "dry_run", **run([sys.executable, "run.py"], HERE)})

    case = load_case()
    incumbent = evaluate_candidate(case.legacy_incumbent_source(), case, include_suffix=True)
    witness_source = (HERE / "tests/witness_candidate.py").read_text()
    witness = evaluate_candidate(witness_source, case, include_suffix=True)
    prompt = json.dumps(construction_messages(case, None), sort_keys=True)
    semantic = {
        "incumbent_visible_passed": incumbent["visible"]["status"] == "passed",
        "incumbent_suffix_compatible_depth": incumbent["suffix"]["compatible_depth"] == 2,
        "incumbent_suffix_first_contradiction": incumbent["suffix"].get("first_contradiction_step") == 8,
        "witness_visible_passed": witness["visible"]["status"] == "passed",
        "witness_suffix_passed": witness["suffix"]["status"] == "passed",
        "prompt_excludes_suffix_hash": case.sealed_suffix_sha256 not in prompt,
        "prompt_excludes_suffix_steps": all(
            token not in prompt for token in ('"step": 6', '"step":6', '"step": 7', '"step":7', '"step": 8', '"step":8')
        ),
        "incumbent_digest_reconciles": hashlib.sha256(case.incumbent_source.encode()).hexdigest() == case.incumbent_sha256,
    }
    for manifest, sibling in [('BASELINE-SHA256.json','model-construction-real-prefix-20260913'),('V2-SHA256.json','model-construction-real-prefix-v2-20260913'),('V3-SHA256.json','model-construction-real-prefix-v3-20260913')]:
        semantic[manifest+'_unchanged'] = all(hashlib.sha256((HERE.parent/sibling/n).read_bytes()).hexdigest()==h for n,h in json.loads((HERE/manifest).read_text()).items())
    selection=json.loads((HERE/'PROVIDER-SELECTION.json').read_text())
    sys.path.insert(0,str(HERE/'tests'))
    from test_contract import first_request
    from constructor import ConstructionLimits
    contracts=[]
    for phase,limits in [('explore',None),('commit',ConstructionLimits(max_model_requests=2))]:
        body=first_request(limits)
        contracts.append({'phase':phase,'request':body,'validation':validate_request(body)})
    (HERE/'REQUEST-PREFLIGHT.json').write_text(json.dumps({'paid_calls':0,'cases':contracts},indent=2)+'\n')
    semantic['complete_request_contract_all_phases']=all(c['validation']['passed'] for c in contracts)
    endpoint=selection['selected']
    semantic['verified_provider_routing'] = endpoint in json.loads((HERE/'provider-endpoints.json').read_text())['data']['endpoints'] and endpoint['model_id']=='qwen/qwen3.8-27b' and endpoint['status']==0 and endpoint['supports_tool_choice']['required'] and {'tools','reasoning'}.issubset(endpoint['supported_parameters']) and selection['routing']=={'only':[endpoint['tag']],'order':[endpoint['tag']],'allow_fallbacks':False,'require_parameters':True,'max_price':{'prompt':1,'completion':10}}
    passed = all(check["exit_code"] == 0 for check in checks) and all(semantic.values())
    record = {
        "schema": "model-construction-prelaunch-v1",
        "passed": passed,
        "subject_sha256": {str(p.relative_to(HERE)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*HERE.glob("*.py"), *HERE.glob("tests/*.py"), *HERE.glob('docs-*.txt'), HERE/'REQUEST-CONTRACT.md', HERE/'REQUEST-PREFLIGHT.json', HERE/'model-metadata.json', HERE/'V3-SHA256.json', HERE/"EXPERIMENT-CARD.md", HERE/"PROVIDER-SELECTION.json", HERE/"provider-endpoints.json", HERE/"V2-SHA256.json", HERE/"BASELINE-SHA256.json", HERE/"PERFORMANCE.json"]},
        "reference_path": str(REFERENCE),
        "reference_manifest_sha256": hashlib.sha256((REFERENCE / "SHA256SUMS").read_bytes()).hexdigest(),
        "case": {
            "frontier": case.frontier,
            "incumbent_sha256": case.incumbent_sha256,
            "sealed_suffix_sha256": case.sealed_suffix_sha256,
        },
        "semantic_checks": semantic,
        "incumbent_suffix": incumbent["suffix"],
        "process_checks": checks,
    }
    (HERE / "PRELAUNCH.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"passed": passed, "checks": len(checks) + len(semantic)}, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
