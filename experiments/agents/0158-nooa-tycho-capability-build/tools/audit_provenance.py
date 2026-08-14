"""Fail closed when source pins, snapshots, diffs, or frozen boundaries drift."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from host.contract import MODEL_CONTRACT, validate_model_contract
from refresh_adaptation_patches import PAIRS, diff_bytes


def file_sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def git(checkout: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(checkout), *args], text=True).strip()


def main() -> None:
    pins = json.loads((ROOT / "source-pins.json").read_text(encoding="utf-8"))
    baseline_snapshot = ROOT / "third_party/source-snapshots/arc-agi-3/0002-nooa-llm.py"
    baseline_source = next(iter(pins["baseline"]["sources"]))
    baseline_bytes = subprocess.check_output(
        ["git", "-C", str(ROOT.parents[2]), "show", f"{pins['baseline']['commit']}:{baseline_source}"]
    )
    if (
        sha256(baseline_bytes).hexdigest() != pins["baseline"]["sources"][baseline_source]
        or baseline_snapshot.read_bytes() != baseline_bytes
    ):
        raise SystemExit("baseline 0002 provider snapshot differs from its frozen source")
    checkouts = {
        "nooa": Path("/home/hipra/projects/nooa"),
        "tycho": Path("/home/hipra/projects/tycho"),
    }
    snapshot_roots = {
        "nooa": ROOT / "third_party/source-snapshots/nooa",
        "tycho": ROOT / "third_party/source-snapshots/tycho",
    }
    source_names = {
        "nooa": {
            "examples/arc_agi_3/solver_agent.py": "solver_agent.py",
            "examples/arc_agi_3/launcher.py": "launcher.py",
            "examples/arc_agi_3/arc_llm.py": "arc_llm.py",
        },
        "tycho": {
            "tycho/workspace/workspace.py": "workspace.py",
            "tycho/workspace/wmlib_template.py": "wmlib_template.py",
            "tycho/workspace/version_store.py": "version_store.py",
        },
    }
    verified_sources = 1
    for project, checkout in checkouts.items():
        if git(checkout, "rev-parse", "HEAD") != pins[project]["commit"]:
            raise SystemExit(f"{project} checkout is not at the frozen commit")
        if subprocess.run(["git", "-C", str(checkout), "diff", "--quiet"], check=False).returncode:
            raise SystemExit(f"{project} checkout has tracked modifications")
        for source, expected in pins[project]["sources"].items():
            snapshot = snapshot_roots[project] / source_names[project][source]
            live = checkout / source
            if file_sha(snapshot) != expected or file_sha(live) != expected:
                raise SystemExit(f"source hash mismatch: {project}:{source}")
            if snapshot.read_bytes() != live.read_bytes():
                raise SystemExit(f"snapshot differs from pinned source: {project}:{source}")
            verified_sources += 1

    manifest = json.loads((ROOT / "provenance/adaptation-patches.json").read_text())
    manifest_by_name = {item["patch"]: item for item in manifest["patches"]}
    for name, pair in PAIRS.items():
        patch = ROOT / "provenance" / name
        expected_payload = diff_bytes(*pair)
        if patch.read_bytes() != expected_payload:
            raise SystemExit(f"adaptation patch is stale: {name}")
        if manifest_by_name[name]["sha256"] != file_sha(patch):
            raise SystemExit(f"adaptation patch manifest mismatch: {name}")

    contract_file = json.loads((ROOT / "model-contract.json").read_text())
    if validate_model_contract(contract_file) != MODEL_CONTRACT:
        raise SystemExit("model contract is not frozen")
    card = ROOT.parents[1] / "cards/0158-gemma-ls20-grounding-capability-build-v1.md"
    if card.stat().st_size > 1024:
        raise SystemExit("live card exceeds 1024 bytes")
    card_manifest = json.loads((ROOT / "frozen-card.json").read_text())
    if (
        card_manifest.get("sha256") != file_sha(card)
        or card_manifest.get("bytes") != card.stat().st_size
        or card_manifest.get("execution_authority") is not False
    ):
        raise SystemExit("frozen card manifest drifted or grants execution authority")
    live_authorities = (
        ("0158-gemma-ls20-live-v2.md", "0158-ls20-live-v1.json"),
        ("0158-gemma-ls20-live-retry-v3.md", "0158-ls20-live-v2.json"),
        ("0158-gemma-ls20-live-8k-v4.md", "0158-ls20-live-v3.json"),
        ("0158-gemma-ls20-12iter-v5.md", "0158-ls20-live-v4.json"),
        ("0158-gemma-ls20-mask-schema-v6.md", "0158-ls20-live-v5.json"),
        ("0158-gemma-ls20-mask-schema-retry-v7.md", "0158-ls20-live-v6.json"),
        ("0158-gemma-ls20-mask-schema-retry-v8.md", "0158-ls20-live-v7.json"),
        ("0158-gemma-ls20-coreweave-v9.md", "0158-ls20-live-v8.json"),
        ("0158-gemma-ls20-complete-claim-v10.md", "0158-ls20-live-v9.json"),
        ("0158-gemma-ls20-429-backoff-v11.md", "0158-ls20-live-v10.json"),
        ("0158-gemma-ls20-codeact-repair-v12.md", "0158-ls20-live-v11.json"),
        ("0158-gemma-ls20-inline-result-v13.md", "0158-ls20-live-v12.json"),
        ("0158-gemma-ls20-required-tool-v14.md", "0158-ls20-live-v13.json"),
        ("0158-qwen36-ls20-model-switch-v15.md", "0158-ls20-live-v14.json"),
        ("0158-qwen36-exact-effect-interface-v16.md", "0158-ls20-live-v15.json"),
        ("0158-qwen36-semantic-investigator-interface-v17.md", "0158-ls20-live-v16.json"),
        ("0158-qwen36-autonomous-44-action-v18.md", "0158-ls20-live-v17.json"),
        ("0158-qwen36-autonomous-44-action-v19.md", "0158-ls20-live-v18.json"),
        ("0158-qwen36-autonomous-44-action-v20.md", "0158-ls20-live-v19.json"),
    )
    for card_name, approval_name in live_authorities:
        live_card = ROOT.parents[1] / "cards" / card_name
        approval_path = ROOT.parents[1] / "approvals" / approval_name
        consumption_path = approval_path.with_name(f"{approval_path.stem}.consumed.json")
        approval = json.loads(approval_path.read_text(encoding="utf-8"))
        if live_card.stat().st_size > 1024 or approval.get("approved_card_sha256") != file_sha(live_card):
            raise SystemExit("live card is oversized or not bound to its separate approval")
        if approval.get("execution_authority") is not True or approval.get("approved_action_cap") != 44:
            raise SystemExit("separate live approval does not grant the frozen 44-action scope")
        if not consumption_path.exists():
            raise SystemExit("one-run live approval has no consumption record")
        consumption = json.loads(consumption_path.read_text(encoding="utf-8"))
        if (
            consumption.get("approval_sha256") != file_sha(approval_path)
            or consumption.get("card_sha256") != file_sha(live_card)
            or consumption.get("resumable") is not False
        ):
            raise SystemExit("live approval consumption record is not bound to its card and approval")

    forbidden_module_stems = {"memory", "world_model", "outcome", "plan", "planner", "builder"}
    mounted_code = [*sorted((ROOT / "host").glob("*.py")), *sorted((ROOT / "nooa-runtime").glob("*.py"))]
    if any(path.stem in forbidden_module_stems for path in mounted_code):
        raise SystemExit("advanced capability module exists in mounted code")
    forbidden_definitions = {"write_helper", "load_helpers", "world_model", "outcome", "plan", "builder", "submit_many"}
    for path in mounted_code:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        definitions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        overlap = definitions & forbidden_definitions
        if overlap:
            raise SystemExit(f"advanced capability definitions in {path.name}: {sorted(overlap)}")

    license_path = ROOT / "THIRD_PARTY_LICENSES/Apache-2.0.txt"
    if file_sha(license_path) != "f855dbf982294c84e40f2c6c96a4da2a237834b9aa828239228b1589014b741e":
        raise SystemExit("Apache-2.0 license copy drifted")
    print(
        json.dumps(
            {
                "status": "pass",
                "verified_sources": verified_sources,
                "verified_patches": len(PAIRS),
                "contract_words_max": 400,
                "card_bytes": card.stat().st_size,
                "advanced_capability_modules": 0,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
