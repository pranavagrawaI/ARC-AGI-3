#!/usr/bin/env python3
"""Fail-closed total gate for one immutable live package."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = REPO / "deployment/kaggle" / HERE.name
R5A = REPO / "analysis/construction-policy-integration-20260913T221412Z-r5"
R5P = REPO / "deployment/kaggle/construction-policy-integration-20260913T221412Z-r5"
R5 = {
    "manifest": "493018986c25f14472fe6c10bd9c2c888cde4cafa8388d7425495687eb68d440",
    "notebook": "50d345ec00f2a5d089e19b4ee74412576f54a3902ba67859bd4bf9258d02dbd6",
    "build": "7135749d2a22bd89299a0306b7c6fb94a08efdba5eac1e6c5ba6e1dd66f2cf4d",
}
SOURCE = ["PROVIDER-SELECTION.json", "accelerator.py", "case.json", "constructor.py", "containment_base.py", "inspection_runner.py", "integration.py", "integration_boundary.py", "local_transport.py", "model-metadata.json", "qualified_runner.py", "real_trace.py", "request_contract.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_document(path):
    if not path.is_file():
        return None, "MISSING"
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None, "MALFORMED"
    return value, "LOADED"


def review_expected(manifest_sha, card_sha, notebook_sha):
    return {"status": "RESOLVED", "identity": HERE.name, "manifest_sha256": manifest_sha, "card_sha256": card_sha, "notebook_sha256": notebook_sha}


def review_valid(review, manifest_sha, card_sha, notebook_sha):
    return type(review) is dict and review == review_expected(manifest_sha, card_sha, notebook_sha)


def authorization_valid(auth, manifest_sha, card_sha, notebook_sha, target, consumed=False):
    if consumed or type(auth) is not dict or not auth:
        return False
    expected = {
        "schema": "one-shot-kaggle-authorization-v1",
        "identity": HERE.name,
        "manifest_sha256": manifest_sha,
        "card_sha256": card_sha,
        "notebook_sha256": notebook_sha,
        "target": target,
        "used": False,
        "maximum_pushes": 1,
        "retries": 0,
        "push_timeout_seconds": 3600,
        "displayed_gpu_hour_cap": 1.0,
    }
    if any(auth.get(key) != value for key, value in expected.items()):
        return False
    return type(auth["used"]) is bool and type(auth["maximum_pushes"]) is int and type(auth["retries"]) is int and type(auth["push_timeout_seconds"]) is int and type(auth["displayed_gpu_hour_cap"]) in (int, float)


def default_branch_valid():
    donor = json.loads((REPO / "deployment/kaggle/submission-20260912-v15/submission.ipynb").read_text())
    notebook = json.loads((PACKAGE / "construction.ipynb").read_text())
    source = notebook["cells"][1]["source"]
    tree = ast.parse(source)
    branch = next(node for node in tree.body if isinstance(node, ast.If) and node.orelse)
    embedded = [node.value.args[0].value for node in branch.orelse if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "exec" and node.value.args and isinstance(node.value.args[0], ast.Constant)]
    donor_sources = ["".join(donor["cells"][index]["source"]) for index in (1, 2)]
    gameplay = "".join(donor["cells"][3]["source"])
    command = next(node for node in branch.orelse if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "command" for target in node.targets))
    rendered = ast.unparse(command)
    return embedded == donor_sources and gameplay not in source and "remote_entry.py" in rendered and "submission_play.py" not in rendered and "/tmp/solver-venv/bin/python" in rendered


def check(write=True):
    errors = []
    if sha(R5A / "MANIFEST.json") != R5["manifest"] or sha(R5P / "BUILD.json") != R5["build"]:
        errors.append("R5_BASELINE_DRIFT")
    if sha(PACKAGE / "construction.ipynb") != sha(R5P / "construction.ipynb") or sha(PACKAGE / "construction.ipynb") != R5["notebook"]:
        errors.append("NOTEBOOK_BYTES_DRIFT")
    prerequisite, state = load_document(HERE / "R5-REVIEW-PREREQUISITE.json")
    expected_verification = {"integration": 21, "constructor": 46, "containment": 53, "default_branch_lifecycle": 3, "r4_regression_exit": 1}
    if state != "LOADED" or type(prerequisite) is not dict or prerequisite.get("status") != "RESOLVED" or prerequisite.get("manifest_sha256") != R5["manifest"] or prerequisite.get("verification") != expected_verification:
        errors.append("R5_REVIEW_PREREQUISITE")
    for name in SOURCE:
        if sha(HERE / name) != sha(R5A / name):
            errors.append("R5_SOURCE_DRIFT:" + name)
    if sha(PACKAGE / "remote_entry.py") != sha(R5P / "remote_entry.py"):
        errors.append("R5_SOURCE_DRIFT:remote_entry.py")
    if not default_branch_valid():
        errors.append("DEFAULT_DONOR_LIFECYCLE_DRIFT")
    expected = json.loads((HERE / "SERVER-REFERENCE.json").read_text())
    meta = json.loads((PACKAGE / "kernel-metadata.json").read_text())
    if meta != expected:
        errors.append("SERVER_REFERENCE_DRIFT")
    donor_meta = json.loads((REPO / "deployment/kaggle/submission-20260912-v15/kernel-metadata.json").read_text())
    for key in ("dataset_sources", "kernel_sources", "competition_sources", "model_sources"):
        if expected.get(key) != donor_meta.get(key):
            errors.append("SOURCE_CONTRACT_DRIFT:" + key)
    if expected.get("competition_sources") != ["arc-prize-2026-arc-agi-3"] or expected.get("is_private") is not True or expected.get("enable_internet") is not False or expected.get("machine_shape") != "NvidiaRtxPro6000" or "@sha256:" not in expected.get("docker_image", ""):
        errors.append("REMOTE_POLICY_DRIFT")
    manifest = json.loads((HERE / "MANIFEST.json").read_text())
    manifest_sha = sha(HERE / "MANIFEST.json")
    for relative, wanted in manifest["files"].items():
        path = HERE / relative.removeprefix("analysis/") if relative.startswith("analysis/") else PACKAGE / relative.removeprefix("package/")
        if not path.is_file() or sha(path) != wanted:
            errors.append("MANIFEST_DRIFT:" + relative)
    card_sha = sha(HERE / "EXPERIMENT-CARD.md")
    notebook_sha = sha(PACKAGE / "construction.ipynb")
    consumed = (HERE / "LAUNCH-CONSUMED.json").exists()
    auth, auth_state = load_document(HERE / "AUTHORIZATION.json")
    if auth_state != "LOADED":
        errors.append("AUTHORIZATION_" + auth_state)
    elif not authorization_valid(auth, manifest_sha, card_sha, notebook_sha, expected["id"], consumed):
        errors.append("AUTHORIZATION_INVALID_OR_CONSUMED")
    review, review_state = load_document(HERE / "REVIEW-RESOLVED.json")
    if review_state == "MISSING":
        final_review = "MISSING"
    elif review_state != "LOADED" or not review_valid(review, manifest_sha, card_sha, notebook_sha):
        final_review = "REJECTED_OR_MISMATCHED"
        errors.append("FINAL_REVIEW_INVALID")
    else:
        final_review = "RESOLVED"
    if "kaggle kernels push" in (PACKAGE / "construction.ipynb").read_text() or "submission.parquet" in ast.unparse(ast.parse(json.loads((PACKAGE / "construction.ipynb").read_text())["cells"][1]["source"])):
        errors.append("NOTEBOOK_SUBMISSION_PATH")
    immutable_errors = [error for error in errors if not error.startswith("FINAL_REVIEW_")]
    if errors:
        status = "HOLD"
    elif final_review == "RESOLVED":
        status = "APPROVED_NOT_LAUNCHED"
    else:
        status = "PREPARED_NOT_APPROVED"
    result = {
        "status": status,
        "approved": status == "APPROVED_NOT_LAUNCHED",
        "errors": errors,
        "final_review": final_review,
        "identity": HERE.name,
        "manifest_sha256": manifest_sha,
        "card_sha256": card_sha,
        "notebook_sha256": notebook_sha,
        "authorization_state": auth_state,
        "authorization_unconsumed": not consumed,
        "r5_hash_inheritance": not any(error.startswith(("R5_", "NOTEBOOK", "DEFAULT_")) for error in immutable_errors),
        "target": expected["id"],
        "pushes": 0,
        "retries": 0,
    }
    if write:
        (HERE / "evidence").mkdir(exist_ok=True)
        (HERE / "evidence/PRELAUNCH.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    result = check()
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["status"] in {"PREPARED_NOT_APPROVED", "APPROVED_NOT_LAUNCHED"} else 2)
