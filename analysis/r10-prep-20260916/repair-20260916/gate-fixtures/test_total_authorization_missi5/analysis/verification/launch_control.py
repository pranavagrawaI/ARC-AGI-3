"""One-shot, coordinator-owned Kaggle gate. No provider call on import/check.

CLI: python launch_control.py {check,push,monitor} --run-dir EXACT_ABSOLUTE
     monitor additionally accepts --deadline-seconds (default 3600), --interval (30).
Exit: 0 checked/pushed/terminal, 2 HOLD, 3 monitor deadline (NOT remote stop).
API: main(argv)->int; validate(run_dir)->manifest (read-only, raises Hold).

Files in the fixed run directory (coordinator creates these; this module never does):
 manifest.json: {schema:1, identity:IDENTITY, run_dir:absolute, package_dir:absolute,
   target:TARGET, card_sha256:hex, notebook_sha256:hex,
   sources:{repo-relative-file:sha256,...}}. Sources MUST cover this script,
   test_launch_control.py, EXPERIMENT-CARD.md, local DECISION.json, every package file.
 manifest.policy: {schema:1, identity:IDENTITY, target:TARGET, limits:LIMITS,
   private:true, internet:false, accelerator:"NvidiaRtxPro6000",
   competition_submission:false, dataset_push:false, local_monitor_cancels_remote:false}
 EXPERIMENT-CARD.md is the hash-bound prose card; independent review checks it.
 authorization.json: {schema:1, identity:IDENTITY, target:TARGET,
   manifest_sha256:hex, card_sha256:hex, notebook_sha256:hex,
   max_pushes:1, retries:0, timeout_seconds:3600, used:false}
 launch-review.json: {status:"APPROVED", manifest_sha256:hex}.
 All required values are type-strict; extra fields do not override required ones.
 Authorization and launch review are excluded from sources (no hash cycles).
 Independent local package review is necessary but NOT launch approval.

Push revalidates before and after read-only quota/list, then exclusively creates
 CONSUMED.json and fsyncs it before the sole subprocess push. Never remove it,
 including on timeout, interruption, malformed output or uncertain admission.
 Receipts are exclusive UUID-named JSON with argv/stdout/stderr/returncode.
 launch.json binds the exact parsed push receipt, URL and positive version.
 Monitor only reads that receipt and sends SDK status requests with version_label
 set explicitly: installed `kaggle kernels status owner/slug/version` silently
 discards version! SDK runs in Kaggle CLI's own interpreter, no credentials read
 by this gate. No cancel/submission/dataset write/output download exists here.
 Monitor deadline only stops local polling; provider --timeout 3600 is the remote
 maximum runtime, not a guarantee about billing or queue time.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
IDENTITY = "r10a-cd82-20260916"
RUN_DIR = ROOT / "analysis/r10-prep-20260916" / IDENTITY
PACKAGE_DIR = ROOT / "analysis/r10-prep-20260916/r10a-package"
SCRIPT = Path(__file__).resolve()
INVENTORY = SCRIPT.parent.parent / "R10A-PACKAGE-INVENTORY.json"
LOCAL_REVIEW = ROOT / "analysis/r10-prep-20260916/r10a-verification/package-review/DECISION.json"
CLI = "/home/pranav/.local/bin/kaggle"
SDK_PYTHON = "/home/pranav/.local/share/uv/tools/kaggle/bin/python"
TARGET = "pranavagrawai/arc3-r10a-cd82-20260916"
NOTEBOOK_HASH = '87afeeb324d33bafe4a7992abf1f5f17ba9ef83631d716b3c5287a061fa770a3'
LOCAL_MANIFEST_HASH = 'b2325fb93db5dfa16254a7d3d6c06b82a8b628bb61f4374276be6fe3e76e8aef'
DATASETS = ["jakobbrggen/qwen3-8-27b-fp8-hf-snapshot", "mahdiseddigh/kaggle-t4-stack-2026-08-28", "pranavagrawai/sglang-r8l3b-orderly-shutdown-runtime", "pranavagrawai/tycho-nooa-solver-wheelhouse-cp312-v11"]
LIMITS = dict(gpu_hours_reserved=1, min_gpu_hours_remaining=1, timeout_seconds=3600,
              local_deadline_seconds=3500, cleanup_seconds=8, reap_seconds=30, work_seconds=3100,
              boot_seconds=1200, launcher_seconds=1800, arms=2, games=["ls20", "s5i5", "cd82"], play_game="cd82",
              admission_seconds_per_arm=600, actions_per_arm=200, server_chat_calls=259,
              warmup_calls=1, application_calls=258, arm_call_cap_a=129, arm_call_cap_b=129,
              llm_timeout_seconds=90, actor_max_tokens=16384, critic_max_tokens=192,
              constructor_requests_per_arm=6, constructor_wall_seconds=240, constructor_hard_seconds=245,
              constructor_http_seconds=30, supervisor_request_timeout_seconds=180,
              max_pushes=1, retries=0)
STATUS_PROGRAM = '''import json,sys
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
api=KaggleApi(); api.authenticate()
target,version=sys.argv[1:]
r=ApiGetKernelSessionStatusRequest()
r.user_name,r.kernel_slug=target.split("/")
r.version_label=version
with api.build_kaggle_client() as client:
    response=client.kernels.kernels_api_client.get_kernel_session_status(r)
provider_status=response.status.name
status="CANCELLED" if provider_status == "CANCEL_ACKNOWLEDGED" else provider_status
print(json.dumps({"target":target,"version":int(version),"status":status,"provider_status":provider_status,"failure_message":response.failure_message}))
'''


class Hold(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Hold(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pairs(items):
    result = {}
    for k, v in items:
        require(k not in result, "duplicate JSON key")
        result[k] = v
    return result


def parse(text):
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(Hold("nonfinite JSON")))


def load(path):
    result = parse(Path(path).read_text())
    require(type(result) is dict and bool(result), f"nonempty object required: {path}")
    return result


def matches(obj, expected):
    require(type(obj) is dict, "object required")
    for key, value in expected.items():
        require(key in obj and type(obj[key]) is type(value) and obj[key] == value,
                f"wrong/missing binding: {key}")


def exact_run(run_dir):
    require(str(run_dir) == str(RUN_DIR), "wrong run_dir (exact absolute required)")
    require(RUN_DIR.resolve() == RUN_DIR and RUN_DIR.is_dir(), "missing/aliased run_dir")
    return RUN_DIR


def validate_package():
    require(sha(INVENTORY) == LOCAL_MANIFEST_HASH, "candidate inventory drift")
    inventory = load(INVENTORY)
    matches(inventory, {"status": "UNFROZEN_CANDIDATE"})
    require(PACKAGE_DIR.resolve() == PACKAGE_DIR and PACKAGE_DIR.is_dir(), "invalid package")
    files = {}
    for path in PACKAGE_DIR.rglob("*"):
        require(not path.is_symlink(), "package symlink forbidden")
        require("__pycache__" not in path.parts and path.suffix != ".pyc", "bytecode forbidden")
        if path.is_file():
            files[str(path.relative_to(PACKAGE_DIR))] = sha(path)
    require(files == inventory.get("files"), "candidate inventory file mismatch")
    require(sha(PACKAGE_DIR / "construction.ipynb") == NOTEBOOK_HASH, "notebook drift")
    matches(load(LOCAL_REVIEW), dict(status="APPROVED_LOCAL_PACKAGE", manifest_sha256=LOCAL_MANIFEST_HASH,
                                    notebook_sha256=NOTEBOOK_HASH, launch_authorized=False))
    return inventory


def validate(run_dir):
    run = exact_run(run_dir)
    require(not os.path.lexists(run / "CONSUMED.json"), "authorization consumed")
    validate_package()
    m = load(run / "manifest.json")
    matches(m, dict(schema=1, identity=IDENTITY, run_dir=str(RUN_DIR),
                    package_dir=str(PACKAGE_DIR), target=TARGET, notebook_sha256=NOTEBOOK_HASH))
    sources = m.get("sources")
    require(type(sources) is dict and bool(sources), "sources mapping required")
    required = {SCRIPT, SCRIPT.with_name("test_launch_control.py"), SCRIPT.with_name("make_manifest.py"), SCRIPT.with_name("test_make_manifest.py"), INVENTORY, run / "EXPERIMENT-CARD.md", LOCAL_REVIEW}
    require(PACKAGE_DIR.resolve() == PACKAGE_DIR and PACKAGE_DIR.is_dir(), "invalid package")
    for p in PACKAGE_DIR.rglob("*"):
        require(not p.is_symlink(), "package symlink forbidden")
        if p.is_file():
            required.add(p)
    require({str(p.relative_to(ROOT)) for p in required} <= sources.keys(), "incomplete source coverage")
    excluded = {run / name for name in ("manifest.json", "authorization.json", "launch-review.json", "CONSUMED.json", "launch.json", "hermes-release.json")}
    for name, digest in sources.items():
        require(type(name) is str and type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest), "bad source/hash")
        p = ROOT / name
        require(not Path(name).is_absolute() and p.resolve().is_relative_to(ROOT)
                and str(p.relative_to(ROOT)) == name and ".." not in Path(name).parts
                and p.resolve() == p and p not in excluded, "unsafe/excluded source path")
        require(p.is_file() and sha(p) == digest, f"source drift: {name}")
    require(sha(PACKAGE_DIR / "construction.ipynb") == NOTEBOOK_HASH, "notebook drift")
    require(type(m.get("card_sha256")) is str and m["card_sha256"] == sha(run / "EXPERIMENT-CARD.md"), "card hash mismatch")
    card = m.get("policy")
    matches(card, dict(schema=1, identity=IDENTITY, target=TARGET, private=True, internet=False,
                       accelerator="NvidiaRtxPro6000", competition_submission=False,
                       dataset_push=False, local_monitor_cancels_remote=False))
    matches(card.get("limits"), LIMITS)
    metadata = load(PACKAGE_DIR / "kernel-metadata.json")
    matches(metadata, dict(id=TARGET, is_private=True, enable_gpu=True, enable_tpu=False,
                           enable_internet=False, machine_shape="NvidiaRtxPro6000",
                           dataset_sources=DATASETS, competition_sources=["arc-prize-2026-arc-agi-3"],
                           kernel_sources=[], model_sources=[], language="python", kernel_type="notebook",
                           code_file="construction.ipynb"))
    matches(load(LOCAL_REVIEW), dict(status="APPROVED_LOCAL_PACKAGE", manifest_sha256=LOCAL_MANIFEST_HASH,
                                    notebook_sha256=NOTEBOOK_HASH, launch_authorized=False))
    digest = sha(run / "manifest.json")
    matches(load(run / "launch-review.json"), dict(status="APPROVED", manifest_sha256=digest))
    matches(load(run / "authorization.json"), dict(schema=1, identity=IDENTITY, target=TARGET,
        manifest_sha256=digest, card_sha256=m["card_sha256"], notebook_sha256=NOTEBOOK_HASH,
        max_pushes=1, retries=0, timeout_seconds=3600, used=False))
    matches(load(run / "hermes-release.json"), dict(status="RELEASED", authority="Hermes",
        identity=IDENTITY, target=TARGET, manifest_sha256=digest, notebook_sha256=NOTEBOOK_HASH,
        max_pushes=1, retries=0))
    return m


def exclusive(path, value):
    # Crash during serialization/fsync still leaves an exclusive marker; fail closed.
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def command(run, label, argv, timeout=120):
    receipt = dict(argv=argv, started_unix=time.time(), returncode=None, stdout="", stderr="")
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                stdin=subprocess.DEVNULL, close_fds=True, check=False)
        receipt.update(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except (subprocess.TimeoutExpired, OSError) as e:
        receipt["error"] = type(e).__name__
        for field in ("stdout", "stderr"):
            value = getattr(e, field, None) or ""
            receipt[field] = value.decode(errors="replace") if isinstance(value, bytes) else value
    finally:
        receipt["finished_unix"] = time.time()
        path = run / f"{label}-{uuid.uuid4().hex}.json"
        exclusive(path, receipt)
    return path, receipt


def output(receipt):
    require(receipt["returncode"] == 0 and "error" not in receipt, "provider command failed; receipt retained")
    return receipt["stdout"].strip()


def parse_push(text):
    matches_found = list(re.finditer(r"^Kernel version ([1-9][0-9]*) successfully pushed\.  Please check progress at (\S+)$", text, re.MULTILINE))
    require(len(matches_found) == 1, "uncertain push: invalid success/version response")
    match = matches_found[0]
    version = int(match[1])
    url = match[2]
    # Both documented base URLs and explicit version URLs bind the same exact target.
    require(url in (f"https://www.kaggle.com/code/{TARGET}",
                    f"https://www.kaggle.com/code/{TARGET}?scriptVersionId={version}",
                    f"https://www.kaggle.com/code/{TARGET}/version/{version}"), "push URL target/version mismatch")
    return version, url


def push(run_dir):
    run = exact_run(run_dir)
    validate(run_dir)
    digest = sha(run / "manifest.json")
    _, quota = command(run, "quota", [CLI, "quota", "--format", "json"])
    rows = parse(output(quota))
    require(type(rows) is list, "malformed quota")
    gpu = [r for r in rows if type(r) is dict and r.get("resource") == "GPU"]
    require(len(gpu) == 1, "GPU quota unavailable")
    remaining = gpu[0].get("remaining")
    require(type(remaining) is str and re.fullmatch(r"[0-9]+\.[0-9]{2}h", remaining), "malformed GPU hours")
    require(Decimal(remaining[:-1]) >= 1, "GPU quota below one hour")
    _, listed = command(run, "target-list", [CLI, "kernels", "list", "--mine", "--search", TARGET.split("/")[1], "--format", "json"])
    text = output(listed)
    require(text == "Not found" or parse(text) == [], "target not provably absent")
    validate(run_dir)
    require(sha(run / "manifest.json") == digest, "manifest changed during preflight")
    exclusive(run / "CONSUMED.json", dict(identity=IDENTITY, target=TARGET,
        manifest_sha256=digest, used=True, consumed_unix=time.time(), max_pushes=1))
    receipt_path, receipt = command(run, "push", [CLI, "kernels", "push", "-p", str(PACKAGE_DIR),
        "--timeout", "3600", "--accelerator", "NvidiaRtxPro6000"], timeout=180)
    version, url = parse_push(output(receipt))
    result = dict(identity=IDENTITY, target=TARGET, version=version, url=url,
                  manifest_sha256=digest, push_receipt=receipt_path.name, push_receipt_sha256=sha(receipt_path))
    exclusive(run / "launch.json", result)
    require(receipt["stdout"].strip() == f"Kernel version {version} successfully pushed.  Please check progress at {url}"
            and not receipt["stderr"].strip(), "push warnings: identity retained for read-only recovery; never retry")
    return result


def launch_identity(run):
    launch = load(run / "launch.json")
    matches(launch, dict(identity=IDENTITY, target=TARGET))
    matches(load(run / "CONSUMED.json"), dict(identity=IDENTITY, target=TARGET,
            manifest_sha256=launch.get("manifest_sha256"), used=True, max_pushes=1))
    name = launch.get("push_receipt")
    require(type(name) is str and re.fullmatch(r"push-[0-9a-f]{32}\.json", name), "invalid receipt path")
    receipt = run / name
    require(sha(receipt) == launch.get("push_receipt_sha256"), "push receipt drift")
    version, url = parse_push(output(load(receipt)))
    matches(launch, dict(version=version, url=url))
    return launch


def monitor(run_dir, deadline_seconds=3600, interval=30):
    run = exact_run(run_dir)
    require(type(deadline_seconds) in (int, float) and math.isfinite(deadline_seconds)
            and 0 < deadline_seconds <= 86400, "bad monitor deadline")
    require(type(interval) in (int, float) and math.isfinite(interval) and 0 < interval <= 300, "bad poll interval")
    launch = launch_identity(run)
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        path, receipt = command(run, "status", [SDK_PYTHON, "-c", STATUS_PROGRAM, TARGET,
                               str(launch["version"])], timeout=min(120, max(.001, deadline-time.monotonic())))
        try:
            status = parse(output(receipt))
            matches(status, dict(target=TARGET, version=launch["version"]))
            require(status.get("status") in ("QUEUED", "RUNNING", "COMPLETE", "ERROR", "CANCELLED"), "unknown status")
            if status["status"] in ("COMPLETE", "ERROR", "CANCELLED"):
                result = dict(status=status["status"], target=TARGET, version=launch["version"], receipt=path.name)
                exclusive(run / f"terminal-{uuid.uuid4().hex}.json", result)
                return result
        except (Hold, ValueError, TypeError):
            pass  # Retain raw attempt; retry read-only, never re-push.
        time.sleep(min(interval, max(0, deadline-time.monotonic())))
    return dict(status="LOCAL_MONITOR_DEADLINE", remote_stop_established=False,
                target=TARGET, version=launch["version"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=("check", "push", "monitor"))
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--deadline-seconds", type=float, default=3600)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args(argv)
    try:
        if args.action == "check":
            validate(args.run_dir)
            result = dict(status="GREEN_LOCAL_GATE", provider_checks="NOT_RUN")
        elif args.action == "push":
            result = push(args.run_dir)
        else:
            result = monitor(args.run_dir, args.deadline_seconds, args.interval)
        print(json.dumps(result, sort_keys=True))
        return 3 if result.get("status") == "LOCAL_MONITOR_DEADLINE" else 0
    except (Hold, OSError, ValueError, TypeError, KeyError) as e:
        print(json.dumps(dict(status="HOLD", reason=str(e))), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
