from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

SCHEMA = "tycho-hardened-host-solver-closure-kaggle-v1"
WORKING = Path(os.environ.get("TYCHO_SMOKE_WORKING", "/kaggle/working"))
OUTPUT = WORKING / "result.json"
SOLVER_VENV = Path("/tmp/tycho-smoke-venv")
SOURCE_ROOTS = (
    Path(__file__).resolve().parent,
    Path("/kaggle/input"),
    Path("/kaggle/working"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find_one(pattern: str) -> Path:
    matches: list[Path] = []
    for root in SOURCE_ROOTS:
        if root.exists():
            matches.extend(path for path in root.rglob(pattern) if path.is_file())
    unique = sorted(set(matches))
    if not unique:
        raise RuntimeError(f"missing required asset {pattern}")
    hashes = {sha256(path) for path in unique}
    if len(hashes) != 1:
        raise RuntimeError(
            f"conflicting copies of {pattern}: {[str(path) for path in unique]}"
        )
    return unique[0]


def canonical_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def verify_wheelhouse(manifest_path: Path) -> tuple[dict[str, object], Path]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "tycho-nooa-full-wheelhouse-v1":
        raise RuntimeError(f"unexpected wheelhouse schema: {manifest.get('schema')!r}")
    wheelhouse = manifest_path.parent
    expected_wheels = {record["filename"] for record in manifest["wheels"]}
    actual_wheels = {path.name for path in wheelhouse.glob("*.whl")}
    if actual_wheels != expected_wheels:
        raise RuntimeError(
            "wheelhouse filename mismatch: "
            f"missing={sorted(expected_wheels - actual_wheels)}, "
            f"unexpected={sorted(actual_wheels - expected_wheels)}"
        )
    for record in [*manifest["wheels"], *manifest["support_files"]]:
        path = wheelhouse / record["filename"]
        if not path.is_file():
            raise RuntimeError(f"wheelhouse file missing: {path.name}")
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise RuntimeError(f"wheelhouse file differs from manifest: {path.name}")
    return manifest, wheelhouse


def install_runtime() -> dict[str, object]:
    manifest_path = find_one("wheelhouse-manifest.json")
    manifest, wheelhouse = verify_wheelhouse(manifest_path)
    uv = os.environ.get("TYCHO_SMOKE_UV") or shutil.which("uv")
    if uv is None:
        raise RuntimeError("Kaggle uv executable is unavailable")
    subprocess.run(
        [
            uv,
            "venv",
            "--python",
            sys.executable,
            str(SOLVER_VENV),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    solver_python = SOLVER_VENV / "bin/python"
    command = [
        uv,
        "pip",
        "install",
        "--python",
        str(solver_python),
        "--quiet",
        "--offline",
        "--no-index",
        "--find-links",
        str(wheelhouse),
        "arc-agi-3-tycho-nooa==0.1.0",
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, timeout=120, check=False
    )
    if completed.returncode != 0:
        raise RuntimeError(f"offline install failed: {completed.stderr[-2000:]}")
    site_packages = subprocess.check_output(
        [str(solver_python), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
        timeout=20,
    ).strip()
    installed_raw = subprocess.check_output(
        [
            str(solver_python),
            "-c",
            "import importlib.metadata as m, json; "
            "print(json.dumps(sorted((d.metadata['Name'], d.version) "
            "for d in m.distributions())))",
        ],
        text=True,
        timeout=30,
    )
    installed = {
        canonical_name(name): version for name, version in json.loads(installed_raw)
    }
    expected = {
        canonical_name(record["name"]): record["version"]
        for record in manifest["wheels"]
    }
    if installed != expected:
        raise RuntimeError(
            "installed closure differs from manifest: "
            f"missing={sorted(expected.keys() - installed.keys())}, "
            f"unexpected={sorted(installed.keys() - expected.keys())}, "
            f"version_mismatch={sorted(name for name in expected.keys() & installed.keys() if expected[name] != installed[name])}"
        )
    sys.path.insert(0, site_packages)
    local_solver = wheelhouse / "local_solver.py"
    sys.path.insert(0, str(local_solver.parent))
    return {
        "manifest": {"sha256": sha256(manifest_path), "schema": manifest["schema"]},
        "wheel_count": len(manifest["wheels"]),
        "installed_count": len(installed),
        "installed": dict(sorted(installed.items())),
        "include_system_site_packages": next(
            line.split("=", 1)[1].strip()
            for line in (SOLVER_VENV / "pyvenv.cfg").read_text().splitlines()
            if line.startswith("include-system-site-packages")
        ),
        "solver_python": str(solver_python),
        "local_solver": {"sha256": sha256(local_solver), "bytes": local_solver.stat().st_size},
    }


def worker_smoke(root: Path) -> dict[str, object]:
    from local_solver import create_solver
    from tycho_nooa.protocol import project_arc_observation

    os.environ.update(
        {
            "TYCHO_KAGGLE_HARDENED_HOST": "1",
            "TYCHO_SANDBOX_RUNTIME": "host",
            "ARC_LLM_MODEL": "tycho-no-model-smoke",
            "ARC_LLM_BASE_URL": "http://127.0.0.1:9/v1",
            "ARC_LLM_API_KEY": "synthetic-no-model-smoke",
            "ARC_LLM_USE_RESPONSES": "0",
            "TYCHO_MODE": "single",
            "TYCHO_NOOA_WORKER_PYTHON": str(SOLVER_VENV / "bin/python"),
        }
    )
    solver = create_solver(
        game_id="exact-no-model-worker-smoke",
        workspace_root=root / "worker-sessions",
        initial_observation=project_arc_observation(
            frames=[[0]],
            available_action_ids=[1],
            state="NOT_FINISHED",
            levels_completed=0,
            full_reset=False,
        ),
    )
    try:
        return {
            "started": True,
            "process_alive": solver._process.poll() is None,
            "session_gid": solver.session_root.stat().st_gid,
            "session_mode": oct(solver.session_root.stat().st_mode & 0o777),
            "workspace_gid": solver.workspace.stat().st_gid,
            "workspace_mode": oct(solver.workspace.stat().st_mode & 0o7777),
            "stdout_mode": oct(solver.stdout_log.stat().st_mode & 0o777),
            "stderr_mode": oct(solver.stderr_log.stat().st_mode & 0o777),
        }
    finally:
        solver.close()


def generated_source(sibling: Path, host: Path, *, writable: bool) -> str:
    return f'''import json, os, socket, subprocess, sys
from pathlib import Path

def attempt(operation):
    try:
        value = operation()
        return {{"allowed": True, "value": value}}
    except Exception as error:  # noqa: BLE001 - preserve the remote diagnostic
        return {{"allowed": False, "error": type(error).__name__, "errno": getattr(error, "errno", None)}}

def parent_environment():
    data = Path(f"/proc/{{os.getppid()}}/environ").read_bytes()
    return {{"bytes": len(data), "marker": b"TYCHO_SMOKE_PARENT_MARKER=" in data}}

def fork_once():
    pid = os.fork()
    if pid == 0:
        os._exit(91)
    return os.waitpid(pid, 0)[1]

def subprocess_once():
    return subprocess.run([sys.executable, "-c", "pass"], check=False).returncode

def socket_once():
    value = socket.socket()
    value.close()
    return True

def caps():
    fields = {{}}
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith(("CapEff:", "CapPrm:", "NoNewPrivs:", "Seccomp:")):
            key, value = line.split(":", 1)
            fields[key] = value.strip()
    return fields

report = {{
    "uid": os.getuid(), "gid": os.getgid(), "groups": os.getgroups(),
    "environment_keys": sorted(os.environ), "stdin": sys.stdin.read(1),
    "caps": caps(),
    "numpy": attempt(lambda: __import__("numpy").__version__),
    "own_read": attempt(lambda: len(Path("dynamics.py").read_bytes())),
    "trusted_write": attempt(lambda: Path("dynamics.py").write_text("corrupt")),
    "trusted_unlink": attempt(lambda: Path("dynamics.py").unlink()),
    "root_create": attempt(lambda: Path("unexpected.txt").write_text("bad")),
    "sibling_read": attempt(lambda: Path({str(sibling)!r}).read_text()),
    "sibling_write": attempt(lambda: Path({str(sibling)!r}).write_text("bad")),
    "host_read": attempt(lambda: Path({str(host)!r}).read_text()),
    "host_write": attempt(lambda: Path({str(host)!r}).write_text("bad")),
    "parent_environment": attempt(parent_environment),
    "socket": attempt(socket_once), "fork": attempt(fork_once),
    "subprocess": attempt(subprocess_once),
    "regain_root": attempt(lambda: os.setresuid(0, 0, 0)),
}}
if {writable!r}:
    report["notes_write"] = attempt(lambda: Path("notes/validated_plan.json").write_text("{{}}\\n"))
print(json.dumps(report, sort_keys=True))
'''


def runtime_smoke_in_process(
    root: Path, *, uid: int, writable: bool, sibling: Path, host: Path
) -> dict[str, object]:
    from tycho_nooa.trusted_runtime import current_runtime
    from tycho_nooa.wm_probe import TrustedRuntimePythonSandbox

    workspace = root / ("writable-workspace" if writable else "readonly-workspace")
    current_runtime().write_to(workspace)
    (workspace / "world_model.py").write_text("VALUE = 1\n")
    os.environ["TYCHO_KAGGLE_GAME_UID"] = str(uid)
    os.environ["TYCHO_KAGGLE_GAME_GID"] = str(uid)
    sandbox = TrustedRuntimePythonSandbox(runtime="host", workspace_read_only=not writable)
    result = sandbox.run_source(
        workspace,
        generated_source(sibling, host, writable=writable),
        timeout=20,
    )
    child = json.loads(result.stdout)
    return {
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "stderr": result.stderr,
        "child": child,
        "trusted_runtime_survived": (workspace / "dynamics.py").is_file(),
        "unexpected_root_exists": (workspace / "unexpected.txt").exists(),
        "notes_published": (workspace / "notes/validated_plan.json").read_text() == "{}\n" if writable and (workspace / "notes/validated_plan.json").is_file() else False,
    }


def runtime_smoke(
    root: Path, *, uid: int, writable: bool, sibling: Path, host: Path
) -> dict[str, object]:
    completed = subprocess.run(
        [
            str(SOLVER_VENV / "bin/python"),
            str(Path(__file__).resolve()),
            "--runtime-lane",
            str(root),
            str(uid),
            "1" if writable else "0",
            str(sibling),
            str(host),
        ],
        capture_output=True,
        text=True,
        timeout=40,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"frozen-interpreter runtime lane failed: {completed.stderr[-2000:]}"
        )
    return json.loads(completed.stdout)


def validate(report: dict[str, object]) -> list[str]:
    failures: list[str] = []
    installation = report.get("installation", {})
    if installation.get("manifest", {}).get("schema") != "tycho-nooa-full-wheelhouse-v1":
        failures.append("installation: wrong or missing closure manifest")
    if installation.get("wheel_count") != 77 or installation.get("installed_count") != 77:
        failures.append("installation: closure package count mismatch")
    if installation.get("include_system_site_packages") != "false":
        failures.append("installation: ambient system packages are visible")
    expected_versions = {
        "arc-agi": "0.9.9", "arcengine": "0.9.3", "numpy": "2.4.6",
        "litellm": "1.97.0", "nooa": "0.0.1.dev95",
        "arc-agi-3-tycho": "0.1.0", "arc-agi-3-tycho-nooa": "0.1.0",
    }
    for name, version in expected_versions.items():
        if installation.get("installed", {}).get(name) != version:
            failures.append(f"installation: {name} is not pinned to {version}")
    worker = report.get("worker", {})
    expected_worker = {
        "started": True, "process_alive": True, "session_gid": 60001,
        "session_mode": "0o710", "workspace_gid": 60001,
        "workspace_mode": "0o2710", "stdout_mode": "0o600", "stderr_mode": "0o600",
    }
    for key, expected in expected_worker.items():
        if worker.get(key) != expected:
            failures.append(f"worker.{key}={worker.get(key)!r}, expected {expected!r}")
    for lane in ("read_only", "writable"):
        item = report.get(lane, {})
        child = item.get("child", {})
        if item.get("returncode") != 0 or item.get("timed_out"):
            failures.append(f"{lane}: execution failed")
            continue
        for key in ("trusted_write", "trusted_unlink", "root_create", "sibling_read", "sibling_write", "host_read", "host_write", "parent_environment", "socket", "fork", "subprocess", "regain_root"):
            if child.get(key, {}).get("allowed") is not False:
                failures.append(f"{lane}.{key} unexpectedly allowed")
        if child.get("uid") != (60011 if lane == "read_only" else 60012):
            failures.append(f"{lane}: wrong uid")
        if child.get("gid") != child.get("uid") or child.get("groups") != []:
            failures.append(f"{lane}: identity not fully dropped")
        if child.get("environment_keys") != ["HOME", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "PATH", "PYTHONDONTWRITEBYTECODE", "PYTHONNOUSERSITE"]:
            failures.append(f"{lane}: environment allowlist mismatch")
        if child.get("stdin") != "":
            failures.append(f"{lane}: stdin not sealed")
        if child.get("caps", {}).get("CapEff") != "0000000000000000" or child.get("caps", {}).get("CapPrm") != "0000000000000000":
            failures.append(f"{lane}: capabilities remain")
        if child.get("caps", {}).get("NoNewPrivs") != "1" or child.get("caps", {}).get("Seccomp") != "2":
            failures.append(f"{lane}: no_new_privs/seccomp missing")
        if child.get("numpy", {}).get("value") != "2.4.6" or child.get("own_read", {}).get("allowed") is not True:
            failures.append(f"{lane}: intended reads/import failed")
        if item.get("trusted_runtime_survived") is not True or item.get("unexpected_root_exists") is not False:
            failures.append(f"{lane}: workspace boundary changed")
    writable = report.get("writable", {})
    if writable.get("child", {}).get("notes_write", {}).get("allowed") is not True or writable.get("notes_published") is not True:
        failures.append("writable: notes publication failed")
    read_only = report.get("read_only", {})
    if "notes_write" in read_only.get("child", {}):
        failures.append("read_only: unexpected notes probe")
    if report.get("canaries_after") != {"sibling": "sibling-canary\n", "host": "host-canary\n"}:
        failures.append("external canary changed")
    return failures


def main() -> None:
    started = time.monotonic()
    WORKING.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"schema": SCHEMA}
    try:
        report["installation"] = install_runtime()
        root = WORKING / "tycho-hardened-host-smoke"
        root.mkdir(exist_ok=True)
        sibling_root = root / "sibling-control"
        host_root = root / "host-control"
        sibling_root.mkdir(exist_ok=True)
        host_root.mkdir(exist_ok=True)
        sibling = sibling_root / "canary.txt"
        host = host_root / "canary.txt"
        sibling.write_text("sibling-canary\n")
        host.write_text("host-canary\n")
        os.chown(sibling_root, 60002, 60002)
        os.chown(sibling, 60002, 60002)
        os.chmod(sibling_root, 0o700)
        os.chmod(sibling, 0o600)
        os.chown(host_root, 0, 0)
        os.chown(host, 0, 0)
        os.chmod(host_root, 0o700)
        os.chmod(host, 0o600)
        os.environ["TYCHO_SMOKE_PARENT_MARKER"] = "present-not-secret"
        report["worker"] = worker_smoke(root)
        report["read_only"] = runtime_smoke(root, uid=60011, writable=False, sibling=sibling, host=host)
        report["writable"] = runtime_smoke(root, uid=60012, writable=True, sibling=sibling, host=host)
        report["canaries_after"] = {"sibling": sibling.read_text(), "host": host.read_text()}
    except Exception as error:  # noqa: BLE001 - preserve the remote diagnostic
        report["supervisor_error"] = {"type": type(error).__name__, "message": str(error)}
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["failures"] = validate(report)
    report["verdict"] = "exact_hardened_host_solver_closure_qualified" if not report["failures"] and "supervisor_error" not in report else "not_qualified"
    OUTPUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"schema": SCHEMA, "verdict": report["verdict"], "failures": report["failures"], "supervisor_error": report.get("supervisor_error")}, sort_keys=True))


if __name__ == "__main__":
    if len(sys.argv) == 7 and sys.argv[1] == "--runtime-lane":
        print(
            json.dumps(
                runtime_smoke_in_process(
                    Path(sys.argv[2]),
                    uid=int(sys.argv[3]),
                    writable=sys.argv[4] == "1",
                    sibling=Path(sys.argv[5]),
                    host=Path(sys.argv[6]),
                ),
                sort_keys=True,
            )
        )
    else:
        main()
