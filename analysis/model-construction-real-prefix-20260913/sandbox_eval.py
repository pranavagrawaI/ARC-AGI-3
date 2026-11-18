from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import selectors
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from real_trace import RealTraceCase

# This runner has neither evidence nor assessment code. Each invocation is fresh.
RUNNER = r'''import importlib.util, json, resource
resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
from pathlib import Path
request = json.loads(Path('/work/args.json').read_text())
spec = importlib.util.spec_from_file_location('candidate', '/work/candidate.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
value = getattr(module, request['method'])(*request['args'])
print(json.dumps(value, allow_nan=False))
'''


def _resource_limits():
    resource.setrlimit(resource.RLIMIT_AS, (268435456, 268435456))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))


def bounded_run(command, *, timeout, output_cap):
    """Drain both pipes incrementally; kill the namespace at a hard byte cap."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True, preexec_fn=_resource_limits)
    assert process.stdout is not None and process.stderr is not None
    buffers = {process.stdout.fileno(): bytearray(), process.stderr.fileno(): bytearray()}
    deadline = time.monotonic() + timeout
    try:
        with selectors.DefaultSelector() as selector:
            for stream in buffers:
                selector.register(stream, selectors.EVENT_READ)
            total = 0
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('candidate execution time limit')
                for key, _ in selector.select(min(remaining, .1)):
                    chunk = os.read(key.fd, min(8192, output_cap-total+1))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    total += len(chunk)
                    if total > output_cap:
                        raise RuntimeError('candidate output limit exceeded')
                    buffers[key.fd].extend(chunk)
            process.wait(timeout=max(.001, deadline-time.monotonic()))
        return subprocess.CompletedProcess(command, process.returncode,
            buffers[process.stdout.fileno()].decode('utf-8', errors='replace'),
            buffers[process.stderr.fileno()].decode('utf-8', errors='replace'))
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        process.stdout.close()
        process.stderr.close()


def observed(state):
    grid = state["grid"]
    cells = [(r, c) for r, row in enumerate(grid) for c, value in enumerate(row) if value == 12]
    if not cells:
        raise ValueError("observer found no block")
    bar_left = next((c for c, value in enumerate(grid[61]) if value == 11), None)
    if bar_left is None:
        raise ValueError("observer found no meter")
    return {"block": [min(r for r, _ in cells), min(c for _, c in cells)], "bar_left": bar_left}


def json_value(value, label):
    encoded = json.dumps(value, allow_nan=False, sort_keys=True)
    if len(encoded) > 200000:
        raise ValueError(f"{label} exceeds JSON size limit")
    return json.loads(encoded)


def branches_for(module, branches, action, cap):
    expanded = []
    for branch in branches:
        produced = module.predict(copy.deepcopy(branch["state"]), action)
        if not isinstance(produced, list) or not produced:
            raise ValueError("predict must return a nonempty list")
        for index, item in enumerate(produced):
            if not isinstance(item, dict) or set(("state", "commitments")) - set(item):
                raise ValueError("each prediction branch needs state and commitments")
            commitments = json_value(item["commitments"], "commitments")
            if not isinstance(commitments, dict):
                raise ValueError("commitments must be a JSON object")
            dependencies = item.get("declared_dependencies", [])
            if not isinstance(dependencies, list) or not all(isinstance(x, str) for x in dependencies):
                raise ValueError("declared_dependencies must be a list of strings")
            expanded.append({
                "ancestry": branch["ancestry"] + [index],
                "state": json_value(item["state"], "predicted state"),
                "commitments": commitments,
                "dependencies": sorted(set(dependencies)),
                "history": branch["history"] + [commitments],
            })
            if len(expanded) > cap:
                raise RuntimeError("branch_cap_exceeded")
    return expanded


def compatible(branch, measurement, required):
    return all(field in branch["commitments"] and branch["commitments"][field] == measurement[field] for field in required)


def replay_visible(module, fixture):
    visible = fixture["visible"]
    initial = {"grid": visible[0]["before"]["grid"]}
    state = json_value(module.initialize(copy.deepcopy(initial)), "initial belief")
    branches = [{"ancestry": [0], "state": state, "commitments": {}, "dependencies": [], "history": []}]
    missing = set()
    dependency_sets = []
    checks = []
    for row in visible:
        forecasts = branches_for(module, branches, row["action"], fixture["branch_cap"])
        measurement = observed(row["after"])
        for forecast in forecasts:
            missing.update(set(fixture["required_targets"]) - set(forecast["commitments"]))
            dependency_sets.append(forecast["dependencies"])
        survivors = [b for b in forecasts if compatible(b, measurement, fixture["required_targets"])]
        checks.append({
            "step": row["step"],
            "action": row["action"],
            "measurement": measurement,
            "forecast_branches": len(forecasts),
            "compatible_branches": len(survivors),
            "status": "passed" if survivors else "contradicted",
        })
        if not survivors:
            return [], {"status": "contradicted", "checks": checks}, missing, dependency_sets
        assimilated = []
        observation = {"grid": row["after"]["grid"]}
        for survivor in survivors:
            verdict = {"status": "compatible", "measurement": measurement}
            next_state = module.assimilate(copy.deepcopy(survivor["state"]), copy.deepcopy(observation), verdict)
            assimilated.append({**survivor, "state": json_value(next_state, "assimilated state")})
        branches = assimilated
    return branches, {"status": "passed", "checks": checks}, missing, dependency_sets


def sealed_rollout(module, start, fixture):
    suffix = fixture["suffix"]
    if not suffix:
        return {"status": "not_run", "compatible_depth": 0}, []
    branches = copy.deepcopy(start)
    dependency_sets = []
    layers = []
    for row in suffix:
        branches = branches_for(module, branches, row["action"], fixture["branch_cap"])
        dependency_sets.extend(branch["dependencies"] for branch in branches)
        layers.append(copy.deepcopy(branches))
    survivors = None
    audit = []
    compatible_depth = 0
    for depth, row in enumerate(suffix):
        measurement = observed(row["after"])
        candidates = layers[depth]
        if survivors is not None:
            live_ancestry = {tuple(branch["ancestry"]) for branch in survivors}
            candidates = [
                branch for branch in candidates
                if tuple(branch["ancestry"][:-1]) in live_ancestry
            ]
        survivors = [
            branch for branch in candidates
            if compatible(branch, measurement, fixture["required_targets"])
        ]
        audit.append({
            "step": row["step"],
            "action": row["action"],
            "measurement": measurement,
            "compatible_paths": len(survivors),
        })
        if not survivors:
            return {
                "status": "contradicted",
                "compatible_depth": compatible_depth,
                "first_contradiction_step": row["step"],
                "audit": audit,
            }, dependency_sets
        compatible_depth += 1
    return {"status": "passed", "compatible_depth": compatible_depth, "audit": audit}, dependency_sets


def _bubblewrap_base() -> list[str]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("bubblewrap is required for generated candidate execution")
    command = [
        bwrap,
        "--die-with-parent",
        "--new-session",
        "--unshare-all",
        "--clearenv",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for path in ("/lib", "/lib64"):
        if Path(path).exists():
            command.extend(("--ro-bind", path, path))
    command.extend(
        (
            "--proc",
            "/proc",
            "--dev",
            "/dev",
            "--tmpfs",
            "/tmp",
            "--dir",
            "/work",
        )
    )
    return command


def _bubblewrap_command(root: Path) -> list[str]:
    command = _bubblewrap_base()
    for name in ('candidate.py', 'args.json', 'runner.py'):
        command.extend(('--ro-bind', str(root / name), '/work/' + name))
    command.extend(('--chdir', '/work', '/usr/bin/python3', '-B', '/work/runner.py'))
    return command


class CandidateRPC:
    def __init__(self, root):
        self.root = root
        self.deadline = time.monotonic() + 15

    def __getattr__(self, method):
        if method not in {'initialize', 'predict', 'assimilate'}:
            raise AttributeError(method)
        def invoke(*args):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError('candidate evaluation time limit')
            (self.root / 'args.json').write_text(json.dumps({'method': method, 'args': args}))
            completed = bounded_run(_bubblewrap_command(self.root),
                                    timeout=min(3, remaining), output_cap=1_000_000)
            if completed.returncode:
                raise RuntimeError(f'candidate sandbox exited {completed.returncode}: {completed.stderr[:2000]}')
            return json_value(json.loads(completed.stdout), 'call result')
        return invoke


def evaluate_candidate(source: str, case: RealTraceCase, *, include_suffix: bool) -> dict[str, Any]:
    fixture = case.evaluator_payload(include_suffix=include_suffix)
    with tempfile.TemporaryDirectory(prefix='model-construction-eval-') as directory:
        root = Path(directory)
        (root / 'candidate.py').write_text(source)
        (root / 'runner.py').write_text(RUNNER)
        module = CandidateRPC(root)
        start, visible, missing, visible_dependencies = replay_visible(module, fixture)
        if visible['status'] == 'passed':
            suffix, rollout_dependencies = sealed_rollout(module, start, fixture)
        else:
            suffix, rollout_dependencies = {'status': 'not_run', 'compatible_depth': 0}, []
    digest = hashlib.sha256(source.encode()).hexdigest()
    return {
        'model_sha256': digest, 'belief_model_sha256': digest,
        'visible': visible, 'suffix': suffix,
        'coverage': {'required': fixture['required_targets'], 'missing_required': sorted(missing)},
        'dependencies': {
            'declared_on_visible_cases': sorted({x for g in visible_dependencies for x in g}),
            'encountered_on_proposed_rollout': sorted({x for g in rollout_dependencies for x in g}),
            'static_completeness_established': False,
        },
    }


def execute_inspection(code: str, case: RealTraceCase) -> dict[str, Any]:
    if not isinstance(code, str) or not code.strip():
        raise ValueError("inspection code must be nonempty text")
    if len(code.encode()) > 32_000:
        raise ValueError("inspection code exceeds 32,000 bytes")
    with tempfile.TemporaryDirectory(prefix="model-construction-inspect-") as directory:
        root = Path(directory)
        (root / "inspection.py").write_text(code)
        (root / "case.json").write_text(json.dumps(case.constructor_view()))
        command = _bubblewrap_base()
        command.extend(
            (
                "--dir",
                "/evidence",
                "--ro-bind",
                str(root / "case.json"),
                "/evidence/case.json",
                "--ro-bind",
                str(root / "inspection.py"),
                "/work/inspection.py",
                "--chdir",
                "/work",
                "/usr/bin/python3",
                "/work/inspection.py",
            )
        )
        completed = bounded_run(command, timeout=8, output_cap=20_000)
    stdout = completed.stdout[:16_000]
    stderr = completed.stderr[:4_000]
    return {
        "status": "completed" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "output_truncated": len(completed.stdout) > len(stdout)
        or len(completed.stderr) > len(stderr),
    }
