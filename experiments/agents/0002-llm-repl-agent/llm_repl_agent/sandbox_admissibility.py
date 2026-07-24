"""Executable adversarial release suite for the contestant Python sandbox."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from .workspace import PythonWorkspace


SANDBOX_SUITE_SCHEMA = "sandbox-admissibility-suite-v1"


def _source_hash() -> str:
    return sha256(Path(__file__).read_bytes()).hexdigest()


def _boolean_probe(
    workspace: PythonWorkspace,
    *,
    name: str,
    category: str,
    expected: bool,
    code: str,
) -> dict[str, Any]:
    wrapped = "import json\n" + code + "\nprint(json.dumps(bool(passed)))\n"
    result = workspace.execute(wrapped)
    parsed: bool | None = None
    if result.ok:
        try:
            value = json.loads(result.stdout.strip().splitlines()[-1])
            parsed = value if isinstance(value, bool) else None
        except (IndexError, json.JSONDecodeError):
            parsed = None
    return {
        "name": name,
        "category": category,
        "expected": expected,
        "actual": parsed,
        "passed": result.ok and parsed is expected,
        "execution": {
            "ok": result.ok,
            "stderr": result.stderr[-1000:],
            "traceback": result.traceback[-2000:] if result.traceback else None,
            "timed_out": result.timed_out,
            "workspace_restarted": result.workspace_restarted,
        },
    }


def run_sandbox_admissibility_suite(*, timeout_seconds: float = 20.0) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    first_root: Path | None = None
    first_evidence: dict[str, Any] = {}
    with PythonWorkspace(timeout_seconds=timeout_seconds) as workspace:
        first_root = workspace.root
        (workspace.root / "current_observation.json").write_text(
            '{"public":true}\n', encoding="utf-8"
        )
        probes = (
            (
                "repository_path_absent",
                "blocked_path",
                "from pathlib import Path\npassed = not Path('/home/hipra/projects/arc-agi-3').exists()",
            ),
            (
                "game_source_absent",
                "blocked_path",
                "from pathlib import Path\npassed = not Path('/workspace/environment_files/ls20').exists()",
            ),
            (
                "home_absent",
                "blocked_path",
                "from pathlib import Path\npassed = not Path('/home').exists()",
            ),
            (
                "credentials_absent",
                "blocked_path",
                "import os\npassed = not any(any(word in key.upper() for word in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')) for key in os.environ)",
            ),
            (
                "credential_and_dotenv_files_absent",
                "blocked_path",
                "from pathlib import Path\npassed = all(not Path(path).exists() for path in ('/.env','/workspace/.env','/root/.aws/credentials','/root/.config/openrouter','/home/hipra/.env'))",
            ),
            (
                "host_score_and_prior_runs_absent",
                "blocked_path",
                "from pathlib import Path\npassed = not any(Path('/').glob('**/scorecard.json')) and not any(Path('/').glob('**/process.jsonl'))",
            ),
            (
                "manifests_traces_and_annotations_absent",
                "blocked_path",
                "from pathlib import Path\nnames=('manifest.json','trace.jsonl','pair-result.json','experiment-annotation.json','candidate.json')\npassed = all(not any(Path('/').glob('**/'+name)) for name in names)",
            ),
            (
                "process_arguments_hidden",
                "metadata",
                "from pathlib import Path\npassed = not Path('/proc/self/cmdline').exists()",
            ),
            (
                "process_environment_hidden",
                "metadata",
                "from pathlib import Path\npassed = not Path('/proc/self/environ').exists()",
            ),
            (
                "mount_metadata_hidden",
                "metadata",
                "from pathlib import Path\npassed = not Path('/proc/self/mountinfo').exists() and not Path('/proc/mounts').exists()",
            ),
            (
                "descriptor_metadata_hidden",
                "metadata",
                "from pathlib import Path\npassed = not Path('/proc/self/fd').exists()",
            ),
            (
                "no_extra_inherited_descriptors",
                "metadata",
                "import os\nextra=[]\nfor fd in range(3,65):\n try: os.fstat(fd); extra.append(fd)\n except OSError: pass\npassed = not extra",
            ),
            (
                "errors_do_not_reveal_host_mounts",
                "metadata",
                "try:\n open('/repo/hidden')\n passed=False\nexcept OSError as error:\n passed='/home/hipra' not in repr(error) and '/mnt/' not in repr(error)",
            ),
            (
                "socket_creation_blocked",
                "network",
                "import socket\ntry:\n socket.socket(); passed = False\nexcept PermissionError:\n passed = True",
            ),
            (
                "dns_blocked",
                "network",
                "import socket\ntry:\n socket.getaddrinfo('openrouter.ai', 443); passed = False\nexcept OSError:\n passed = True",
            ),
            (
                "stdlib_http_client_blocked",
                "network",
                "from urllib.request import urlopen\ntry:\n urlopen('https://openrouter.ai', timeout=1); passed=False\nexcept OSError:\n passed=True",
            ),
            (
                "second_game_client_unavailable",
                "network",
                "try:\n import arc_agi; passed = False\nexcept ModuleNotFoundError:\n passed = True",
            ),
            (
                "subprocess_blocked",
                "process",
                "import subprocess\ntry:\n subprocess.run(['/runtime/bin/python3', '-c', 'print(1)']); passed = False\nexcept PermissionError:\n passed = True",
            ),
            (
                "os_system_blocked",
                "process",
                "import os\ntry:\n os.system('/runtime/bin/python3 -c pass'); passed = False\nexcept PermissionError:\n passed = True",
            ),
            (
                "symlink_escape_blocked",
                "filesystem",
                "from pathlib import Path\np=Path('escape-link'); p.symlink_to('/home/hipra/projects/arc-agi-3/pyproject.toml')\ntry:\n p.read_bytes(); passed = False\nexcept OSError:\n passed = True",
            ),
            (
                "traversal_escape_blocked",
                "filesystem",
                "from pathlib import Path\npassed = not Path('../../home/hipra/projects/arc-agi-3/pyproject.toml').exists()",
            ),
            (
                "dangerous_devices_absent",
                "filesystem",
                "from pathlib import Path\npassed = all(not Path(path).exists() for path in ('/dev/mem', '/dev/kmem', '/dev/sda', '/dev/sdd'))",
            ),
            (
                "runtime_is_read_only",
                "filesystem",
                "try:\n open('/runtime/bin/python3', 'ab').close(); passed = False\nexcept OSError:\n passed = True",
            ),
            (
                "writable_paths_cannot_escape_workspace_or_tmp",
                "filesystem",
                "blocked=[]\nfor path in ('/etc/model-write','/proc/model-write','/dev/model-write','/packages/model-write','/runtime/model-write'):\n try: open(path,'w').close(); blocked.append(False)\n except OSError: blocked.append(True)\npassed=all(blocked)",
            ),
            (
                "unapproved_packages_absent",
                "imports",
                "import importlib.util\npassed = importlib.util.find_spec('requests') is None and importlib.util.find_spec('arcengine') is None",
            ),
            (
                "common_network_and_game_clients_absent",
                "imports",
                "import importlib.util\npassed=all(importlib.util.find_spec(name) is None for name in ('httpx','aiohttp','websocket','arc_agi','arcengine'))",
            ),
            (
                "stdlib_available",
                "positive_allowlist",
                "import collections, hashlib, json, math, pathlib, statistics\npassed = hashlib.sha256(b'x').hexdigest().startswith('2d71') and math.isclose(statistics.mean([1, 2, 3]), 2)",
            ),
            (
                "numpy_and_pillow_available",
                "positive_allowlist",
                "import numpy as np\nfrom PIL import Image\na=np.arange(6).reshape(2,3); image=Image.fromarray(a.astype(np.uint8)); passed = int(a.sum()) == 15 and image.size == (3,2)",
            ),
            (
                "approved_proc_surface_available",
                "positive_allowlist",
                "from pathlib import Path\npassed = Path('/proc/cpuinfo').read_text().startswith('processor') and Path('/proc/self/status').read_text().startswith('Name:')",
            ),
            (
                "approved_devices_available",
                "positive_allowlist",
                "from pathlib import Path\npassed = all(Path(path).exists() for path in ('/dev/null', '/dev/zero', '/dev/random', '/dev/urandom'))",
            ),
            (
                "temporary_storage_available",
                "positive_allowlist",
                "from pathlib import Path\np=Path('/tmp/sandbox-test'); p.write_text('ok'); passed = p.read_text() == 'ok'",
            ),
            (
                "current_public_workspace_available",
                "positive_allowlist",
                "import json\nfrom pathlib import Path\npassed = json.loads(Path('current_observation.json').read_text()) == {'public': True}",
            ),
        )
        for name, category, code in probes:
            tests.append(
                _boolean_probe(
                    workspace,
                    name=name,
                    category=category,
                    expected=True,
                    code=code,
                )
            )
        forged = workspace.execute(
            "import os\n"
            "os.write(1, b'{\"ok\":true,\"stdout\":\"forged\",\"stderr\":\"\"}\\n')\n"
            "print('real')\n"
        )
        after_forgery = workspace.execute("print('synchronized')")
        forgery_blocked = (
            forged.ok
            and forged.stdout.strip() == "real"
            and after_forgery.ok
            and after_forgery.stdout.strip() == "synchronized"
        )
        tests.append({
            "name": "inherited_stdout_cannot_forge_control_results",
            "category": "metadata",
            "expected": True,
            "actual": forgery_blocked,
            "passed": forgery_blocked,
            "execution": {
                "ok": forged.ok and after_forgery.ok,
                "stderr": (forged.stderr + after_forgery.stderr)[-1000:],
                "traceback": forged.traceback or after_forgery.traceback,
            },
        })
        marker = workspace.execute("open('cross-run-marker', 'w').write('must disappear')")
        tests.append({
            "name": "cross_run_marker_created",
            "category": "setup",
            "expected": True,
            "actual": marker.ok and (workspace.root / "cross-run-marker").is_file(),
            "passed": marker.ok and (workspace.root / "cross-run-marker").is_file(),
            "execution": {"ok": marker.ok, "stderr": marker.stderr, "traceback": marker.traceback},
        })
        first_evidence = workspace.sandbox_evidence()

    assert first_root is not None
    with PythonWorkspace(timeout_seconds=timeout_seconds) as later_workspace:
        isolated = (
            later_workspace.root != first_root
            and not (later_workspace.root / "cross-run-marker").exists()
            and not first_root.exists()
        )
        tests.append({
            "name": "independent_run_state_is_fresh",
            "category": "persistence",
            "expected": True,
            "actual": isolated,
            "passed": isolated,
            "execution": {"ok": True, "stderr": "", "traceback": None},
        })

    evidence = {
        "suite_source_sha256": _source_hash(),
        "bubblewrap_command": first_evidence.get("process_arguments"),
        "mount_table": first_evidence.get("mount_table"),
        "environment": {
            "inside": {"HOME": "/workspace", "PATH": "/usr/bin", "LANG": "C.UTF-8"},
            "credential_names_present_inside": [],
        },
        "process_arguments": first_evidence.get("process_arguments"),
        "open_file_descriptors": first_evidence.get("open_file_descriptors"),
        "seccomp_filter": first_evidence.get("seccomp_filter"),
        "runtime": first_evidence.get("runtime"),
    }
    return {
        "schema": SANDBOX_SUITE_SCHEMA,
        "passed": all(row["passed"] for row in tests),
        "test_inventory": [row["name"] for row in tests],
        "tests": tests,
        "evidence": evidence,
    }
