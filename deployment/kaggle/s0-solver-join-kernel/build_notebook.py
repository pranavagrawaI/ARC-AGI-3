"""Build the one-turn Solver join from the accepted R8-L3B v14 notebook."""

from __future__ import annotations

import base64
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
V14 = (
    ROOT
    / "analysis/deployment-probes/20260831T160520Z-qwen38-r8l3b-kaggle-v14"
    / "remote-source-v14/prototype-qwen3-8-fp8-r8-b-request-path.ipynb"
)
HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "s0_solver_join.ipynb"
RUNTIME = HERE / "s0_solver_join_runtime.py"
JOIN = HERE.parent / "s0-solver-join/solver_join_probe.py"
LOCAL_SOLVER = HERE.parent / "runtime/local_solver.py"
WHEELHOUSE_MANIFEST_SHA256 = (
    "b9ae213b070ee20223c46fbba06647838bc452b517128693f97c66205e3da081"
)


def encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def build() -> dict[str, object]:
    notebook = json.loads(V14.read_text())
    notebook["cells"][0]["source"] = (
        "# PROTOTYPE — one real Tycho-NOOA Solver turn\n\n"
        "This preserves the accepted R8-L3B v14 model/runtime/bootstrap/lifecycle "
        "route, installs the frozen 77-package Solver closure into a separate venv, "
        "and asks the actual persistent worker for one action on a synthetic public "
        "grid. It performs no ARC environment action and makes no gameplay claim."
    )
    assets = "".join(notebook["cells"][1]["source"])
    assets += f"""
Path('/tmp/s0_solver_join_runtime.py').write_bytes(base64.b64decode({encoded(RUNTIME)!r}))
Path('/tmp/s0_solver_join_probe.py').write_bytes(base64.b64decode({encoded(JOIN)!r}))
Path('/tmp/runtime').mkdir(exist_ok=True)
Path('/tmp/runtime/local_solver.py').write_bytes(base64.b64decode({encoded(LOCAL_SOLVER)!r}))
"""
    notebook["cells"][1]["source"] = assets

    install = "".join(notebook["cells"][2]["source"])
    install += f"""

# Install the independently frozen Solver closure without exposing the SGLang
# environment's packages to the worker.
solver_manifest_path = unique_exact_file(
    'wheelhouse-manifest.json',
    {WHEELHOUSE_MANIFEST_SHA256!r},
    'Solver closure manifest',
)
solver_manifest = json.loads(solver_manifest_path.read_text())
if solver_manifest.get('schema') != 'tycho-nooa-full-wheelhouse-v1':
    raise RuntimeError('Solver closure manifest has the wrong schema')
solver_wheelhouse = solver_manifest_path.parent
expected_wheels = {{record['filename'] for record in solver_manifest['wheels']}}
actual_wheels = {{path.name for path in solver_wheelhouse.glob('*.whl')}}
if actual_wheels != expected_wheels:
    raise RuntimeError('Solver wheel filenames differ from the frozen manifest')
for record in [*solver_manifest['wheels'], *solver_manifest['support_files']]:
    path = solver_wheelhouse / record['filename']
    if (
        not path.is_file()
        or path.stat().st_size != record['bytes']
        or sha256_file(path) != record['sha256']
    ):
        raise RuntimeError(f'Solver closure file differs: {{record["filename"]}}')
uv = shutil.which('uv')
if uv is None:
    raise RuntimeError('Kaggle uv executable is unavailable')
solver_venv = Path('/tmp/solver-venv')
subprocess.run(
    [uv, 'venv', '--python', sys.executable, str(solver_venv)],
    check=True,
    timeout=60,
)
solver_python = solver_venv / 'bin/python'
subprocess.run(
    [
        uv, 'pip', 'install', '--python', str(solver_python), '--offline',
        '--no-index', '--find-links', str(solver_wheelhouse),
        'arc-agi-3-tycho-nooa==0.1.0',
    ],
    check=True,
    timeout=180,
)
installed_raw = subprocess.check_output(
    [
        str(solver_python), '-c',
        "import importlib.metadata as m,json,re; "
        "c=lambda s:re.sub(r'[-_.]+','-',s).lower(); "
        "print(json.dumps(dict(sorted((c(d.metadata['Name']),d.version) for d in m.distributions()))))",
    ],
    text=True,
    timeout=30,
)
installed = json.loads(installed_raw)
canonical = lambda value: __import__('re').sub(r'[-_.]+', '-', value).lower()
expected = {{canonical(record['name']): record['version'] for record in solver_manifest['wheels']}}
if installed != expected:
    raise RuntimeError('installed Solver distributions differ from the frozen manifest')
pyvenv = (solver_venv / 'pyvenv.cfg').read_text()
if 'include-system-site-packages = false' not in pyvenv:
    raise RuntimeError('Solver venv admits ambient system packages')
Path('/tmp/solver-closure-provenance.json').write_text(json.dumps({{
    'manifest_sha256': {WHEELHOUSE_MANIFEST_SHA256!r},
    'wheel_count': len(expected_wheels),
    'installed_count': len(installed),
    'include_system_site_packages': False,
}}, indent=2) + '\\n')
"""
    notebook["cells"][2]["source"] = install

    notebook["cells"][3]["source"] = """import subprocess
import sys
from pathlib import Path

output_dir = Path('/kaggle/working/tycho-nooa-s0-solver-join')
probe_command = [
    sys.executable,
    '/tmp/s0_solver_join_runtime.py',
    '--output-dir', str(output_dir),
    '--manifest', '/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json',
    '--input-root', '/kaggle/input',
    '--model-source', 'jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1',
    '--runtime-provenance', '/tmp/sglang-runtime-provenance.json',
    '--startup-timeout-s', '1200',
    '--request-timeout-s', '180',
    '--attention-backend', 'triton',
    '--sampling-backend', 'pytorch',
    '--profile-imports',
    '--r8-l3b',
    '--solver-join-probe', '/tmp/s0_solver_join_probe.py',
    '--solver-python', '/tmp/solver-venv/bin/python',
]
subprocess.run(probe_command, check=True)
print((output_dir / 'result.json').read_text())
"""
    notebook["metadata"]["kaggle"]["isInternetEnabled"] = False
    notebook["metadata"]["kaggle"]["accelerator"] = "nvidiaRtx6000"
    return notebook


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(OUTPUT)
