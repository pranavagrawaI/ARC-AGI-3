"""Package a frozen solver with shallow MTP. No network calls or launch."""

import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DONOR = (
    ROOT
    / "analysis/deployment-probes/20260906-q6-prediction-feedback/long-run/candidate"
)
DONOR_NOTEBOOK_SHA = "f40d36d33deed4ac909c515247795a581aa54af2ca092eefae49e6c7af223893"
FLAGS = [
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    "1",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "2",
    "--enable-metrics",
]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def build():
    raw = (DONOR / "probe.ipynb").read_bytes()
    assert sha(raw) == DONOR_NOTEBOOK_SHA, "Frozen donor notebook drift"
    notebook = json.loads(raw)
    identity = json.loads((DONOR / "CHASSIS.json").read_text())
    source = (DONOR / "exact_request_outer_runtime.py").read_text()
    assert sha(source.encode()) == identity["sources"]["exact_request_outer_runtime.py"]
    fn = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "sglang_command"
    )
    old_function = "\n".join(source.splitlines()[fn.lineno - 1 : fn.end_lineno])
    assert old_function.count("    return command") == 1
    new_function = old_function.replace(
        "    return command", f"    command.extend({FLAGS!r})\n    return command"
    )
    modified = source.replace(old_function, new_function, 1)
    # Snapshot counters before orderly shutdown; telemetry failure must never
    # prevent the existing lifecycle observer from retiring the server.
    lifecycle_call = "                lifecycle = qualify_lifecycle(process, server_log, output_dir, port=args.port)"
    assert modified.count(lifecycle_call) == 1
    telemetry = "\n".join(
        [
            "                try:",
            "                    write_json(output_dir / 'mtp-server-info-before-teardown.json', http_json(f'http://127.0.0.1:{args.port}/server_info', timeout_s=5))",
            "                except Exception as telemetry_error:",
            "                    print(f'MTP telemetry unavailable: {telemetry_error}', flush=True)",
        ]
    )
    modified = modified.replace(lifecycle_call, telemetry + "\n" + lifecycle_call)
    compile(modified, "integrated-runtime", "exec")
    assets = "".join(notebook["cells"][1]["source"])
    old_encoded = base64.b64encode(source.encode()).decode()
    assert assets.count(old_encoded) == 1
    assets = assets.replace(old_encoded, base64.b64encode(modified.encode()).decode())
    # Check the actual notebook-embedded worker, not a loose-source import path.
    archives = []
    for node in ast.walk(ast.parse(assets)):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "b64decode"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            data = base64.b64decode(node.args[0].value)
            if data.startswith(b"PK"):
                archives.append(data)
    matches = []
    for data in archives:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            if "chassis/prediction_feedback.py" not in archive.namelist():
                continue
            for name, digest in identity["files"].items():
                assert sha(archive.read(name)) == digest, name
            matches.append(data)
    assert len(matches) == 1
    manifest = {
        "schema": "q6-mtp-integration-v1",
        "upload_ready": True,
        "donor_notebook_sha256": DONOR_NOTEBOOK_SHA,
        "solver_contract_sha256": identity["contract_sha256"],
        "solver_files": identity["files"],
        "join_zip_sha256": sha(matches[0]),
        "runtime_sha256": sha(modified.encode()),
        "engine_delta": FLAGS,
        "game": "ls20",
        "max_actions": 10,
        "admission_seconds": 1800,
        "worker_timeout_seconds": 900,
        "launcher_timeout_seconds": 3000,
        "session_timeout_seconds": 4800,
        "limits": "Integration pilot, not matched gameplay or a deployment verdict. Frozen prediction-feedback solver remains a candidate. No additional tool recovery or reasoning changes.",
    }
    assets += (
        "\n"
        + "\n".join(
            [
                "import hashlib",
                f"assert hashlib.sha256(Path('/tmp/official_launcher_runtime.py').read_bytes()).hexdigest() == {manifest['runtime_sha256']!r}",
                f"assert hashlib.sha256(Path('/tmp/nooa-join.zip').read_bytes()).hexdigest() == {manifest['join_zip_sha256']!r}",
                f"Path('/kaggle/working/INTEGRATION-IDENTITY.json').write_text({json.dumps(manifest, indent=2)!r})",
            ]
        )
        + "\n"
    )
    notebook["cells"][1]["source"] = assets
    run = "".join(notebook["cells"][3]["source"])
    for old, new in [
        ("TYCHO_KAGGLE_MAX_ACTIONS'] = '100'", "TYCHO_KAGGLE_MAX_ACTIONS'] = '10'"),
        (
            "TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '7200'",
            "TYCHO_GAMEPLAY_ADMISSION_SECONDS'] = '1800'",
        ),
        ("'--launcher-timeout-s', '8400'", "'--launcher-timeout-s', '3000'"),
    ]:
        assert run.count(old) == 1, old
        run = run.replace(old, new)
    notebook["cells"][3]["source"] = run
    notebook["cells"][0]["source"] = (
        "# Frozen Q6 prediction-feedback solver + shallow MTP\nPrivate offline ls20 pilot, 10 actions, user launch only. No competition submission. Read the experiment card.\n"
    )
    preflight = {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": "import os, subprocess\nassert not os.getenv('KAGGLE_IS_COMPETITION_RERUN'), 'Private qualifier only'\n_gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()\nassert 'RTX PRO 6000' in _gpu, _gpu\n",
    }
    notebook["cells"].insert(1, preflight)
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
    output = (json.dumps(notebook, indent=1) + "\n").encode()
    manifest["notebook_sha256"] = sha(output)
    metadata = json.loads((DONOR / "kernel-metadata.json").read_text())
    metadata.update(
        id="pranavagrawai/q6-mtp-integration-20260906",
        title="Q6 MTP Integration 20260906",
        code_file="q6_mtp.ipynb",
    )
    return {
        "q6_mtp.ipynb": output,
        "runtime.py": modified.encode(),
        "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
        "kernel-metadata.json": (json.dumps(metadata, indent=2) + "\n").encode(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, data in build().items():
        if args.check:
            assert (HERE / name).read_bytes() == data, name
        else:
            (HERE / name).write_bytes(data)
    print("READY: frozen solver identities verified; no Kaggle launch")
