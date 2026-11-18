"""Successor replay with toolkit selection and actual sampling preflight; no launch."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DONOR = HERE.parent / "actor-mtp-20260906"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def replace_once(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new, 1)


def build():
    donor_manifest = json.loads((DONOR / "manifest.json").read_text())
    raw = (DONOR / "actor_mtp.ipynb").read_bytes()
    assert sha(raw) == donor_manifest["notebook_sha256"]
    notebook = json.loads(raw)
    original = (DONOR / "runner.py").read_text()
    assert sha(original.encode()) == donor_manifest["assets"]["runner.py"]
    runner = original.replace(
        "/kaggle/working/actor-mtp-20260906",
        "/kaggle/working/mtp-sampling-repair-20260906",
    )
    runner = replace_once(
        runner,
        "import urllib.request",
        "import urllib.request\nfrom install_sampling_patch import install\nfrom watchdog import guarded_call",
    )
    runner = replace_once(
        runner,
        "def main():\n    OUT.mkdir(exist_ok=True, parents=True)\n",
        "def main():\n    OUT.mkdir(exist_ok=True, parents=True)\n    save(OUT / 'sampling-patch.json', install())\n    os.environ['SGLANG_SPEC_NATIVE_RENORM'] = '1'\n",
    )
    runner = replace_once(
        runner,
        "                post(warm)",
        "                save(directory / 'stochastic-preflight-request.json', warm)\n                warm_response, warm_elapsed = guarded_call(lambda: post(warm), process)\n                save(directory / 'stochastic-preflight-response.json', warm_response)\n                if arm == 'shallow-mtp':\n                    assert warm_response['choices'][0]['meta_info']['spec_verify_ct'] > 0, 'MTP verification not exercised'",
    )
    runner = replace_once(
        runner,
        "response, elapsed = post(request)",
        "response, elapsed = guarded_call(lambda: post(request), process)",
    )
    runner = replace_once(
        runner,
        "            finally:\n                if process is not None:",
        "            finally:\n                original_error = sys.exc_info()[1]\n                if process is not None:",
    )
    runner = replace_once(
        runner,
        "                    except BaseException:\n",
        "                    except BaseException as cleanup_error:\n                        save(directory / 'cleanup-error.json', {'type': type(cleanup_error).__name__, 'error': str(cleanup_error)})\n",
    )
    runner = replace_once(
        runner,
        "                        raise\n    save(",
        "                        if original_error is not None:\n                            raise original_error from cleanup_error\n                        raise\n    save(",
    )
    assets = "".join(notebook["cells"][2]["source"])
    old_encoded = base64.b64encode(original.encode()).decode()
    assets = replace_once(
        assets, old_encoded, base64.b64encode(runner.encode()).decode()
    )
    assets = replace_once(
        assets, donor_manifest["assets"]["runner.py"], sha(runner.encode())
    )
    hashes = {"runner.py": sha(runner.encode())}
    for name in ("install_sampling_patch.py", "sampling_preflight.py", "watchdog.py"):
        data = (HERE / name).read_bytes()
        hashes[name] = sha(data)
        assets += f"\n_data=base64.b64decode({base64.b64encode(data).decode()!r})\nassert hashlib.sha256(_data).hexdigest()=={sha(data)!r}\nPath('/tmp/{name}').write_bytes(_data)\n"
    notebook["cells"][2]["source"] = assets
    notebook["cells"][4]["source"] = "".join(notebook["cells"][4]["source"]).replace(
        "/kaggle/working/actor-mtp-20260906",
        "/kaggle/working/mtp-sampling-repair-20260906",
    )
    notebook["cells"].insert(
        4,
        {
            "cell_type": "code",
            "execution_count": None,
            "outputs": [],
            "metadata": {},
            "source": "import subprocess, sys\nfrom pathlib import Path\npreflight=Path('/kaggle/working/mtp-sampling-preflight'); preflight.mkdir(exist_ok=True)\nwith (preflight/'preflight.log').open('w') as log:\n    result=subprocess.run([sys.executable,'/tmp/sampling_preflight.py','--out',str(preflight)],stdout=log,stderr=subprocess.STDOUT,timeout=600)\nprint((preflight/'preflight.log').read_text()[-12000:])\nassert result.returncode==0, 'Sampling preflight failed; do not load model'\n",
        },
    )
    notebook["cells"][0]["source"] = (
        "# MTP stochastic sampling repair qualification\nPrecompiled native renormalization; kernel preflight before model load; same Actor corpus and sampling. User upload only.\n"
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "notebook", "exec")
    data = (json.dumps(notebook, indent=1) + "\n").encode()
    meta = json.loads((DONOR / "kernel-metadata.json").read_text())
    meta.update(
        id="pranavagrawai/mtp-sampling-repair-20260906",
        title="MTP Sampling Repair 20260906",
        code_file="sampling_repair.ipynb",
    )
    manifest = {
        "schema": "mtp-sampling-repair-v1",
        "donor_notebook_sha256": sha(raw),
        "corpus_sha256": donor_manifest["corpus_sha256"],
        "notebook_sha256": sha(data),
        "assets": hashes,
        "session_cap_s": 7800,
        "kernel_preflight_cap_s": 600,
        "runner_cap_s": 6400,
        "upload_ready": True,
        "scope": "Replay qualification only; integrated relaunch follows successful stochastic MTP evidence",
    }
    return {
        "sampling_repair.ipynb": data,
        "runner.py": runner.encode(),
        "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
        "kernel-metadata.json": (json.dumps(meta, indent=2) + "\n").encode(),
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
    print("READY for bounded user qualification; no launch")
