"""Separate capture notebook, derived from the hardware-qualified q6 package."""

import argparse
import base64
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / "q6-quick-probe-20260905"
SLUG = "pranavagrawai/actor-capture-20260906"
BASE_SHA = "286a5be2767628e16e079b576364b5beb720f0d3943fc3bd0f57290c16aca430"


def build():
    assert hashlib.sha256((BASE / "probe.ipynb").read_bytes()).hexdigest() == BASE_SHA
    notebook = json.loads((BASE / "probe.ipynb").read_text())
    old = (BASE / "exact_request_outer_runtime.py").read_bytes()
    target = '                    base_url.rstrip("/") + "/v1",'
    assert old.decode().count(target) == 1
    runtime = (
        old.decode()
        .replace(target, '                    "http://127.0.0.1:30001/v1",')
        .encode()
    )
    assets = "import hashlib\n" + "".join(notebook["cells"][1]["source"])
    encoded = base64.b64encode(old).decode()
    assert assets.count(encoded) == 1
    assets = assets.replace(encoded, base64.b64encode(runtime).decode())
    assets = assets.replace(
        hashlib.sha256(old).hexdigest(), hashlib.sha256(runtime).hexdigest()
    )
    proxy = (HERE / "capture_proxy.py").read_bytes()
    assets += f"\n_capture=base64.b64decode({base64.b64encode(proxy).decode()!r})\nassert hashlib.sha256(_capture).hexdigest()=={hashlib.sha256(proxy).hexdigest()!r}\nPath('/tmp/capture_proxy.py').write_bytes(_capture)\n"
    notebook["cells"][1]["source"] = assets
    launch = "".join(notebook["cells"][3]["source"])
    assert launch.count("subprocess.run(command, check=True)") == 1
    wrapper = """import importlib.util
spec=importlib.util.spec_from_file_location('capture_proxy','/tmp/capture_proxy.py')
capture=importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)
capture_dir=Path('/kaggle/working/actor-capture-20260906')
proxy,proxy_thread=capture.start(capture_dir)
try:
    runtime_spec=importlib.util.spec_from_file_location('capture_runtime','/tmp/official_launcher_runtime.py')
    runtime=importlib.util.module_from_spec(runtime_spec)
    sys.modules['capture_runtime']=runtime
    runtime_spec.loader.exec_module(runtime)
    model_path=runtime.discover_model(Path('/kaggle/input'),Path('/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json'))
    for filename in ['config.json','generation_config.json']:
        shutil.copy2(model_path/filename,capture_dir/filename)
    subprocess.run(command, check=True)
finally:
    proxy.shutdown()
    proxy.server_close()
    proxy_thread.join(timeout=5)
    (capture_dir/'capture-summary.json').write_text(json.dumps({'recorded_attempts':proxy.count,'max_requests':proxy.max_requests,'status':'CAPTURED_REVIEW_REQUIRED','no_exact_corpus_claim_until_review':True},indent=2))
"""
    launch = launch.replace("subprocess.run(command, check=True)", wrapper)
    # Collection is not a competition submission, including placeholder output.
    launch = launch[: launch.index("submission = pd.DataFrame(")]
    notebook["cells"][3]["source"] = launch
    notebook["cells"][0]["source"] = (
        "# Actor request capture\nQualified q6 control host, ls20, 10 actions. One-hour cap. No MTP or solver behavior changes. User upload only.\n"
    )
    preflight = "import subprocess\nassert 'RTX PRO 6000' in subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)\n"
    notebook["cells"].insert(
        1,
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": preflight,
        },
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), "<notebook>", "exec")
    data = (json.dumps(notebook, indent=1) + "\n").encode()
    metadata = json.loads((BASE / "kernel-metadata.json").read_text())
    metadata.update(id=SLUG, title="Actor capture 20260906", code_file="capture.ipynb")
    manifest = {
        "baseline_notebook_sha256": BASE_SHA,
        "notebook_sha256": hashlib.sha256(data).hexdigest(),
        "runtime_before_sha256": hashlib.sha256(old).hexdigest(),
        "runtime_after_sha256": hashlib.sha256(runtime).hexdigest(),
        "proxy_sha256": hashlib.sha256(proxy).hexdigest(),
        "delta": "Only launcher model URL routed through recorder; solver ZIP and engine setup unchanged",
        "session_cap_s": 3600,
        "gameplay_cap_s": 1800,
        "actions": 10,
        "max_recorded_attempts": 96,
        "slug": SLUG,
    }
    return {
        "capture.ipynb": data,
        "kernel-metadata.json": (json.dumps(metadata, indent=2) + "\n").encode(),
        "manifest.json": (json.dumps(manifest, indent=2) + "\n").encode(),
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
    print("Verified" if args.check else "Built; not uploaded")
