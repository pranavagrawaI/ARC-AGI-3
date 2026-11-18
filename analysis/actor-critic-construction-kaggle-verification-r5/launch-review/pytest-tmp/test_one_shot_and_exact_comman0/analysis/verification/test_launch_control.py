"""Offline entrypoint tests. All provider subprocess calls are intercepted.
Run: deployment/kaggle/.venv/bin/python -m pytest -q THIS_FILE
Fixtures create only disposable tmp_path state, never the real authorization.
"""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

SPEC = importlib.util.spec_from_file_location("launch_gate_under_test", Path(__file__).with_name("launch_control.py"))
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def world(tmp_path, monkeypatch):
    real_script = Path(gate.__file__)
    root = tmp_path
    run = root / "analysis" / gate.IDENTITY
    package = root / "deployment/kaggle/actor-critic-construction-kaggle-r5"
    script = root / "analysis/verification/launch_control.py"
    review = root / "analysis/verification/repaired-review/DECISION.json"
    for key, value in dict(ROOT=root, RUN_DIR=run, PACKAGE_DIR=package, SCRIPT=script, LOCAL_REVIEW=review).items():
        monkeypatch.setattr(gate, key, value)
    run.mkdir(parents=True)
    package.mkdir(parents=True)
    script.parent.mkdir(parents=True)
    script.write_bytes(real_script.read_bytes())
    script.with_name("test_launch_control.py").write_bytes(Path(__file__).read_bytes())
    (package / "construction.ipynb").write_text("offline notebook")
    monkeypatch.setattr(gate, "NOTEBOOK_HASH", gate.sha(package / "construction.ipynb"))
    save(package / "kernel-metadata.json", dict(id=gate.TARGET, is_private=True, enable_gpu=True,
        enable_tpu=False, enable_internet=False, machine_shape="NvidiaRtxPro6000",
        dataset_sources=gate.DATASETS, competition_sources=["arc-prize-2026-arc-agi-3"],
        kernel_sources=[], model_sources=[], language="python", kernel_type="notebook", code_file="construction.ipynb"))
    save(review, dict(status="APPROVED_LOCAL_PACKAGE", manifest_sha256=gate.LOCAL_MANIFEST_HASH,
                     notebook_sha256=gate.NOTEBOOK_HASH, launch_authorized=False))
    (run / "EXPERIMENT-CARD.md").write_text("# Offline bounded experiment\n")
    policy = dict(schema=1, identity=gate.IDENTITY, target=gate.TARGET, limits=gate.LIMITS.copy(),
                  private=True, internet=False, accelerator="NvidiaRtxPro6000", competition_submission=False,
                  dataset_push=False, local_monitor_cancels_remote=False)
    manifest = dict(schema=1, identity=gate.IDENTITY, run_dir=str(run), package_dir=str(package),
                    target=gate.TARGET, card_sha256=gate.sha(run / "EXPERIMENT-CARD.md"),
                    notebook_sha256=gate.NOTEBOOK_HASH, policy=policy, sources={})
    files = [script, script.with_name("test_launch_control.py"), review, run / "EXPERIMENT-CARD.md", *package.iterdir()]
    def freeze():
        manifest["sources"] = {str(p.relative_to(root)):gate.sha(p) for p in files}
        save(run / "manifest.json", manifest)
        digest = gate.sha(run / "manifest.json")
        save(run / "launch-review.json", dict(status="APPROVED", manifest_sha256=digest))
        save(run / "authorization.json", dict(schema=1, identity=gate.IDENTITY, target=gate.TARGET,
            manifest_sha256=digest, card_sha256=manifest["card_sha256"], notebook_sha256=gate.NOTEBOOK_HASH,
            max_pushes=1, retries=0, timeout_seconds=3600, used=False))
    freeze()
    calls = []
    responses = []
    success = f"Kernel version 7 successfully pushed.  Please check progress at https://www.kaggle.com/code/{gate.TARGET}"
    def fake(argv, **kwargs):
        calls.append(argv)
        assert kwargs["stdin"] is subprocess.DEVNULL
        if responses:
            result = responses.pop(0)
            if callable(result):
                return result(argv, kwargs)
            if isinstance(result, BaseException):
                raise result
            code, text = result
        elif "quota" in argv:
            code, text = 0, '[{"resource":"GPU","remaining":"28.47h"}]'
        elif "list" in argv:
            code, text = 0, "Not found"
        elif "push" in argv:
            assert (run / "CONSUMED.json").exists()
            code, text = 0, success
        else:
            code, text = 0, json.dumps(dict(target=gate.TARGET, version=7, status="COMPLETE"))
        return subprocess.CompletedProcess(argv, code, text, "")
    monkeypatch.setattr(gate.subprocess, "run", fake)
    return types.SimpleNamespace(run=run, package=package, calls=calls, responses=responses,
        manifest=manifest, freeze=freeze, success=success, review=review, script=script)


def invoke(w, action="push", *extra):
    return gate.main([action, "--run-dir", str(w.run), *extra])


def assert_no_push(w):
    assert not any("push" in c for c in w.calls)


def test_green_check_is_completely_offline(world):
    assert invoke(world, "check") == 0
    assert world.calls == []


def test_one_shot_and_exact_command(world):
    assert invoke(world) == 0
    assert world.calls[-1] == [gate.CLI, "kernels", "push", "-p", str(world.package), "--timeout", "3600", "--accelerator", "NvidiaRtxPro6000"]
    launch = gate.load(world.run / "launch.json")
    assert launch["version"] == 7
    assert invoke(world) == 2
    assert sum("push" in c for c in world.calls) == 1


@pytest.mark.parametrize("name", ["authorization.json", "launch-review.json", "manifest.json"])
@pytest.mark.parametrize("bad", [None, {}, [], "string", 1, False])
def test_malformed_top_level(world, name, bad):
    save(world.run / name, bad)
    assert invoke(world) == 2
    assert world.calls == []


@pytest.mark.parametrize("field", ["schema", "identity", "target", "manifest_sha256", "card_sha256", "notebook_sha256", "max_pushes", "retries", "timeout_seconds", "used"])
@pytest.mark.parametrize("mutation", ["missing", "wrong", "wrong_type"])
def test_total_authorization(world, field, mutation):
    auth = gate.load(world.run / "authorization.json")
    if mutation == "missing":
        del auth[field]
    elif mutation == "wrong_type":
        auth[field] = [auth[field]]
    else:
        auth[field] = True if field == "used" else "WRONG"
    save(world.run / "authorization.json", auth)
    assert invoke(world) == 2
    assert world.calls == []


@pytest.mark.parametrize("field,value", [("max_pushes", True), ("retries", False), ("timeout_seconds",3600.0), ("used",0)])
def test_boolean_integer_confusion(world, field, value):
    auth = gate.load(world.run / "authorization.json")
    auth[field] = value
    save(world.run / "authorization.json", auth)
    assert invoke(world) == 2
    assert world.calls == []


@pytest.mark.parametrize("review", [{"status":"HOLD"}, {"manifest_sha256":"MATCH"}, {"status":"APPROVED", "manifest_sha256":"wrong"}])
def test_negative_review(world, review):
    if review.get("manifest_sha256") == "MATCH":
        review["manifest_sha256"] = gate.sha(world.run / "manifest.json")
    save(world.run / "launch-review.json", review)
    assert invoke(world) == 2
    assert world.calls == []


def test_missing_authorization(world):
    (world.run / "authorization.json").unlink()
    assert invoke(world) == 2
    assert world.calls == []


@pytest.mark.parametrize("kind", ["script", "test", "card", "notebook", "extra_package", "local_review", "coverage"])
def test_source_drift(world, kind):
    if kind == "coverage":
        world.manifest["sources"].pop(str(world.script.relative_to(gate.ROOT)))
        save(world.run / "manifest.json", world.manifest)
    elif kind == "extra_package":
        (world.package / "unreviewed.py").write_text("unsafe")
    else:
        paths = dict(script=world.script, test=world.script.with_name("test_launch_control.py"),
                     card=world.run / "EXPERIMENT-CARD.md", notebook=world.package / "construction.ipynb", local_review=world.review)
        paths[kind].write_text("changed")
    assert invoke(world) == 2
    assert world.calls == []


@pytest.mark.parametrize("field,value", [("is_private",False), ("enable_internet",True), ("enable_gpu",1), ("dataset_sources",[]), ("competition_sources",[]), ("id","other/target"), ("machine_shape","T4")])
def test_even_rehashed_wrong_metadata_is_rejected(world, field, value):
    metadata = gate.load(world.package / "kernel-metadata.json")
    metadata[field] = value
    save(world.package / "kernel-metadata.json", metadata)
    world.freeze()
    assert invoke(world) == 2
    assert world.calls == []


@pytest.mark.parametrize("field,value", [("run_dir","relative"), ("package_dir","relative"), ("target","other/target"), ("schema",True)])
def test_wrong_manifest_binding(world, field, value):
    world.manifest[field] = value
    world.freeze()
    assert invoke(world) == 2
    assert world.calls == []


@pytest.mark.parametrize("code,text", [(1,"Not found"),(0,""),(0,"{}"),(0,"null"),(0,'[{"ref":"other"}]'),(0,"Not found\nwarning")])
def test_target_absence_fail_closed(world, code, text):
    world.responses[:] = [(0,'[{"resource":"GPU","remaining":"28.47h"}]'), (code,text)]
    assert invoke(world) == 2
    assert_no_push(world)
    assert not (world.run / "CONSUMED.json").exists()


@pytest.mark.parametrize("text", ["Not found", "[]"])
def test_only_supported_absence(world, text):
    world.responses[:] = [(0,'[{"resource":"GPU","remaining":"1.00h"}]'), (0,text)]
    assert invoke(world) == 0


@pytest.mark.parametrize("code,text", [(1,'[{"resource":"GPU","remaining":"28.47h"}]'), (0,"[]"), (0,'[{"resource":"GPU","remaining":"0.99h"}]'), (0,'[{"resource":"GPU","remaining":true}]'), (0,'[{"resource":"GPU","remaining":"NaNh"}]'), (0,'{}')])
def test_quota_fail_closed(world, code, text):
    world.responses[:] = [(code,text)]
    assert invoke(world) == 2
    assert_no_push(world)


def test_source_rechecked_after_provider_reads(world):
    def drift(argv, kwargs):
        world.script.write_text("changed during preflight")
        return subprocess.CompletedProcess(argv,0,"Not found","")
    world.responses[:] = [(0,'[{"resource":"GPU","remaining":"28.47h"}]'), drift]
    assert invoke(world) == 2
    assert_no_push(world)


def test_atomic_race_retains_marker(world, monkeypatch):
    original = gate.exclusive
    def race(path, value):
        if path.name == "CONSUMED.json":
            original(path, {"other_owner":True})
        return original(path,value)
    monkeypatch.setattr(gate,"exclusive",race)
    assert invoke(world) == 2
    assert_no_push(world)
    assert gate.load(world.run / "CONSUMED.json") == {"other_owner":True}


@pytest.mark.parametrize("outcome", ["timeout","failure","malformed","wrong_url","warning"])
def test_uncertain_push_never_retries(world, outcome):
    bad = dict(timeout=subprocess.TimeoutExpired("push",180,output=b"partial"), failure=(1,"failed"),
               malformed=(0,"pushed"), wrong_url=(0,world.success.replace(gate.TARGET,"other/kernel")),
               warning=(0,"The following are not valid dataset sources and could not be added\n" + world.success))
    world.responses[:] = [(0,'[{"resource":"GPU","remaining":"28.47h"}]'),(0,"Not found"),bad[outcome]]
    assert invoke(world) == 2
    assert (world.run / "CONSUMED.json").exists()
    assert len(list(world.run.glob("push-*.json"))) == 1
    assert invoke(world) == 2
    assert sum("push" in c for c in world.calls) == 1
    if outcome == "warning":
        assert gate.load(world.run / "launch.json")["version"] == 7
        assert invoke(world,"monitor") == 0


@pytest.mark.parametrize("terminal", ["COMPLETE","ERROR","CANCELLED"])
def test_monitor_exact_version_and_transient_recovery(world, terminal):
    assert invoke(world) == 0
    world.calls.clear()
    world.responses[:] = [(1,"transient"),(0,"not-json"),(0,json.dumps(dict(target=gate.TARGET,version=7,status=terminal)))]
    assert invoke(world,"monitor","--interval","0.001") == 0
    assert len(world.calls) == 3
    assert all(c == [gate.SDK_PYTHON,"-c",gate.STATUS_PROGRAM,gate.TARGET,"7"] for c in world.calls)
    assert len(list(world.run.glob("status-*.json"))) == 3


def test_monitor_deadline_not_remote_shutdown(world, capsys):
    assert invoke(world) == 0
    world.calls.clear()
    world.responses[:] = [(0,json.dumps(dict(target=gate.TARGET,version=8,status="COMPLETE")))] * 100
    assert invoke(world,"monitor","--deadline-seconds","0.005","--interval","0.001") == 3
    assert '"remote_stop_established": false' in capsys.readouterr().out
    assert_no_push(world)
    assert not list(world.run.glob("terminal-*.json"))


def test_monitor_checks_receipt_integrity(world):
    assert invoke(world) == 0
    next(world.run.glob("push-*.json")).write_text("tampered")
    world.calls.clear()
    assert invoke(world,"monitor") == 2
    assert world.calls == []


@pytest.mark.parametrize("provider,normalized", [("COMPLETE","COMPLETE"), ("CANCEL_ACKNOWLEDGED","CANCELLED")])
def test_status_program_sets_sdk_version_label(monkeypatch, capsys, provider, normalized):
    """Execute the actual adapter, not just assert text contains a field."""
    observed = []
    class Request:
        pass
    class Client:
        def __enter__(self):
            return types.SimpleNamespace(kernels=types.SimpleNamespace(kernels_api_client=self))
        def __exit__(self,*args):
            pass
        def get_kernel_session_status(self, request):
            observed.append((request.user_name,request.kernel_slug,request.version_label))
            return types.SimpleNamespace(status=types.SimpleNamespace(name=provider),failure_message="")
    class Api:
        def authenticate(self):
            pass
        def build_kaggle_client(self):
            return Client()
    monkeypatch.setitem(sys.modules,"kaggle.api.kaggle_api_extended",types.SimpleNamespace(KaggleApi=Api))
    monkeypatch.setitem(sys.modules,"kagglesdk.kernels.types.kernels_api_service",types.SimpleNamespace(ApiGetKernelSessionStatusRequest=Request))
    monkeypatch.setattr(sys,"argv",["adapter",gate.TARGET,"7"])
    exec(gate.STATUS_PROGRAM,{})
    assert observed == [("pranavagrawai","actor-critic-construction-kaggle-r5","7")]
    assert json.loads(capsys.readouterr().out) == dict(target=gate.TARGET,version=7,status=normalized,provider_status=provider,failure_message="")


@pytest.mark.parametrize("name", ["launch-review.json", "manifest.json"])
def test_missing_control_file(world, name):
    (world.run / name).unlink()
    assert invoke(world) == 2
    assert world.calls == []


@pytest.mark.parametrize("status", ["HOLD", "APPROVED", None])
def test_local_review_is_separate_and_affirmative(world, status):
    review = gate.load(world.review)
    if status is None:
        del review["status"]
    else:
        review["status"] = status
    save(world.review, review)
    world.freeze()
    assert invoke(world) == 2
    assert world.calls == []


@pytest.mark.parametrize("bad", [None, [], {}, "digest", {"../escape":"0" * 64}, {"safe":False}])
def test_malformed_sources(world, bad):
    world.manifest["sources"] = bad
    save(world.run / "manifest.json", world.manifest)
    assert invoke(world) == 2
    assert world.calls == []


def test_empty_consumption_marker_is_irrevocable(world):
    (world.run / "CONSUMED.json").touch()
    assert invoke(world) == 2
    assert world.calls == []
    assert (world.run / "CONSUMED.json").read_bytes() == b""


def test_run_path_requires_literal_identity(world):
    assert gate.main(["push", "--run-dir", str(world.run) + "/../" + gate.IDENTITY]) == 2
    assert world.calls == []


def test_duplicate_json_keys_rejected(world):
    p = world.run / "authorization.json"
    p.write_text(p.read_text()[:-1] + ', "used": false}')
    assert invoke(world) == 2
    assert world.calls == []


def test_authorization_rechecked_after_provider_reads(world):
    def revoke(argv, kwargs):
        auth = gate.load(world.run / "authorization.json")
        auth["used"] = True
        save(world.run / "authorization.json", auth)
        return subprocess.CompletedProcess(argv,0,"Not found","")
    world.responses[:] = [(0,'[{"resource":"GPU","remaining":"28.47h"}]'), revoke]
    assert invoke(world) == 2
    assert_no_push(world)
