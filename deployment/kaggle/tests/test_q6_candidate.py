from __future__ import annotations

import base64
import hashlib
import importlib.util
import io
import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parents[1] / "q6-nooa-candidate"
sys.path.insert(0, str(HERE))


def load(name):
    spec = importlib.util.spec_from_file_location("q6_" + name, HERE / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_candidate_payload_contains_every_verified_file_and_shared_sources():
    frozen = load("freeze_candidate").verify()
    builder = load("build_notebook")
    raw = base64.b64decode(builder.join_zip_b64())
    assert raw == base64.b64decode(builder.join_zip_b64())
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert set(frozen["files"]) <= set(archive.namelist())
        for name, expected in frozen["files"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
        assert json.loads(archive.read("CHASSIS.json")) == frozen


def test_arm_notebooks_compile_and_differ_only_in_treatment_flag():
    builder = load("build_notebook")
    control = builder.build(treatment=False)
    treatment = builder.build(treatment=True)
    builder.validate(control)
    builder.validate(treatment)
    assert control == builder.build(treatment=False)
    c = json.dumps(control, sort_keys=True)
    t = json.dumps(treatment, sort_keys=True)
    assert c.replace("ARC_STALL_REVIEW'] = '0'", "ARC_STALL_REVIEW'] = '1'") == t
