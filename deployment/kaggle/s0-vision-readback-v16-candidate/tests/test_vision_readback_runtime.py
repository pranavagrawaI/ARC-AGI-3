"""Checks on the outer runtime's V16 vision-readback mode.

The mode reuses the R8-L3B lifecycle wholesale — model hash verification, SGLang launch,
warmup, teardown classification — and only swaps what runs against a ready server. These
tests pin the seams where that reuse could go wrong: the mode must not be usable without
its lifecycle owner, must not co-exist with the game launcher, and must declare a claim
ceiling that refuses gameplay.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

CANDIDATE = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def runtime():
    spec = importlib.util.spec_from_file_location(
        "v16_outer_runtime", CANDIDATE / "exact_request_outer_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


BASE_ARGS = [
    "runtime",
    "--output-dir", "/tmp/out",
    "--attention-backend", "triton",
    "--sampling-backend", "pytorch",
    "--startup-timeout-s", "1200",
    "--request-timeout-s", "180",
]


def parse(runtime, extra: list[str], monkeypatch):
    monkeypatch.setattr(sys, "argv", BASE_ARGS + extra)
    return runtime.parse_args()


def vision_args(runtime) -> list[str]:
    return [
        "--r8-l3b",
        "--port", str(runtime.R8_L3B_PORT),
        "--vision-readback-probe", "/tmp/probe.py",
        "--vision-frames", "/tmp/frames.json",
    ]


def test_vision_mode_parses_with_its_lifecycle_owner(runtime, monkeypatch):
    args = parse(runtime, vision_args(runtime), monkeypatch)
    assert args.vision_readback_probe == Path("/tmp/probe.py")
    assert args.vision_frames == Path("/tmp/frames.json")
    assert args.vision_probe_timeout_s == 5400.0


def test_probe_and_frames_are_required_together(runtime, monkeypatch):
    with pytest.raises(SystemExit):
        parse(runtime, [
            "--r8-l3b", "--port", str(runtime.R8_L3B_PORT),
            "--vision-readback-probe", "/tmp/probe.py",
        ], monkeypatch)
    with pytest.raises(SystemExit):
        parse(runtime, [
            "--r8-l3b", "--port", str(runtime.R8_L3B_PORT),
            "--vision-frames", "/tmp/frames.json",
        ], monkeypatch)


def test_vision_mode_requires_the_r8_l3b_lifecycle(runtime, monkeypatch):
    """Without --r8-l3b there is no teardown/containment gate to inherit."""
    with pytest.raises(SystemExit):
        parse(runtime, [
            "--vision-readback-probe", "/tmp/probe.py",
            "--vision-frames", "/tmp/frames.json",
        ], monkeypatch)


def test_vision_mode_cannot_run_alongside_the_game_launcher(runtime, monkeypatch):
    with pytest.raises(SystemExit):
        parse(runtime, vision_args(runtime) + [
            "--official-launcher-probe", "/tmp/launcher.py",
            "--launcher-python", "/tmp/python",
        ], monkeypatch)


def test_vision_probe_timeout_must_be_positive(runtime, monkeypatch):
    with pytest.raises(SystemExit):
        parse(runtime, vision_args(runtime) + ["--vision-probe-timeout-s", "0"], monkeypatch)


def test_vision_mode_keeps_the_qualified_backend_requirement(runtime, monkeypatch):
    """R8-L3B pinned Triton/PyTorch; the probe must measure that same route."""
    monkeypatch.setattr(sys, "argv", [
        "runtime", "--output-dir", "/tmp/out",
        "--attention-backend", "flashinfer",
        "--startup-timeout-s", "1200", "--request-timeout-s", "180",
    ] + vision_args(runtime))
    with pytest.raises(SystemExit):
        runtime.parse_args()


def test_the_notebooks_actual_command_line_parses(runtime, monkeypatch):
    """The test kernel version 1 needed and did not have.

    V16 kernel 1 died in 62 s because the notebook passed `--request-timeout-s 600`
    while the R8-L3B guard pins it to 180. Every other test here parsed an argument list
    written by hand in this file, so all of them passed while the run could not start.
    Extract the command the builder actually emits and feed that through parse_args.
    """
    command = notebook_command()
    assert command[0] == "sys.executable"
    monkeypatch.setattr(sys, "argv", ["runtime", *command[2:]])
    args = runtime.parse_args()
    assert args.r8_l3b is True
    assert args.vision_readback_probe is not None
    assert args.vision_frames is not None


def test_the_notebook_gives_the_probe_room_for_the_incumbent_arm(runtime, monkeypatch):
    """text_think decodes up to 16,000 tokens at ~41 tok/s, so ~390 s per request."""
    monkeypatch.setattr(sys, "argv", ["runtime", *notebook_command()[2:]])
    args = runtime.parse_args()
    assert args.request_timeout_s == 180, "R8-L3B pins the frozen one-shot request"
    assert args.vision_request_timeout_s >= 400, "the probe needs its own ceiling"
    assert args.vision_request_timeout_s <= args.vision_probe_timeout_s


def notebook_command() -> list[str]:
    """The argument list the built notebook runs, read out of the run cell."""
    import ast
    import json

    notebook = json.loads((CANDIDATE / "s0_vision_readback.ipynb").read_text())
    run_cell = "".join(notebook["cells"][3]["source"])
    tree = ast.parse(run_cell)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and any(getattr(t, "id", None) == "command" for t in node.targets)
            and isinstance(node.value, ast.List)
        ):
            out = []
            for element in node.value.elts:
                if isinstance(element, ast.Constant):
                    out.append(element.value)
                elif isinstance(element, ast.Attribute):
                    out.append(f"{element.value.id}.{element.attr}")  # sys.executable
                elif isinstance(element, ast.Call):
                    out.append("/kaggle/working/placeholder")  # str(output_dir) etc.
                else:  # pragma: no cover - the builder emits nothing else
                    raise AssertionError(f"unhandled command element {ast.dump(element)}")
            return out
    raise AssertionError("run cell has no `command` list")


def test_claim_ceiling_refuses_gameplay_and_rhae(runtime):
    source = (CANDIDATE / "exact_request_outer_runtime.py").read_text()
    start = source.index('if args.vision_readback_probe:\n        claim_ceiling = (')
    ceiling = source[start : source.index("elif args.official_launcher_probe:", start)]
    assert "plays no game" in ceiling
    assert "commits no environment action" in ceiling
    for refused in ("RHAE", "level completion", "gameplay"):
        assert refused in ceiling


def test_mode_name_is_distinct(runtime):
    source = (CANDIDATE / "exact_request_outer_runtime.py").read_text()
    assert '"s0-vision-readback"' in source
    assert source.count('"s0-vision-readback"') == 1


def test_request_reconciliation_is_enforced_not_merely_recorded(runtime):
    """A count that does not match SGLang's access log must abort, not be written."""
    source = (CANDIDATE / "exact_request_outer_runtime.py").read_text()
    block_start = source.index('stage = "vision_readback"')
    block = source[block_start : source.index('if args.official_launcher_probe:', block_start)]
    assert 'if not request_count_facts["verified"]:' in block
    assert "raise RuntimeError(" in block.split('if not request_count_facts["verified"]:')[1]
    # the probe's own success is a precondition, not an outcome to summarise
    assert 'vision_result.get("status") != "passed"' in block
