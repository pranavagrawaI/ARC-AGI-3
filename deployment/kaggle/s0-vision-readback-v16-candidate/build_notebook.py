"""Build the frozen V16 vision-readback candidate notebook.

V16 keeps the whole R8-L3B lifecycle the route already qualified — the same pinned
checkpoint hashes, the same patched SGLang wheel, the same Triton/PyTorch backends,
warmup, and TERM-only teardown — and replaces the game launcher with a perception probe.
It therefore needs neither the Solver wheelhouse nor the ARC-AGI-3-Agents launcher, so
the V15 additions to the install cell are deliberately not carried over.

See ``analysis/deployment-probes/20260903T-vision-readback-v16``.
"""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
BASE = (
    ROOT
    / "analysis/deployment-probes/20260831T160520Z-qwen38-r8l3b-kaggle-v14"
    / "remote-source-v14/prototype-qwen3-8-fp8-r8-b-request-path.ipynb"
)
OUTPUT = HERE / "s0_vision_readback.ipynb"
RUNTIME = HERE / "exact_request_outer_runtime.py"
PROBE = HERE / "vision_readback_probe.py"
FRAMES = HERE / "frames.json"

# The pinned ls20 evidence. Its hash is checked at build time so a rebuilt notebook can
# never quietly score against a different set of frames than the card declares.
FRAMES_SHA256 = "fd4031afec63975cd5bc98e8cc32445f54b510ac927aa486708447ca02b2d21d"

TITLE = (
    "# VISION READBACK V16 — can the served model read a 64x64 ARC frame from an image?\n\n"
    "The V15 deployment sent the actor a pipe-delimited text grid and nothing else, and "
    "three of its six first-decision calls spent 16,000 reasoning tokens each parsing "
    "that text without emitting a tool call. Tycho's `agent/vision.py` claims Qwen is "
    "the one model family whose encoder resolves such a grid exactly — 32 px per merged "
    "visual token, so a 64x64 grid drawn at 32 px/cell is 2048x2048 and costs 4,096 "
    "visual tokens, one per ARC cell — but conditions that on the serving processor "
    "admitting 4,194,304 pixels without downscaling. That is a property of this SGLang "
    "deployment and has never been measured.\n\n"
    "This notebook measures it two ways on three recorded public ls20 frames: the "
    "prompt-token cost of each render scale, by differencing an identical request with "
    "and without the image; and per-cell readback, palette recovery, and bounding-box "
    "accuracy across text, image at three scales, and both together. The incumbent "
    "text-with-thinking contract runs as a reference arm on one frame.\n\n"
    "It plays no game, commits no environment action, creates no competition "
    "submission, and refuses `KAGGLE_IS_COMPETITION_RERUN`. It keeps the accepted "
    "R8-L3B model-server route, warmup, and TERM-only lifecycle unchanged."
)


def encoded(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict[str, Any]:
    if sha256(FRAMES) != FRAMES_SHA256:
        raise RuntimeError("pinned ls20 frame asset drifted from the frozen manifest")

    notebook = json.loads(BASE.read_text())
    notebook["cells"][0]["source"] = TITLE

    # Cell 1 already writes the model manifest and the R8-L3B probe; append only what
    # V16 adds. Cell 2 (the SGLang runtime install) is carried over untouched.
    assets = "".join(notebook["cells"][1]["source"])
    for destination, source in (
        ("/tmp/vision_outer_runtime.py", RUNTIME),
        ("/tmp/vision_readback_probe.py", PROBE),
        ("/tmp/vision-readback-frames.json", FRAMES),
    ):
        assets += (
            f"\nPath({destination!r}).write_bytes("
            f"base64.b64decode({encoded(source)!r}))\n"
        )
    notebook["cells"][1]["source"] = assets

    notebook["cells"][3]["source"] = """import json
import os
import subprocess
import sys
from pathlib import Path

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    raise RuntimeError('vision-readback probe refuses competition rerun mode')

frames = json.loads(Path('/tmp/vision-readback-frames.json').read_text())
if frames.get('schema') != 'tycho-nooa-vision-readback-frames-v1':
    raise RuntimeError('pinned frame asset has an unexpected schema')
if len(frames['frames']) != 3:
    raise RuntimeError('expected exactly three pinned ls20 frames')

output_dir = Path('/kaggle/working/tycho-nooa-vision-readback-v16')
command = [
    sys.executable,
    '/tmp/vision_outer_runtime.py',
    '--output-dir', str(output_dir),
    '--manifest', '/tmp/PROTOTYPE-qwen3_8_fp8-manifest.json',
    '--input-root', '/kaggle/input',
    '--model-source', 'jakobbrggen/qwen3-8-27b-fp8-hf-snapshot/versions/1',
    '--runtime-provenance', str(runtime_provenance),
    '--startup-timeout-s', '1200',
    '--request-timeout-s', '180',
    '--attention-backend', 'triton',
    '--sampling-backend', 'pytorch',
    '--profile-imports',
    '--r8-l3b',
    '--vision-readback-probe', '/tmp/vision_readback_probe.py',
    '--vision-frames', '/tmp/vision-readback-frames.json',
    '--vision-probe-timeout-s', '5400',
    '--vision-request-timeout-s', '600',
]
subprocess.run(command, check=True)

result = json.loads((output_dir / 'result.json').read_text())
print(json.dumps(result['vision_readback']['summary'], indent=1, sort_keys=True))
print(json.dumps(
    result['vision_readback']['visual_token_measurement'], indent=1, sort_keys=True
))
print(json.dumps(result['request_count_facts'], indent=1, sort_keys=True))
print(result['claim_ceiling'])
"""

    notebook["metadata"]["kaggle"]["isInternetEnabled"] = False
    notebook["metadata"]["kaggle"]["accelerator"] = "nvidiaRtx6000"
    return notebook


def validate(notebook: dict[str, Any]) -> None:
    """Fail the build, not the GPU run, on anything checkable from here."""
    if len(notebook["cells"]) != 4:
        raise RuntimeError("the frozen candidate is four cells")
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"notebook-cell-{index}", "exec")

    assets = "".join(notebook["cells"][1]["source"])
    for destination in (
        "/tmp/vision_outer_runtime.py",
        "/tmp/vision_readback_probe.py",
        "/tmp/vision-readback-frames.json",
    ):
        if destination not in assets:
            raise RuntimeError(f"asset cell does not write {destination}")

    run_cell = "".join(notebook["cells"][3]["source"])
    for required in ("--r8-l3b", "--vision-readback-probe", "KAGGLE_IS_COMPETITION_RERUN"):
        if required not in run_cell:
            raise RuntimeError(f"run cell lost {required}")
    for forbidden in ("submission.parquet", "competitions submit", "TYCHO_PUBLIC_GAME_IDS"):
        if forbidden in run_cell:
            raise RuntimeError(f"run cell must not contain {forbidden}")

    install_cell = "".join(notebook["cells"][2]["source"])
    if "wheelhouse-manifest.json" in install_cell:
        raise RuntimeError("V16 must not carry the Solver wheelhouse install")


def main() -> None:
    notebook = build()
    validate(notebook)
    OUTPUT.write_text(json.dumps(notebook, indent=1) + "\n")
    digests = {
        path.name: sha256(path) for path in (RUNTIME, PROBE, FRAMES, OUTPUT)
    }
    (HERE / "candidate-hashes.preupload.sha256").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(digests.items()))
    )
    print(json.dumps(digests, indent=1, sort_keys=True))


if __name__ == "__main__":
    main()
