"""Replay process qualification from the delivered notebook's embedded ZIP."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import tempfile

from build_package import HERE, NOTEBOOK, ROOT, build, sha

PROBES = ROOT / "analysis/deployment-probes/20260905-request-ownership"
sys.path.insert(0, str(PROBES))
from worker_probe import Trial, extract, repeated_polls  # noqa: E402


async def multicall(trial):
    trial.response_codes[0] = 'print("Continue the live decision before submitting.")'
    await trial.turn()
    await trial.until(lambda: len(trial.admissions) == 1, "first live-decision call")
    trial.release[0].set()
    await trial.until(
        lambda: len(trial.admissions) == 2, "second call within same live handle"
    )
    assert not (trial.path / "ipc/actions.jsonl").read_text().strip()
    assert not any(e["event"] == "handle_returned" for e in trial.traces())
    trial.release[1].set()
    answer = await trial.receive("decision")
    assert answer["decision"]["action"] == "UP"
    await trial.until(
        lambda: any(
            e["event"] == "stale_state_notification_skipped" for e in trial.traces()
        ),
        "post-submit stale state skipped",
    )
    assert len(trial.admissions) == 2
    assert len((trial.path / "ipc/actions.jsonl").read_text().splitlines()) == 1
    return "pass: two legitimate model calls before one submission; no stale state admission"


async def main():
    for path, data in build().items():
        assert path.read_bytes() == data, f"package drift {path}"
    manifest = json.loads((HERE / "package-manifest.json").read_text())
    output = Path(tempfile.mkdtemp(prefix="package-qualification-", dir=PROBES))
    source = output / "embedded-execution"
    source.mkdir()
    assert extract(source, NOTEBOOK) == manifest["packaged_sources"]
    results = []
    for name, kind in [
        ("repeated-polls-and-successor", "poll"),
        ("receive-gap-identical-grid-successor", "gap"),
        ("multi-call-live-decision", "multi"),
    ]:
        trial = Trial(output / name, source, name)
        verdict = "unresolved"
        try:
            await trial.start()
            if kind == "multi":
                verdict = await multicall(trial)
            else:
                verdict = await repeated_polls(
                    trial,
                    candidate=True,
                    defer_receive=kind == "gap",
                    successor_value=0 if kind == "gap" else 1,
                )
            loaded = [e for e in trial.traces() if e["event"] == "loaded_source"]
            for basename in ["headless.py", "nooa_worker.py"]:
                records = [e for e in loaded if e["basename"] == basename]
                assert len(records) == 1, records
                assert (
                    records[0]["source_sha256"]
                    == manifest["packaged_sources"][basename]
                )
                assert records[0]["code_matches_compiled_source"]
                assert Path(records[0]["path"]).parent == source
            results.append(
                {
                    "case": name,
                    "verdict": verdict,
                    "http_admissions": len(trial.admissions),
                    "loaded_sources": loaded,
                }
            )
        finally:
            await trial.stop()
            trial.save(verdict)
        print(json.dumps(results[-1]), flush=True)
    result = {
        "status": "LOCAL_PACKAGE_QUALIFIED",
        "notebook_sha256": sha(NOTEBOOK.read_bytes()),
        "package_manifest_sha256": sha((HERE / "package-manifest.json").read_bytes()),
        "execution_root": str(source),
        "results": results,
        "limits": "local dependencies; no GPU, integrated gameplay, supersession, interrupted termination or scheduler-retirement qualification",
    }
    (output / "VERDICT.json").write_text(json.dumps(result, indent=2) + "\n")
    (HERE / "qualification.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"verdict": str(output / "VERDICT.json")}))


if __name__ == "__main__":
    asyncio.run(main())
