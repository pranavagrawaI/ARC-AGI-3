"""Host broker owns canonical state, paid transport and immutable evidence.

The current Actor executes inside a fresh bubblewrap mount/PID/network namespace.
Only its own scratch directory is writable. Its JSON pipe is the only host route.
"""

import json
import os
import selectors
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

from bootstrap import HERE, RUNTIME, ROOT
from experiment import catalog, evidence_rows
from theory import Store, IntegrityError, replay, write
from transport import Stop, BudgetEnd

WORKER_FILES = (
    "bootstrap.py",
    "actor.py",
    "client.py",
    "experiment.py",
    "theory.py",
    "transport.py",
    "request_contract.py",
    "protocol.py",
    "local_wire.py",
    "worker.py",
    "operational-theory.schema.json",
    "workflow-skill.md",
)


def command(stage, scratch):
    venv = ROOT / "deployment/kaggle/.venv"
    python_home = Path(sys.executable).resolve().parent.parent
    return [
        "bwrap",
        "--unshare-all",
        "--die-with-parent",
        "--new-session",
        "--ro-bind",
        "/usr",
        "/usr",
        "--ro-bind",
        "/lib",
        "/lib",
        "--ro-bind",
        "/lib64",
        "/lib64",
        "--proc",
        "/proc",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        str(python_home),
        str(python_home),
        "--ro-bind",
        str(python_home),
        str(Path(os.readlink(venv / "bin/python")).parent.parent),
        "--ro-bind",
        str(venv),
        "/venv",
        "--ro-bind",
        str(stage),
        "/package",
        "--ro-bind",
        str(stage / "chassis"),
        "/runtime/chassis",
        "--bind",
        str(scratch),
        "/work",
        "--chdir",
        "/work",
        "--clearenv",
        "--setenv",
        "PATH",
        "/venv/bin:/usr/bin",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "OT_RUNTIME",
        "/runtime",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "ARC_DISABLE_DOTENV",
        "1",
        "--setenv",
        "ARC_DISCOVERY_ARM",
        "a",
        "--setenv",
        "ARC_CONTINGENT_ARM",
        "off",
        "--setenv",
        "ARC_STALL_REVIEW",
        "0",
        "--setenv",
        "OPENBLAS_NUM_THREADS",
        "1",
        "--setenv",
        "LITELLM_LOCAL_MODEL_COST_MAP",
        "True",
        "/venv/bin/python",
        "-B",
        "/package/worker.py",
    ]


def summarize(output, arm, store, proposals, elapsed):
    if store:
        replayed = replay(store.root, store.evidence, store.artifacts)
        write(output / "REPLAY.json", replayed)
        completed = store.root / "completed.jsonl"
        cycles = (
            [json.loads(s) for s in completed.read_text().splitlines()]
            if completed.exists()
            else []
        )
        decisions = replayed["decisions"]
        metrics = dict(
            construction_completion=bool(cycles and len(proposals) == 2),
            evidence_entered=sorted(
                {e["id"] for row in store.rows for e in row["theory"]["evidence_refs"]}
            ),
            contradiction_driven_revisions=sum(
                r["reason"] == "counterevidence"
                for row in store.rows
                for r in row["theory"]["revisions"]
            ),
            active_uncertainty_or_procedure=[
                d["policy"]["selection"] for d in decisions
            ],
            declared_modes=[d["policy"]["mode"] for d in decisions],
            prospective_decision_completeness=len(decisions),
            canonical_action_binding=bool(decisions)
            and all(d["actions"] == d["policy"]["actions"] for d in decisions),
            post_observation_same_lineage=bool(cycles),
            replayability=bool(decisions),
        )
    else:
        metrics = dict(
            construction_completion=len(proposals) == 2,
            canonical_metrics="not applicable; native control artifacts require the predeclared review rubric",
        )
    result = dict(
        arm=arm,
        metrics=metrics,
        proposals=proposals,
        elapsed_seconds=elapsed,
        gameplay="not run",
        truth="not evaluated",
        canonical_causation="reference/action binding only",
    )
    write(output / "RESULT.json", result)
    return result


def run_arm(arm, output, transport, *, wall_seconds=240):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scratch, stage = output / "actor", output / "worker-source"
    scratch.mkdir()
    stage.mkdir()
    shutil.copytree(
        RUNTIME / "chassis",
        stage / "chassis",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in WORKER_FILES:
        shutil.copyfile(HERE / name, stage / name)
    rows = evidence_rows()
    write(output / "EVIDENCE.json", rows[:1])
    store = (
        Store(output / "canonical", catalog(rows[:1])) if arm == "treatment" else None
    )
    proposals = []
    started = time.monotonic()
    deadline = started + wall_seconds
    worker = None
    complete = False
    rpc_count = 0
    exhausted = False
    prepared_digests = set()
    previous_handler = signal.getsignal(signal.SIGALRM)

    def arm_cutoff(*_):
        raise Stop("arm absolute deadline")

    signal.signal(signal.SIGALRM, arm_cutoff)
    signal.setitimer(signal.ITIMER_REAL, wall_seconds)
    try:
        with (output / "worker.stderr").open("xb") as errors:
            worker = subprocess.Popen(
                command(stage.resolve(), scratch.resolve()),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=errors,
                start_new_session=True,
                env={"PATH": "/usr/bin:/bin"},
                bufsize=0,
            )
            selector = selectors.DefaultSelector()
            selector.register(worker.stdout, selectors.EVENT_READ)
            buffer = b""
            while not complete:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise Stop("arm absolute deadline")
                part = os.read(worker.stdout.fileno(), 65536)
                if not part:
                    raise Stop("worker exited before completion")
                buffer += part
                if len(buffer) > 2_000_000:
                    raise Stop("oversize worker protocol")
                while b"\n" in buffer and not complete:
                    line, buffer = buffer.split(b"\n", 1)
                    request = json.loads(line)
                    rpc_count += 1
                    if rpc_count > 120:
                        complete, exhausted = True, True
                        break
                    if set(request) != {"op", "payload"}:
                        raise IntegrityError("worker protocol fields")
                    op, payload = request["op"], request["payload"]
                    if op == "initial":
                        value = {"arm": arm, "rows": rows[:1]}
                    elif op == "model":
                        if store and store.errors > 3:
                            complete, exhausted = True, True
                            break
                        try:
                            value = transport(payload)
                        except BudgetEnd:
                            complete, exhausted = True, True
                            break
                    elif op == "theory" and store:
                        if (
                            payload["operation"] == "result"
                            and isinstance(payload["payload"], dict)
                            and payload["payload"].get("relation")
                            != "recorded_continuation_not_action_outcome"
                        ):
                            value = store.feedback(
                                "only recorded_continuation_not_action_outcome is licensed in this experiment"
                            )
                        else:
                            value = store.call(payload["operation"], payload["payload"])
                    elif op == "feedback" and store:
                        value = store.feedback(payload)
                    elif op == "artifact" and store:
                        value = store.register(payload["filename"], payload["source"])
                    elif op == "proposed":
                        if len(proposals) >= 2:
                            raise IntegrityError("too many proposed turns")
                        if store and (
                            store.phase != "prepared"
                            or payload["actions"] != store.pending["actions"]
                        ):
                            raise IntegrityError(
                                "proposal lacks active canonical preparation"
                            )
                        if store:
                            identity = store.pending["theory_sha256"]
                            if identity in prepared_digests or (
                                proposals
                                and not (store.root / "completed.jsonl").exists()
                            ):
                                raise IntegrityError(
                                    "stale proposal or skipped result reconciliation"
                                )
                            prepared_digests.add(identity)
                        proposals.append(payload)
                        write(output / "EVIDENCE.json", rows)
                        if store:
                            store.evidence = catalog(rows)
                        value = {"rows": rows}
                    elif op == "incomplete":
                        value = None
                    elif op == "done":
                        complete, value = True, None
                    else:
                        raise IntegrityError("unrecognized or cross-arm host operation")
                    with (output / "host-events.jsonl").open("a") as f:
                        event = {
                            "ordinal": rpc_count,
                            "op": op,
                            "request": payload,
                            "result": value,
                        }
                        # Raw model wire has its own durable receipts; avoid another large transcript.
                        if op == "model":
                            event = {
                                "ordinal": rpc_count,
                                "op": op,
                                "receipt_owned_by": "transport",
                            }
                        f.write(
                            json.dumps(event, sort_keys=True, allow_nan=False) + "\n"
                        )
                    worker.stdin.write(
                        (json.dumps({"value": value}, allow_nan=False) + "\n").encode()
                    )
                    worker.stdin.flush()
            if complete:
                # Normal completion exits immediately; feedback exhaustion is killed below.
                if not exhausted:
                    worker.wait(timeout=min(5, max(0.001, deadline - time.monotonic())))
                    if worker.returncode:
                        raise Stop("worker unsuccessful exit")
            if time.monotonic() > deadline:
                raise Stop("late completion")
            return summarize(output, arm, store, proposals, time.monotonic() - started)
    except BaseException as exc:
        write(
            output / "FAILURE.json",
            {
                "error_type": type(exc).__name__,
                "reason": str(exc)
                if isinstance(exc, (Stop, IntegrityError))
                else "local failure",
                "elapsed_seconds": time.monotonic() - started,
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if worker:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait()
            write(
                output / "SUPERVISION.json",
                {
                    "returncode": worker.returncode,
                    "reaped": True,
                    "elapsed_seconds": time.monotonic() - started,
                    "isolation": "bubblewrap mount,PID,network namespaces; sterile env; pipe-only broker",
                },
            )
