"""Fetch three hash-committed checkpoints; restore isolated exact-call packets."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
READOUT = HERE.parents[1] / "evidence/readout-20261007"
SELECTED = (("re86", 70, "re86-stall"), ("ls20", 75, "ls20-stall"), ("re86", 36, "re86-success"))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def load_restore():
    source = READOUT / "frozen-inspection/lean_prune/tycho_nooa/decision_checkpoints.py"
    spec = importlib.util.spec_from_file_location("packet_checkpoints", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(destination=HERE / "packets", *, fetch=False, workers=2):
    if workers not in (1, 2):
        raise ValueError("Retrieval uses at most two workers")
    index = json.loads((READOUT / "lean-prune/tycho-dfranzen/evidence-index.json").read_text())
    indexed = {r["path"]: r for r in index["files"]}
    cache = HERE / "downloads"
    cache.mkdir(exist_ok=True)
    receipts, lock, local = [], threading.Lock(), threading.local()

    def retrieve(name):
        expected = indexed[name]["sha256"]
        target = cache / name
        prior = READOUT / "lean-prune" / name
        if target.exists():
            raw = target.read_bytes()
        elif prior.exists():
            raw = prior.read_bytes()
        else:
            if not fetch:
                raise FileNotFoundError(name + "; pass --fetch to retrieve this committed output")
            if not hasattr(local, "client"):
                sys.path.insert(0, str(HERE.parent))
                from collect_outputs import KaggleOutput
                local.client = KaggleOutput("pranavagrawai/tycho-dfranzen-wave-2-lean-prune", 1)
            for attempt in range(6):
                try:
                    time.sleep(1)
                    raw = local.client.fetch(name)
                    break
                except Exception as error:
                    response = getattr(error, "response", None)
                    code = getattr(response, "status_code", getattr(error, "status", None))
                    if code != 429 or attempt == 5:
                        raise RuntimeError(f"Retrieval failed for {name}: {type(error).__name__}") from None
                    headers = getattr(response, "headers", {}) or {}
                    delay = max(60, int(headers.get("Retry-After", 60)))
                    print(json.dumps({"phase": "rate_limit", "wait_seconds": delay}), flush=True)
                    time.sleep(delay)
        if digest(raw) != expected:
            raise ValueError("Evidence hash mismatch: " + name)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        with lock:
            receipts.append({"path": name, "sha256": expected, "bytes": len(raw)})
        return target

    packet_rows = []
    first = []
    for game, call, label in SELECTED:
        req = next(n for n in indexed if f"/{game}-" in n and n.endswith(f"/calls/{call:06d}/request.json"))
        cp = req.split("/calls/")[0]
        first.extend((req, req.replace("request.json", "workspace.json")))
        packet_rows.append((game, call, label, cp))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(retrieve, first))
        blob_sources = {}
        for game, call, label, cp in packet_rows:
            manifest = json.loads((cache / cp / "calls" / f"{call:06d}" / "workspace.json").read_text())
            for entry in manifest["files"]:
                blob_sources.setdefault(entry["sha256"], cp + "/blobs/sha256/" + entry["sha256"])
        print(json.dumps({"phase": "manifests", "unique_blobs": len(blob_sources), "packets": len(packet_rows)}), flush=True)
        paths = list(pool.map(retrieve, blob_sources.values()))
    blobs = {p.name: p for p in paths}
    checkpoint = load_restore()
    results = []
    for game, call, label, cp in packet_rows:
        packet = destination / label
        if (packet / "manifest.json").exists():
            results.append(json.loads((packet / "manifest.json").read_text()))
            continue
        request = json.loads((cache / cp / "calls" / f"{call:06d}" / "request.json").read_text())
        workspace_manifest = json.loads((cache / cp / "calls" / f"{call:06d}" / "workspace.json").read_text())
        for entry in workspace_manifest["files"]:
            target = cache / cp / "blobs/sha256" / entry["sha256"]
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                os.link(blobs[entry["sha256"]], target)
        packet.mkdir(parents=True, exist_ok=True)
        checkpoint.restore_workspace(cache / cp, call_index=call, destination=packet / "workspace")
        shutil.copy2(cache / cp / "calls" / f"{call:06d}" / "request.json", packet / "request.json")
        shutil.copy2(cache / cp / "calls" / f"{call:06d}" / "workspace.json", packet / "workspace.json")
        host_path = READOUT / "lean-prune" / cp.rsplit("/decision-checkpoints", 1)[0] / "host-actions.jsonl"
        host = [json.loads(line) for line in host_path.read_text().splitlines()]
        point = request["decision_point"]
        sequence = point["sequence"]
        prefix = [row for row in host if row["sequence"] < sequence]
        if not prefix or prefix[-1]["sequence"] != sequence - 1:
            raise ValueError("Incomplete host action prefix")
        last = prefix[-1]
        obs = last["after"]
        turn = {"grid": obs["decision_frame"], "available_actions": obs["available_actions"],
                "sequence": sequence, "level": point["level"], "turn_in_level": point["turn_in_level"],
                "levels_completed": obs["levels_completed"], "state": obs["state"],
                "previous_action": last["decision"]["action"], "previous_row": last["decision"].get("row"),
                "previous_col": last["decision"].get("col")}
        write(packet / "turn.json", turn)
        (packet / "host-prefix.jsonl").write_text("".join(json.dumps(x) + "\n" for x in prefix))
        game_id = cp.split("/hybrid-games/")[1].split("/", 1)[0].rsplit("-", 1)[0]
        record = {"schema": "tycho-exact-call-branch-v1", "label": label, "game_id": game_id,
                  "call_index": call, "decision_point": point,
                  "request_payload_commitment": request["request_payload_commitment"],
                  "workspace_commitment": request["workspace_commitment"],
                  "workspace_bytes": sum(x["size"] for x in workspace_manifest["files"]),
                  "workspace_files": len(workspace_manifest["files"]), "source_kernel_version": 1,
                  "source_checkpoint": cp, "claim_ceiling": "Exact saved request boundary and filesystem; no hidden environment state or NOOA event graph restored."}
        write(packet / "manifest.json", record)
        results.append(record)
    write(HERE / "packet-preparation.json", {"packets": results, "retrieval": receipts, "unique_blob_bytes": sum(p.stat().st_size for p in blobs.values())})
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--destination", type=Path, default=HERE / "packets")
    args = parser.parse_args()
    print(json.dumps(prepare(args.destination, fetch=args.fetch, workers=args.workers), indent=2))
