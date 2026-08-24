"""Preflight or execute the exact paid V13 Gemma runway diagnostic."""
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path
from tycho_nooa.codeact_runway import execute_authorized_codeact_runway, preflight_codeact_runway
from tycho_nooa.compatibility_manifest import sha256_file
from tycho_nooa.llm import build_llm
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
MANIFEST=HERE/"experiment.json"
AUTHORIZATION=HERE/"authorization.json"

def client():
    llm,lane=build_llm()
    if lane != "live": raise RuntimeError("paid diagnostic requires a live client")
    return llm

async def main(preflight_only: bool):
    llm=client()
    if preflight_only:
        manifest,gate,authorization,authorization_hash=preflight_codeact_runway(MANIFEST,AUTHORIZATION,repository_root=ROOT,llm=llm)
        print(json.dumps({"status":"preflight_pass","experiment_manifest_sha256":sha256_file(MANIFEST),"authorization_sha256":authorization_hash,"authorization_id":authorization.authorization_id,"gate_id":gate.task.gate_id,"maximum_calls":authorization.maximum_calls,"maximum_spend_usd":str(authorization.maximum_spend_usd),"output_root":manifest.output_root},indent=2,sort_keys=True))
        return
    raw=json.loads(MANIFEST.read_text())
    await execute_authorized_codeact_runway(MANIFEST,AUTHORIZATION,repository_root=ROOT,destination=ROOT/raw["output_root"],llm=llm)

if __name__ == "__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--preflight",action="store_true"); args=parser.parse_args(); asyncio.run(main(args.preflight))
