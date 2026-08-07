from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from game_judgment import prepare_requests, prepared_bytes  # noqa: E402
from game_judgment.experiment import (  # noqa: E402
    ExperimentRunner,
    HTTPTransport,
    ReplayTransport,
    RunError,
    audit_openai_compatible_endpoint,
    audit_openrouter_endpoint,
)
from game_judgment.serialization import digest  # noqa: E402


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


def _api_key(names: list[str]) -> str:
    return next((os.environ[name] for name in names if os.environ.get(name)), "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "runtime-config.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--replay-responses", type=Path)
    parser.add_argument("--authorize-provider-calls", action="store_true")
    parser.add_argument(
        "--phase", choices=("ladder", "perturbation", "full"), default="full"
    )
    parser.add_argument("--only-arm", choices=tuple("ABCDEFGH"))
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    _write(args.output / "config.json", config)
    _write(args.output / "manifest.json", {"config_sha256": digest(config), "batch_sha256": digest(json.loads((ROOT / "fixtures/batch.json").read_text())), "environment_actions": 0})

    if args.prepare_only:
        rows = prepare_requests(ROOT / "fixtures/batch.json", ROOT, config)
        (args.output / "requests.jsonl").write_bytes(prepared_bytes(rows))
        _write(args.output / "result.json", {"status": "PREPARED", "requests": len(rows), "provider_calls": 0, "environment_actions": 0})
        return 0

    transfer_transport = None
    transfer_config = None
    if args.replay_responses:
        primary = ReplayTransport.load(args.replay_responses)
        transfer_transport = primary
        transport_mode = "replay"
    else:
        transport_mode = "provider"
        if not args.authorize_provider_calls:
            raise RunError("live execution requires --authorize-provider-calls")
        key = _api_key(config["api_key_environment"])
        if not key:
            raise RunError("no primary provider API key is configured")
        primary_audit = audit_openrouter_endpoint(config)
        _write(args.output / "provider-audit-primary.json", primary_audit)
        transport_config = config["transport"]
        primary = HTTPTransport(
            config["endpoint"],
            key,
            int(config["request_timeout_seconds"]),
            float(transport_config["minimum_request_interval_seconds"]),
            int(transport_config["max_attempts"]),
            float(transport_config["retry_backoff_seconds"]),
            tuple(int(status) for status in transport_config["retry_http_statuses"]),
        )

        if args.phase == "full" and config["transfer"]["required"]:
            endpoint = os.environ.get(config["transfer"]["endpoint_environment"])
            if not endpoint:
                raise RunError(f"required E4B endpoint environment {config['transfer']['endpoint_environment']} is missing")
            transfer_audit = audit_openai_compatible_endpoint(endpoint, config["transfer"]["model_audit_path"], config["transfer"]["model"])
            _write(args.output / "provider-audit-transfer.json", transfer_audit)
            transfer_key = os.environ.get(config["transfer"]["api_key_environment"], "")
            transfer_transport = HTTPTransport(
                endpoint.rstrip("/") + "/v1/chat/completions",
                transfer_key,
                int(config["request_timeout_seconds"]),
                float(transport_config["minimum_request_interval_seconds"]),
                int(transport_config["max_attempts"]),
                float(transport_config["retry_backoff_seconds"]),
                tuple(int(status) for status in transport_config["retry_http_statuses"]),
            )

    runner = ExperimentRunner(ROOT, args.output, config, primary, transfer_transport)
    try:
        arm = runner.run_ladder((args.only_arm,) if args.only_arm else tuple("ABCDEFGH"))
        if args.phase == "perturbation":
            perturbation = runner.run_perturbation(arm)
            if not perturbation.get("eligible") or not perturbation.get("pass"):
                raise RunError("reconstruction-content perturbation failed or was ineligible")
        elif args.phase == "full":
            thinking = runner.run_thinking_sequence(arm)
            if not thinking["first_off_equals_last_off"] or not thinking["all_contract_pass"]:
                raise RunError("thinking off-on-off sequence failed its stability gate")
            perturbation = runner.run_perturbation(arm)
            if not perturbation.get("eligible") or not perturbation.get("pass"):
                raise RunError("reconstruction-content perturbation failed or was ineligible")
            transfer_config = deepcopy(config)
            transfer_config["model"] = config["transfer"]["model"]
            transfer_config.pop("provider", None)
            transfer = runner.run_transfer(arm, transfer_config)
            if not transfer["stable_success"]:
                raise RunError("E4B transfer failed repeated signature stability")
        runner.finish(
            "PASS_REPLAY" if transport_mode == "replay" else "PASS",
            transport_mode=transport_mode,
        )
        return 0
    except Exception as exc:
        runner.finish("FAIL", str(exc), transport_mode=transport_mode)
        if isinstance(exc, (RunError, ValueError)):
            print(str(exc), file=sys.stderr)
            return 1
        raise


if __name__ == "__main__":
    raise SystemExit(main())
